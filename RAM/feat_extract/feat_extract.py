import argparse
import os
import re
import glob
import logging
import numpy as np
import pandas as pd
from PIL import Image
from natsort import natsorted
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from src.model import FeatureExtractor  # 確保此路徑與你的專案結構一致

device = torch.device("cuda")

# ResNet50 標準輸入轉換 (不含縮放，縮放交給 PIL 處理)
transform = transforms.Compose([
    transforms.ToTensor(),
])

def log_information(vid_id, frame_id, track_id, e):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('./configs/Feature_Extraction_Error.log')
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
    logger.info(f"Error: {e} ------ Video: {vid_id} -- Frame: {frame_id} -- TrackID: {track_id}")

def get_args():
    parser = argparse.ArgumentParser()
    # 規則 5：分開儲存的雙根目錄設定
    parser.add_argument("--csv_rgb_root", default="/home/wayne/Documents/MMAU")
    parser.add_argument("--flow_npz_root", default="/media/wayne/27CD255760735841/MMAU/")
    args = parser.parse_args()
    return args

def featureExt(image, extractor):
    """
    高效特徵提取：複用外部傳入的預訓練權重，避免重複初始化
    """
    # 確保輸入是 224x224 尺寸
    if image.size != (224, 224):
        image = image.resize((224, 224), Image.BILINEAR)
        
    image_tensor = transform(image)
    image_tensor = torch.unsqueeze(image_tensor, 0).float().to(device=device)
    
    with torch.no_grad():
        feat = extractor(image_tensor)  # 輸出 1 x 2048 x 1 x 1
        feat = torch.squeeze(feat, 2)   # 1 x 2048 x 1
        feat = torch.squeeze(feat, 2)   # 1 x 2048
    return feat

def get_frame_file(folder_path, frame_num):
    """
    安全跨格式影格搜尋器：自動無視 4 位數或 6 位數補零以及副檔名落差
    """
    if not os.path.exists(folder_path):
        return None
    for f in os.listdir(folder_path):
        try:
            if int(os.path.splitext(f)[0]) == frame_num:
                return os.path.join(folder_path, f)
        except ValueError:
            continue
    return None

def main():
    args = get_args()
    
    # 1. 在主線程僅初始化一次 FeatureExtractor 權重
    print(f"正在載入預訓練 FeatureExtractor 骨幹網路至 {device}...")
    extractor = FeatureExtractor().to(device=device)
    extractor.eval()
    
    # 2. 遞迴搜尋內部所有先前生成的時間切片 CSV 檔案
    print("正在掃描基準測試切片 CSV 檔案...")
    csv_pattern = os.path.join(args.csv_rgb_root, "**", "detections_*.csv")
    csv_files = natsorted(glob.glob(csv_pattern, recursive=True))
    print(f"共計找到 {len(csv_files)} 個追蹤片段待處理。")

    for csv_file in tqdm(csv_files, desc="Feature Extraction"):
        # 解析當前 CSV 的檔名特徵
        csv_basename = os.path.basename(csv_file)
        match = re.search(r'detections_(\d+)_(\d+)\.csv', csv_basename)
        if not match:
            continue
            
        start_frame = int(match.group(1))
        end_frame = int(match.group(2))
        num_frames = end_frame - start_frame + 1  # 動態自適應影格長度

        # 解析相對路徑與 Video ID (例如: CAP-DATA/1/001537)
        rel_dir = os.path.relpath(os.path.dirname(csv_file), args.csv_rgb_root)
        vid_id = os.path.basename(rel_dir)
        
        # 建立對應的實體路徑
        rgb_image_dir = os.path.join(args.csv_rgb_root, rel_dir, "images")
        flow_image_dir = os.path.join(args.flow_npz_root, rel_dir, f"flow_{start_frame}_{end_frame}")
        output_npz_dir = os.path.join(args.flow_npz_root, rel_dir)
        os.makedirs(output_npz_dir, exist_ok=True)

        # 初始化動態長度的儲存矩陣
        detections = np.zeros((num_frames, 30, 6), dtype=np.float32)
        feature = np.zeros((num_frames, 31, 2048), dtype=np.float32)

        # 暫存當前短片已解鎖去墊的光流 PIL 物件，避免二次讀取硬碟
        flow_images_cleaned = {}

        # 步驟 A：提取全域光流特徵 (Global Frame-level) 與動態去墊處理
        for t in range(num_frames):
            current_frame_num = start_frame + t
            
            rgb_path = get_frame_file(rgb_image_dir, current_frame_num)
            flow_path = get_frame_file(flow_image_dir, current_frame_num)

            if rgb_path is None or flow_path is None:
                continue

            # 讀取影像物件
            rgb_img = Image.open(rgb_path).convert("RGB")
            flow_img = Image.open(flow_path).convert("RGB")

            # 規則 1 & 2：動態逆向去墊 (Center Unpadding) 核心算法
            W_flow, H_flow = flow_img.size
            W_rgb, H_rgb = rgb_img.size
            
            pad_w = W_flow - W_rgb
            pad_h = H_flow - H_rgb

            if pad_w < 0 or pad_h < 0:
                raise ValueError(
                    f"Flow image {flow_path} ({W_flow},{H_flow}) is smaller than RGB {rgb_path} ({W_rgb},{H_rgb})"
    )
            left = pad_w // 2
            top = pad_h // 2
            
            # 精準裁切回原始解析度畫布
            flow_img_clean = flow_img.crop((left, top, left + W_rgb, top + H_rgb))
            flow_images_cleaned[current_frame_num] = flow_img_clean

            # 提取全域光流特徵並存入 index 0
            global_feat = featureExt(flow_img_clean, extractor)
            feature[t, 0, :] = global_feat.detach().cpu().numpy()

        # 步驟 B：提取物件區域光流特徵 (Object ROI-level) 
        df = pd.read_csv(csv_file)
        
        # 依據影格進行群組，動態生成 object_num (0~29)
        for frame_val, group in df.groupby('frame'):
            frame_num = int(frame_val)
            t = frame_num - start_frame
            
            # 邊界安全檢查
            if t < 0 or t >= num_frames or frame_num not in flow_images_cleaned:
                continue
                
            flow_img_clean = flow_images_cleaned[frame_num]
            W_rgb, H_rgb = flow_img_clean.size

            for obj_idx, (_, row) in enumerate(group.iterrows()):
                if obj_idx >= 30:  # 限制最大物件收容量
                    break

                track_id = int(row['track_id'])
                x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])

                # 邊界極值溢出過濾
                x1, x2 = max(0, min(x1, W_rgb)), max(0, min(x2, W_rgb))
                y1, y2 = max(0, min(y1, H_rgb)), max(0, min(y2, H_rgb))
                
                if x2 <= x1 or y2 <= y1:
                    continue

                # 規則 4：將風險標籤 (第 6 欄位) 預設強制全歸零，供後續大語言模型複寫
                detections[t, obj_idx] = [track_id, x1, y1, x2, y2, 0.0]

                # 核心優化：直接從去墊的光流圖上依原圖 BBox 裁切局部區塊
                roi_img = flow_img_clean.crop((x1, y1, x2, y2))
                
                try:
                    # 提取物件級光流特徵並存入 index (obj_idx + 1)
                    object_feat = featureExt(roi_img, extractor)
                    object_feat_np = object_feat.detach().cpu().numpy()
                    feature[t, obj_idx + 1, :] = object_feat_np
                except Exception as e:
                    print(f"物件特徵提取錯誤: Video {vid_id}, Frame {frame_num}")
                    log_information(vid_id, frame_num, track_id, e)
                    continue

        # 儲存壓縮的 .npz 檔案 (加註時間尾碼防止不同變長片段覆蓋)
        # 註：RiskyObject 內部並未使用 toa，此處遵循 Dataloader 規範填入 0 作為相容常數
        save_file = os.path.join(output_npz_dir, f"clip_{start_frame}_{end_frame}.npz")
        np.savez_compressed(
            save_file, 
            feature=feature,
            detection=detections, 
            vid_id=vid_id, 
            toa=0
        )

    print("🎉 所有基準測試切片的 .npz 特徵檔案已全數對齊並安全產出！")

if __name__ == '__main__':
    main()