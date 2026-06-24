import argparse
import os
import cv2
import glob
import numpy as np
import pandas as pd  # 補上 pandas 載入
import torch
from tqdm import tqdm
from PIL import Image
from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

ROOT_DIR = "/home/wayne/Documents/MMAU"
DEVICE = 'cuda'

def parse_benchmark_txt(txt_path):
    """
    解析 Benchmark 的 .txt 檔案，轉換為 DataFrame 導航表
    """
    parsed_data = []
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"找不到基準測試檔案: {txt_path}")
        
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ')
            if len(parts) >= 5:
                toa_clean = parts[4].split(',')[0]
                parsed_data.append([
                    parts[0],       # video_id (例如 "1/001537" 或 "1/001")
                    int(parts[1]),  # label 
                    int(parts[2]),  # start_frame
                    int(parts[3]),  # end_frame
                    float(toa_clean)# toa
                ])
                
    return pd.DataFrame(
        parsed_data, 
        columns=['video_id', 'label', 'start_frame', 'end_frame', 'toa']
    )

def process_clip(model, image_dir, flow_output_dir, start_frame, end_frame):
    """
    針對特定時間切片進行 RAFT 光流圖推論與儲存
    """
    all_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    clip_image_paths = []
    clip_frame_ids = []
    ext = ".jpg"  # 預設副檔名預留
    
    for fname in all_files:
        try:
            frame_num = int(os.path.splitext(fname)[0])
        except ValueError:
            continue
            
        if start_frame <= frame_num <= end_frame:
            clip_image_paths.append(os.path.join(image_dir, fname))
            clip_frame_ids.append(os.path.splitext(fname)[0])
            ext = os.path.splitext(fname)[1]

    # 光流推論至少需要兩張影格
    if len(clip_image_paths) < 2:
        return

    last_flow_bgr = None

    with torch.no_grad():
        # 時序成對推進：(Frame 1, Frame 2), (Frame 2, Frame 3) ...
        for i in range(len(clip_image_paths) - 1):
            imfile1 = clip_image_paths[i]
            imfile2 = clip_image_paths[i+1]
            frame_id = clip_frame_ids[i]
            
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            # RAFT 尺寸 Padding 分解
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # 執行 RAFT 光流估計
            _, flow_up = model(image1, image2, iters=20, test_mode=True)
            
            # 將 2D 位移矩陣轉換為 RGB 視覺化圖片 (H, W, 3)
            flow_img = flow_up[0].permute(1, 2, 0).cpu().numpy()
            flow_rgb = flow_viz.flow_to_image(flow_img)
            
            # 關鍵修正：將 RGB 轉換為 OpenCV 預期的 BGR，防止存檔顏色錯亂
            flow_bgr = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR)
            
            # 儲存當前影格的光流結果
            out_path = os.path.join(flow_output_dir, f"{frame_id}{ext}")
            cv2.imwrite(out_path, flow_bgr)
            
            # 暫存當前光流，供最後一幀 Padding 使用
            last_flow_bgr = flow_bgr

        # 🔥 修正差一錯誤 (Fencepost Error)：
        # 複製最後一組光流圖，賦予給最後一個 end_frame 的 ID，使總光流圖數量完美等於 N
        if last_flow_bgr is not None:
            last_frame_id = clip_frame_ids[-1]
            out_path = os.path.join(flow_output_dir, f"{last_frame_id}{ext}")
            cv2.imwrite(out_path, last_flow_bgr)

def process_txt(path, model, args):
    print(f"正在載入基準測試導航表: {path}")
    df_benchmark = parse_benchmark_txt(path)
    print(f"成功載入，共計 {len(df_benchmark)} 個短片片段 (Clips) 待處理。")

    for idx, row in tqdm(df_benchmark.iterrows(), total=len(df_benchmark), desc="總進度"):
        video_id_str = row['video_id']  
        start_f = row['start_frame']
        end_f = row['end_frame']
        
        accident_type, folder_name = video_id_str.split('/')
        
        # 自動適應 CAP-DATA 與 DADA-DATA 的目錄佈局
        chosen_dataset = None
        for dataset_name in ["CAP-DATA", "DADA-DATA"]:
            potential_dir = os.path.join(ROOT_DIR, dataset_name, accident_type, folder_name, "images")
            if os.path.exists(potential_dir):
                chosen_dataset = dataset_name
                break
                
        if chosen_dataset is None:
            print(f"\n[警告] 根據結構找不到該影片實體路徑: {video_id_str}，已跳過。")
            continue
            
        video_root_dir = os.path.join(ROOT_DIR, chosen_dataset, accident_type, folder_name)
        image_dir = os.path.join(video_root_dir, "images")

        flow_output_dir = os.path.join("/media/wayne/27CD255760735841/MMAU/", chosen_dataset, accident_type, folder_name, f"flow_{start_f}_{end_f}")
        os.makedirs(flow_output_dir, exist_ok=True)

        # 執行光流運算
        process_clip(model, image_dir, flow_output_dir, start_f, end_f)
            
    print(f"🎉 處理完 {path} ！！\n")

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", required=True)
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    # 在主程序中初始化一次 RAFT 模型
    print("正在初始化 RAFT 模型...")
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()
    print("RAFT 模型初始化成功，準備開始批次處理。")

    # 依序處理各個基準測試文字檔
    txt_tasks = [
        '/home/wayne/Documents/Progress/RAM/configs/mini_training.txt',
        '/home/wayne/Documents/Progress/RAM/configs/mini_test.txt',
        '/home/wayne/Documents/Progress/RAM/configs/full_training.txt',
        '/home/wayne/Documents/Progress/RAM/configs/full_test_5s.txt',
        '/home/wayne/Documents/Progress/RAM/configs/full_test_4s.txt',
        '/home/wayne/Documents/Progress/RAM/configs/full_test_2s.txt'
    ]

    for txt_path in txt_tasks:
        if os.path.exists(txt_path):
            process_txt(txt_path, model, args)
        else:
            print(f"[跳過] 找不到指定文字檔路徑: {txt_path}")
            
    print("🚀 所有任務的 RAFT 光流圖皆已順利生產完畢！")