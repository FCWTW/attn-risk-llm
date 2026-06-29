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
from src.model import FeatureExtractor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
])

def log_information(vid_id, frame_id, track_id, e, configs_dir):
    """
    Dynamically locates the project's specified configs directory
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_path = os.path.join(configs_dir, 'Feature_Extraction_Error.log')
    
    if not logger.hasHandlers():
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    logger.info(f"Error: {e} ------ Video: {vid_id} -- Frame: {frame_id} -- TrackID: {track_id}")

def featureExt(image, extractor):
    """
    Reusing externally provided pre-trained weights to avoid repeated initialization
    """
    if image.size != (224, 224):
        image = image.resize((224, 224), Image.BILINEAR)
        
    image_tensor = transform(image)
    image_tensor = torch.unsqueeze(image_tensor, 0).float().to(device=DEVICE)
    
    with torch.no_grad():
        feat = extractor(image_tensor)  # 1 x 2048 x 1 x 1
        feat = torch.squeeze(feat, 2)   # 1 x 2048 x 1
        feat = torch.squeeze(feat, 2)   # 1 x 2048
    return feat

def get_frame_file(folder_path, frame_num):
    """
    Automatically ignores 4-digit or 6-digit leading zeros and file extension discrepancies
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

def process_features(root_dir, flow_root_dir, configs_dir):
    """
    Main Program for Feature Extraction
    """
    print(f"-> Loading ResNet50 to {DEVICE}...")
    extractor = FeatureExtractor().to(device=DEVICE)
    extractor.eval()
    
    print("-> Loading YOLO CSV files...")
    csv_pattern = os.path.join(flow_root_dir, "**", "detections_*.csv")
    csv_files = natsorted(glob.glob(csv_pattern, recursive=True))
    print(f"-> Loaded successfully. A total of {len(csv_files)} clips are waiting to be processed.")

    for csv_file in tqdm(csv_files, desc="Feature Extraction"):
        csv_basename = os.path.basename(csv_file)
        match = re.search(r'detections_(\d+)_(\d+)\.csv', csv_basename)
        if not match:
            continue
            
        start_frame = int(match.group(1))
        end_frame = int(match.group(2))
        num_frames = end_frame - start_frame + 1
        rel_dir = os.path.relpath(os.path.dirname(csv_file), flow_root_dir)
        vid_id = os.path.basename(rel_dir)
        
        rgb_image_dir = os.path.join(root_dir, rel_dir, "images")
        flow_image_dir = os.path.join(flow_root_dir, rel_dir, f"flow_{start_frame}_{end_frame}")
        output_npz_dir = os.path.join(flow_root_dir, rel_dir)
        os.makedirs(output_npz_dir, exist_ok=True)

        detections = np.zeros((num_frames, 30, 6), dtype=np.float32)
        feature = np.zeros((num_frames, 31, 2048), dtype=np.float32)

        flow_images_cleaned = {}
        for t in range(num_frames):
            current_frame_num = start_frame + t
            rgb_path = get_frame_file(rgb_image_dir, current_frame_num)
            flow_path = get_frame_file(flow_image_dir, current_frame_num)

            if rgb_path is None or flow_path is None:
                continue

            rgb_img = Image.open(rgb_path).convert("RGB")
            flow_img = Image.open(flow_path).convert("RGB")
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
            flow_img_clean = flow_img.crop((left, top, left + W_rgb, top + H_rgb))
            flow_images_cleaned[current_frame_num] = flow_img_clean
            global_feat = featureExt(flow_img_clean, extractor)
            feature[t, 0, :] = global_feat.detach().cpu().numpy()

        df = pd.read_csv(csv_file)
        for frame_val, group in df.groupby('frame'):
            frame_num = int(frame_val)
            t = frame_num - start_frame
            
            if t < 0 or t >= num_frames or frame_num not in flow_images_cleaned:
                continue
                
            flow_img_clean = flow_images_cleaned[frame_num]
            W_rgb, H_rgb = flow_img_clean.size

            for obj_idx, (_, row) in enumerate(group.iterrows()):
                if obj_idx >= 30:
                    break

                track_id = int(row['track_id'])
                x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                x1, x2 = max(0, min(x1, W_rgb)), max(0, min(x2, W_rgb))
                y1, y2 = max(0, min(y1, H_rgb)), max(0, min(y2, H_rgb))
                
                if x2 <= x1 or y2 <= y1:
                    continue

                detections[t, obj_idx] = [track_id, x1, y1, x2, y2, 0.0]
                roi_img = flow_img_clean.crop((x1, y1, x2, y2))
                try:
                    object_feat = featureExt(roi_img, extractor)
                    object_feat_np = object_feat.detach().cpu().numpy()
                    feature[t, obj_idx + 1, :] = object_feat_np
                except Exception as e:
                    print(f"[ERROR] Object Feature Extraction Error on Video {vid_id}, Frame {frame_num}")
                    log_information(vid_id, frame_num, track_id, e, configs_dir)
                    continue

        save_file = os.path.join(output_npz_dir, f"clip_{start_frame}_{end_frame}.npz")
        np.savez_compressed(
            save_file, 
            feature=feature,
            detection=detections, 
            vid_id=vid_id, 
            toa=0
        )
    print("-> Mission Complete !")

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    RAM_dir = os.path.dirname(current_dir)
    configs_dir = os.path.join(RAM_dir, "configs")

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default="/home/wayne/Documents/MMAU", help="Path to MM-AU dataset")
    parser.add_argument('--output_dir', default="/media/wayne/27CD255760735841/MMAU/", help="Path to output folder")
    args = parser.parse_args()

    process_features(args.root_dir, args.output_dir, configs_dir)