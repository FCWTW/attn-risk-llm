import argparse
import os
import cv2
import glob
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image

from ultralytics import YOLO
from RAFT.core.raft import RAFT
from RAFT.core.utils import flow_viz
from RAFT.core.utils.utils import InputPadder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VALID_CLASSES = {
    "person",
    "car",
    "truck",
    "bus",
    "bicycle",
    "motorcycle"
}

def parse_benchmark_txt(txt_path):
    """
    Parse the Benchmark .txt file and convert it into a DataFrame table of contents
    """
    parsed_data = []
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"[ERROR] Failed to load {txt_path}.")
        
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ')
            if len(parts) >= 5:
                t_ai = parts[4].split(',')[0]
                parsed_data.append([
                    parts[0],       # video_id (例如 "1/001537")
                    int(parts[1]),  # label 
                    int(parts[2]),  # start_frame
                    int(parts[3]),  # end_frame
                    float(t_ai)     # the begining time of the accident
                ])
                
    return pd.DataFrame(
        parsed_data, 
        columns=['video_id', 'label', 'start_frame', 'end_frame', 't_ai']
    )

def load_image(imfile):
    """
    Image Loading and Tensor Conversion for RAFT
    """
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def get_clip_frames(image_dir, start_frame, end_frame):
    """
    Scan once and filter frame paths and IDs within a specific time slice
    """
    all_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    clip_image_paths = []
    clip_frame_ids = []
    ext = ".jpg"
    
    for fname in all_files:
        try:
            frame_num = int(os.path.splitext(fname)[0])
        except ValueError:
            continue
        if start_frame <= frame_num <= end_frame:
            clip_image_paths.append(os.path.join(image_dir, fname))
            clip_frame_ids.append(os.path.splitext(fname)[0])
            ext = os.path.splitext(fname)[1]
    return clip_image_paths, clip_frame_ids, ext

def run_yolo_tracking(yolo_model, clip_image_paths, clip_frame_ids):
    """
    Performing YOLO Object Tracking and Hood Geometry Filtering
    """
    records = []
    results = yolo_model.track(
        source=clip_image_paths,
        tracker="botsort.yaml",  
        conf=0.7,                 
        verbose=False
    )

    for frame_id, result in zip(clip_frame_ids, results):
        if result.boxes is None:
            continue
        img_h, img_w = result.orig_shape 

        for box in result.boxes:
            cls_id = int(box.cls.item())
            cls_name = yolo_model.names[cls_id]

            if cls_name not in VALID_CLASSES:
                continue

            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            track_id = int(box.id.item()) if box.id is not None else -1
            if cls_name == "car":
                box_width = x2 - x1
                if box_width > (img_w * 0.65) and y2 > (img_h * 0.90):
                    continue
            if track_id == -1:
                continue
            records.append([
                frame_id, track_id, cls_name,
                int(x1), int(y1), int(x2), int(y2), conf
            ])

    if not records:
        return None
    return pd.DataFrame(records, columns=["frame", "track_id", "label", "x1", "y1", "x2", "y2", "confidence"])

def run_raft_flow(raft_model, clip_image_paths, clip_frame_ids, ext, flow_output_dir):
    """
    Implementation of RAFT for Optical Flow Estimation and Storage (Including Fencepost Error Correction)
    """
    if len(clip_image_paths) < 2:
        return
    last_flow_bgr = None
    with torch.no_grad():
        for i in range(len(clip_image_paths) - 1):
            imfile1 = clip_image_paths[i]
            imfile2 = clip_image_paths[i+1]
            frame_id = clip_frame_ids[i]
            
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            _, flow_up = raft_model(image1, image2, iters=20, test_mode=True)
            
            flow_img = flow_up[0].permute(1, 2, 0).cpu().numpy()
            flow_rgb = flow_viz.flow_to_image(flow_img)
            flow_bgr = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR)
            
            out_path = os.path.join(flow_output_dir, f"{frame_id}{ext}")
            cv2.imwrite(out_path, flow_bgr)
            
            last_flow_bgr = flow_bgr
        if last_flow_bgr is not None:
            last_frame_id = clip_frame_ids[-1]
            out_path = os.path.join(flow_output_dir, f"{last_frame_id}{ext}")
            cv2.imwrite(out_path, last_flow_bgr)

def process_txt(path, yolo_model, raft_model, root_dir, output):
    print(f"-> Loading {path}")
    df_benchmark = parse_benchmark_txt(path)
    print(f"-> Loaded successfully. A total of {len(df_benchmark)} clips are waiting to be processed.")

    for idx, row in tqdm(df_benchmark.iterrows(), total=len(df_benchmark), desc="Progress"):
        video_id_str = row['video_id']  
        start_f = row['start_frame']
        end_f = row['end_frame']
        
        accident_type, folder_name = video_id_str.split('/')
        chosen_dataset = None
        for dataset_name in ["CAP-DATA", "DADA-DATA"]:
            potential_dir = os.path.join(root_dir, dataset_name, accident_type, folder_name, "images")
            if os.path.exists(potential_dir):
                chosen_dataset = dataset_name
                break
                
        if chosen_dataset is None:
            print(f"\n[Warning] {video_id_str} could not be found based on the structure; it has been skipped.")
            continue

        # Get video clip path            
        video_root_dir = os.path.join(root_dir, chosen_dataset, accident_type, folder_name)
        image_dir = os.path.join(video_root_dir, "images")
        output_dir = os.path.join(output, chosen_dataset, accident_type, folder_name)
        clip_image_paths, clip_frame_ids, ext = get_clip_frames(image_dir, start_f, end_f)
        if not clip_image_paths:
            continue

        # YOLO
        df_clip_res = run_yolo_tracking(yolo_model, clip_image_paths, clip_frame_ids)
        if df_clip_res is not None:
            output_csv = os.path.join(output_dir, f"detections_{start_f}_{end_f}.csv")
            df_clip_res.to_csv(output_csv, index=False)  

        # RAFT
        flow_output_dir = os.path.join(output_dir, f"flow_{start_f}_{end_f}")
        os.makedirs(flow_output_dir, exist_ok=True)
        run_raft_flow(raft_model, clip_image_paths, clip_frame_ids, ext, flow_output_dir)
    print(f"-> Finish {path} \n")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    RAM_dir = os.path.dirname(current_dir)
    configs_dir = os.path.join(RAM_dir, "config")
    raft_path = os.path.join(current_dir, "raft-kitti.pth")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=raft_path, help="Path to RAFT checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--root_dir', default="/home/wayne/Documents/MMAU", help="Path to MM-AU dataset")
    parser.add_argument('--output_dir', default="/media/wayne/27CD255760735841/MMAU/", help="Path to output folder")
    args = parser.parse_args()

    print("-> Initializing the YOLO11 model...")
    yolo_model = YOLO("yolo11m.pt")
    print("-> Initializing the RAFT model...")
    raft_model = torch.nn.DataParallel(RAFT(args))
    raft_model.load_state_dict(torch.load(args.model))
    raft_model = raft_model.module
    raft_model.to(DEVICE)
    raft_model.eval()

    print(f"-> Scanning {configs_dir}...")
    txt_tasks = sorted([
        os.path.join(configs_dir, f)
        for f in os.listdir(configs_dir)
        if f.lower().endswith('.txt')
    ])

    if not txt_tasks:
        print(f"[ERROR] No .txt files were found in {configs_dir}.")
    else:
        print(f"-> A total of {len(txt_tasks)} .txt have been detected.")
        for path in txt_tasks:
            print(f"--- {os.path.basename(path)}")
        print("-" * 50)
        for txt_path in txt_tasks:
            process_txt(txt_path, yolo_model, raft_model, args.root_dir, args.ouput_dir)
    print("-> Mission Complete !")