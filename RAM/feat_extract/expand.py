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

def process_gazemap(gazemap_arr, x1, y1, x2, y2):
    """
    Perform dynamic boundary cropping and mean pooling on the saliency map matrix that has been converted to NumPy,
    and normalize it to the range [0, 1]
    """
    H, W = gazemap_arr.shape
    x1, x2 = max(0, min(x1, W)), max(0, min(x2, W))
    y1, y2 = max(0, min(y1, H)), max(0, min(y2, H))
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
        
    roi = gazemap_arr[y1:y2, x1:x2]
    mean_val = np.mean(roi) / 255.0
    return mean_val

def log_information(vid_id, frame_id, track_id, e, configs_dir):
    """
    Dynamically locates the project's specified configs directory
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_path = os.path.join(configs_dir, 'Feature_Extraction_Error.log')
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    logger.info(f"Error: {e} ------ Video: {vid_id} -- Frame: {frame_id} -- TrackID: {track_id}")

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

def process_npz(npz_dir, gazemap_dir, output_dir, configs_dir):
    """
    Main Program for Expanding .npz Files
    """
    print("-> Scanning existing .npz feature files...")
    npz_pattern = os.path.join(npz_dir, "**", "clip_*.npz")
    npz_files = natsorted(glob.glob(npz_pattern, recursive=True))
    print(f"-> Scan successful. A total of {len(npz_files)} clips are waiting to be expanded.")

    for npz_file in tqdm(npz_files, desc="Expanding Saliency Priors"):
        npz_basename = os.path.basename(npz_file)
        match = re.search(r'clip_(\d+)_(\d+)\.npz', npz_basename)
        if not match:
            continue

        start_frame = int(match.group(1))
        end_frame = int(match.group(2))
        num_frames = end_frame - start_frame + 1

        # ex: CAP-DATA/1/001537
        rel_dir = os.path.relpath(os.path.dirname(npz_file), npz_dir)
        gazemap_folder = os.path.join(gazemap_dir, rel_dir, "gazemap")

        with np.load(npz_file, allow_pickle=True) as data:
            feature = data['feature']
            detection = data['detection']  # Shape: (num_frames, 30, 6)
            vid_id = data['vid_id']
            toa = data['toa']

        # (num_frames, 30, 1)
        saliency_prior = np.zeros((num_frames, 30, 1), dtype=np.float32)

        for t in range(num_frames):
            current_frame = start_frame + t
            
            # gazemap does not have the first 15 frames
            if current_frame < 16:
                continue

            gazemap_path = get_frame_file(gazemap_folder, current_frame)
            if gazemap_path is None or not os.path.exists(gazemap_path):
                continue

            # No BBox
            if not np.any(detection[t, :, 0] > 0):
                continue

            try:
                with Image.open(gazemap_path).convert('L') as img:
                    gazemap_arr = np.array(img, dtype=np.float32)

                for bbox in range(30):
                    track_id = int(detection[t, bbox, 0])
                    if track_id <= 0:
                        continue
                    x1, y1, x2, y2 = map(int, detection[t, bbox, 1:5])
                    s_ti = process_gazemap(gazemap_arr, x1, y1, x2, y2)
                    saliency_prior[t, bbox, 0] = s_ti

            except Exception as e:
                log_information(str(vid_id), current_frame, -1, e, configs_dir)
                continue

        # detection: (num_frames, 30, 6) --> (num_frames, 30, 7)
        extended_detection = np.concatenate([detection, saliency_prior], axis=-1)

        clip_output_dir = os.path.join(output_dir, rel_dir)
        os.makedirs(clip_output_dir, exist_ok=True)
        save_file = os.path.join(clip_output_dir, npz_basename)

        # Packaging
        np.savez_compressed(
            save_file,
            feature=feature,
            detection=extended_detection,
            vid_id=vid_id,
            toa=toa
        )
    print("-> Mission Complete!")

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    RAM_dir = os.path.dirname(current_dir)
    configs_dir = os.path.join(RAM_dir, "config")

    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_dir', default="/media/wayne/27CD255760735841/MMAU/", help="Path to .npz folder")
    parser.add_argument('--gazemap_dir', default="/home/wayne/Documents/MMAU", help="Path to gaze map folder")
    parser.add_argument('--output_dir', default="/media/wayne/27CD255760735841/MMAU-expand/", help="Path to output folder")
    args = parser.parse_args()

    process_npz(args.npz_dir, args.gazemap_dir, args.output_dir, configs_dir)