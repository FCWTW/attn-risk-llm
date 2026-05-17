import argparse
import os
import glob
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def get_heatmap(ori_img, mask_img):
    mask_img = cv2.normalize(mask_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mask_img = cv2.resize(mask_img, (ori_img.shape[1], ori_img.shape[0]))
    heatmap = cv2.applyColorMap(mask_img, cv2.COLORMAP_JET)
    ori_img = ori_img.astype(np.uint8)
    overlay = cv2.addWeighted(ori_img, 0.7, heatmap, 0.3, 0)
    return overlay

def process_single_pair(mask_path, camera_dir, output_dir):
    try:
        folder_name = os.path.basename(os.path.dirname(mask_path))
        file_name = os.path.basename(mask_path)
        jpg_file_name = file_name.replace('.png', '.jpg')
        
        ori_path = os.path.join(camera_dir, folder_name, jpg_file_name)

        if not os.path.exists(ori_path):
            return f"Image doesn't exist: {ori_path}"
        if not os.path.exists(mask_path):
            return f"Image doesn't exist: {mask_path}"

        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        ori_img = cv2.imread(ori_path)

        if mask_img is None or ori_img is None:
            return f"Failed to load images..."

        heatmap_img = get_heatmap(ori_img, mask_img)

        output_folder = os.path.join(output_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, jpg_file_name)
        cv2.imwrite(output_path, heatmap_img)
        
        return None
    except Exception as e:
        return f"Error on {mask_path}: {str(e)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gazemap_dir', required=True, type=str, help='Path to gazemap folder')
    parser.add_argument('--rgb_dir', required=True, type=str, help='Path to rgb image folder')
    parser.add_argument('--output_dir', required=True, type=str, help='Path to output folder')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    search_pattern = os.path.join(args.gazemap_dir, '*', '*.png')
    mask_paths = glob.glob(search_pattern)
    total_files = len(mask_paths)
    
    print(f"A total of {total_files} mask images were found and are ready for processing...")

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_single_pair, path, args.rgb_dir, args.output_dir): path for path in mask_paths}
        
        with tqdm(total=total_files, desc="Processing Images") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    tqdm.write(result) 
                pbar.update(1)

    print("All heatmaps have been processed!")