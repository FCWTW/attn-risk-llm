import os
import math
import multiprocessing
import mmcv
import torch
from mmseg.apis import inference_model, init_model, show_result_pyplot
from tqdm import tqdm  # 加入 tqdm

CONFIG_FILE = '/home/wayne/Documents/Progress/mmsegmentation/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py'
CHECKPOINT_FILE = 'mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024_20221202_141901-28ad20f1.pth'
BASE_ROOT = '/home/wayne/Documents/MMAU'

GPU_IDS = [0]
NUM_PROCESSES = 2

def worker_process(file_list, gpu_id, process_idx):
    device = f'cuda:{gpu_id}'
    try:
        model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=device)
    except Exception as e:
        print(f'\n[Process {process_idx}] Failed to load model: {e}')
        return

    for input_path, output_path in tqdm(file_list, desc=f'Worker {process_idx} (GPU {gpu_id})', position=process_idx, leave=True):
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img = mmcv.imread(input_path)
            result = inference_model(model, img)
            show_result_pyplot(model, img, result, show=False, out_file=output_path, opacity=1)
            
        except Exception as e:
            tqdm.write(f'[Process {process_idx}] Failed to process {input_path}: {e}')

def main():
    all_tasks = []
    print("Scanning files...")
    
    for set_dir in ['DADA-DATA', 'CAP-DATA']:
        set_base_path = os.path.join(BASE_ROOT, set_dir)
        if not os.path.exists(set_base_path):
            print(f'Can not find {set_base_path}')
            continue
        
        for category_folder in os.listdir(set_base_path):
            category_path = os.path.join(set_base_path, category_folder)
            if not os.path.exists(category_path):
                print(f'Can not find {category_path}')
                continue
            
            for video_folder in os.listdir(category_path):
                video_path = os.path.join(category_path, video_folder)
                if not os.path.exists(video_path):
                    print(f'Can not find {video_path}')
                    continue

                input_dir = os.path.join(video_path, 'images')
                output_dir = os.path.join(video_path, 'segmentation')
                if not os.path.exists(input_dir):
                    print(f'--- Can not find {input_dir}/images ---')
                    continue
                
                os.makedirs(output_dir, exist_ok=True)
                for filename in os.listdir(input_dir):
                    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                        input_image_path = os.path.join(input_dir, filename)
                        output_image_path = os.path.join(output_dir, filename)
                        all_tasks.append((input_image_path, output_image_path))

    total_files = len(all_tasks)
    print(f"A total of {total_files} images were found. Processing will be performed using {NUM_PROCESSES} processes.\n")

    if total_files == 0:
        print("No image found.")
        return

    chunk_size = math.ceil(total_files / NUM_PROCESSES)
    chunks = [all_tasks[i:i + chunk_size] for i in range(0, total_files, chunk_size) if all_tasks[i:i + chunk_size]]
    processes = []
    multiprocessing.set_start_method('spawn', force=True)

    for i in range(len(chunks)):
        gpu_id = GPU_IDS[i % len(GPU_IDS)]
        p = multiprocessing.Process(
            target=worker_process, 
            args=(chunks[i], gpu_id, i)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print('\n' * len(chunks) + 'All processes finish !!!')

if __name__ == '__main__':
    main()