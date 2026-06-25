import os
import math
import multiprocessing
import mmcv
import torch
from mmseg.apis import inference_model, init_model, show_result_pyplot

# Path for pretrained model
CONFIG_FILE = '/home/wayne/Documents/Progress/mmsegmentation/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py'
CHECKPOINT_FILE = 'mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024_20221202_141901-28ad20f1.pth'

# Path for BDDA dataset
BASE_ROOT = '/home/wayne/Documents/BDDA'

GPU_IDS = [0] 
NUM_PROCESSES = 2 

def worker_process(file_list, gpu_id, process_idx):
    """
    The work function of each process
    """
    print(f'Process {process_idx} is running on GPU {gpu_id} and needs to process {len(file_list)} images.')
    
    device = f'cuda:{gpu_id}'
    try:
        model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=device)
    except Exception as e:
        print(f'[Process {process_idx}] Failed to load model: {e}')
        return

    for input_path, output_path in file_list:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img = mmcv.imread(input_path)
            result = inference_model(model, img)
            show_result_pyplot(model, img, result, show=False, out_file=output_path, opacity=1)
            
        except Exception as e:
            print(f'[Process {process_idx}] Failed to process {input_path}: {e}')
            
    print(f'[Process {process_idx}] Finish！')

def main():
    all_tasks = []
    print("Scanning files...")
    
    for set_dir in ['training', 'test', 'validation']:
        input_base_path = os.path.join(BASE_ROOT, set_dir, 'camera_frames')
        output_base_path = os.path.join(BASE_ROOT, set_dir, 'segmentation')
        
        if not os.path.exists(input_base_path):
            continue

        for digit_folder in os.listdir(input_base_path):
            input_digit_path = os.path.join(input_base_path, digit_folder)
            output_digit_path = os.path.join(output_base_path, digit_folder)
            
            if not os.path.isdir(input_digit_path):
                continue
                
            os.makedirs(output_digit_path, exist_ok=True)
            for filename in os.listdir(input_digit_path):
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    input_image_path = os.path.join(input_digit_path, filename)
                    output_image_path = os.path.join(output_digit_path, filename)
                    all_tasks.append((input_image_path, output_image_path))

    total_files = len(all_tasks)
    print(f"A total of {total_files} images were found. Processing will be performed using {NUM_PROCESSES} processes.")

    if total_files == 0:
        print("No image found.")
        return

    chunk_size = math.ceil(total_files / NUM_PROCESSES)
    chunks = [all_tasks[i:i + chunk_size] for i in range(0, total_files, chunk_size)]
    processes = []
    multiprocessing.set_start_method('spawn', force=True)

    for i in range(NUM_PROCESSES):
        gpu_id = GPU_IDS[i % len(GPU_IDS)]
        p = multiprocessing.Process(
            target=worker_process, 
            args=(chunks[i], gpu_id, i)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print('All processes finish !!!')

if __name__ == '__main__':
    main()