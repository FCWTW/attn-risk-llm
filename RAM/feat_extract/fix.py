import argparse
import os
import glob
import numpy as np
from tqdm import tqdm

def fix_npz_toa(flow_root_dir, configs_dir):
    print("-> Scanning for .txt annotation files in the config directory...")
    txt_files = glob.glob(os.path.join(configs_dir, "*.txt"))
    print(f"-> Found {len(txt_files)} annotation files; starting to parse and correct .npz files...")

    success_count = 0
    missing_count = 0

    for txt_file in txt_files:
        print(f"-> Loading {os.path.basename(txt_file)}...")
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split(',')
                    meta_info = parts[0].strip().split()
                    
                    if len(meta_info) < 5:
                        continue

                    rel_dir = meta_info[0]
                    start_frame = int(meta_info[2])
                    end_frame = int(meta_info[3])
                    actual_toa = int(meta_info[4])
                    
                    resolved_npz_path = None
                    for dataset_name in ["CAP-DATA", "DADA-DATA"]:
                        potential_path = os.path.join(
                            flow_root_dir, dataset_name, rel_dir, f"clip_{start_frame}_{end_frame}.npz"
                        )
                        if os.path.exists(potential_path):
                            resolved_npz_path = potential_path
                            break

                    if resolved_npz_path is not None:
                        with np.load(resolved_npz_path, allow_pickle=True) as data:
                            content = {key: data[key] for key in data.files}
                        
                        # Update toa
                        content['toa'] = np.array(actual_toa)
                        np.savez_compressed(resolved_npz_path, **content)
                        success_count += 1
                    else:
                        missing_count += 1
                        
                except Exception as e:
                    print(f"[Error] A problem occurred while parsing this line: {line} Error cause: {e}")
                    continue

    print("\n" + "="*30)
    print("-> Mission Complete !")
    print(f"-> Number of successfully repaired .npz files: {success_count}")
    print(f"-> Number of files for which no matching file was found (skipped): {missing_count}")
    print("="*30)

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    RAM_dir = os.path.dirname(current_dir)
    default_configs_dir = os.path.join(RAM_dir, "RAM/config")

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default="/media/wayne/27CD255760735841/MMAU/", help="Path to npz files folder")
    parser.add_argument('--configs_dir', default=default_configs_dir, help="Path to txt")
    args = parser.parse_args()
    print(default_configs_dir)

    fix_npz_toa(args.output_dir, args.configs_dir)