from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_path, config_dir, phase, strategy, toTensor=False, device=torch.device('cuda')):
        self.data_path = data_path
        self.config_dir = config_dir
        self.phase = phase
        self.strategy = strategy
        self.toTensor = toTensor
        self.device = device
        
        if self.strategy == 'mini':
            if 'train' in self.phase:
                txt_name = 'mini_training.txt'
            elif 'val' in self.phase:
                txt_name = 'mini_val.txt'
            else:
                txt_name = 'mini_test.txt'
        else:
            if 'train' in self.phase:
                txt_name = 'full_training.txt'
            elif 'val' in self.phase:
                txt_name = 'full_val.txt'
            elif '5s' in self.phase:
                txt_name = 'full_test_5s.txt'
            elif '4s' in self.phase:
                txt_name = 'full_test_4s.txt'
            elif '2s' in self.phase:
                txt_name = 'full_test_2s.txt'
            else:
                print("-> Invalid phase, using full_test_5s.txt")
                txt_name = 'full_test_5s.txt'

        print(f"-> Use {txt_name}...")
        self.txt_path = os.path.join(self.config_dir, txt_name)
        self.files_list = self.get_filelist(self.data_path)

    def __len__(self):
        data_len = len(self.files_list)
        return data_len

    def get_filelist(self, filepath):
        assert os.path.exists(self.txt_path), f"-> Benchmark config file does not exist: {self.txt_path}"
        file_list = []
        
        with open(self.txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ')
                if len(parts) >= 5:
                    video_id = parts[0]       # 例如 "1/001537" 或 "1/001"
                    start_frame = int(parts[2])
                    end_frame = int(parts[3])
                    npz_name = f"clip_{start_frame}_{end_frame}.npz"
                    
                    # 依據 MM-AU 雙目錄佈局，自動搜尋對應大容量磁碟上的實體 .npz 檔案路徑
                    resolved_path = None
                    for dataset_name in ["CAP-DATA", "DADA-DATA"]:
                        potential_path = os.path.join(filepath, dataset_name, video_id, npz_name)
                        if os.path.exists(potential_path):
                            resolved_path = potential_path
                            break
                    
                    # 僅將真實存在的實體特徵檔案路徑寫入清單，防止無效片段導致 DataLoader 崩潰
                    if resolved_path is not None:
                        file_list.append(resolved_path)     
        return file_list

    def __getitem__(self, index):
        data_file = self.files_list[index]
        try:
            data = np.load(data_file, allow_pickle=True)
            toa = [data['toa'] + 0]  
            detection = data['detection']  
            
            # 兼容處理自訂特徵提取腳本產出的 'feature' 與原始官方的 'flow_feat' 鍵值
            flow = data['feature'] if 'feature' in data else data['flow_feat']
            vid_id = str(data['vid_id'])

        except Exception as e:
            raise IOError(f'Load data error! File: {data_file}, Error: {e}')

        if self.toTensor:
            detection = torch.Tensor(detection).to(self.device)
            toa = torch.Tensor(toa).to(self.device)
            flow = torch.Tensor(flow).to(self.device)
            
        return detection, toa, flow, vid_id