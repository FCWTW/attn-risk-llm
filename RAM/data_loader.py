import cv2
import torch
import torch.nn.functional as F
import pandas as pd
from torchvision import transforms as video_transforms
import os
from tqdm import tqdm
from utils import parse_timestamp, subtract_timestamps, uniform_frame_sampling, pad_tensor

class FetchData():
    def __init__(self, txt_path, set_name='train', length=None, segment_duration=1, segment_interval=0.5, target_size=(224, 224), after_acc = True, strategy='full'):
        self.set_name = set_name
        self.segment_duration = segment_duration
        self.segment_interval = segment_interval
        self.target_size = target_size
        self.frames_list = []
        self.labels = []
        self.after_acc = after_acc
        self.num_classes = 2
        
        if self.set_name == 'val':
            file_name = 'val.txt'
        elif self.set_name == 'train':
            file_name = f"{strategy}_training.txt"
        elif self.set_name == 'test':
            file_name = f"{strategy}_test.txt"
        else:
            raise ValueError(f"Unknown parameters: {self.set_name}")
            
        full_txt_path = os.path.join(txt_path, file_name)
        parsed_data = []
        with open(full_txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 用空格切分
                parts = line.split(' ')
                if len(parts) >= 5:
                    # parts[4] 大多會是 "151,a" 這種型態，我們用逗號切開只拿前面的數字 "151"
                    toa_clean = parts[4].split(',')[0]
                    
                    parsed_data.append([
                        parts[0],       # video_id
                        int(parts[1]),  # label
                        int(parts[2]),  # start_frame
                        int(parts[3]),  # end_frame
                        float(toa_clean)# toa
                    ])
                    
        # 將乾淨的資料建立為 DataFrame
        self.df = pd.DataFrame(
            parsed_data, 
            columns=['video_id', 'label', 'start_frame', 'end_frame', 'toa']
        )
        
        if file_name == 'full_test.txt':
            self.is_dada = 0
        else:
            self.is_dada = 1
        if length is not None:
            self.length_videos = min(length, len(self.df))
            self.df = self.df.iloc[:self.length_videos].reset_index(drop=True)
        else:
            self.length_videos = len(self.df)
        self.prepare_data()
        
    def __len__(self):
        return len(self.frames_list)
    
    def __getitem__(self, id):
        if torch.is_tensor(id):
            id = id.tolist()
        frames = self.frames_list[id] # frames 會是 [vid_id, start_frame, end_frame]
        label_data = self.labels[id]
        label = label_data[0]        # 1.0 或 0.0 (車禍與否)
        current_time = label_data[1] # 當前片段的結束幀數
        toa = label_data[2]          # 車禍發生的幀數 (-1 代表沒車禍)
        
        # 回傳四個物件，對接 utils.py 的 x, y, time_idx, toa = batch
        return frames, label, current_time, toa
        
    def check_overlap(self, t1, t2):
        return 0 if t1<t2 else 1
    
    def create_time_segments(self, vid_name, start_frame, end_frame, accident_frame, label, is_dada, fps=30):
        time_segments = []
        vid_segments = []
        window_size = int(self.segment_duration * fps)
        step_size = int(self.segment_interval * fps)
        
        segment_len = end_frame - start_frame + 1
        
        # 【核心修正 3】解決片段小於 30 幀的遺失問題
        if segment_len < window_size:
            # 如果太短，直接以 start_frame 為起點，強行向後拉滿 30 幀
            # (備註：這需要確保原始影片長度夠，或者在 DataLoader 讀取時做 Padding)
            current_end = start_frame + window_size
            time_segments.append([label, current_end, accident_frame])
            vid_segments.append([vid_name, start_frame, current_end, is_dada])
        else:
            # 【核心修正 2】從該片段的真實 start_frame 開始滑動，而不是從 0 開始
            current_start = start_frame
            while current_start + window_size <= end_frame:
                current_end = current_start + window_size
                time_segments.append([label, current_end, accident_frame])
                vid_segments.append([vid_name, current_start, current_end, is_dada])
                
                current_start += step_size
                
        if not time_segments:
            return None
            
        time_segments = torch.stack([torch.tensor(x, dtype=torch.float32) for x in time_segments])
        return time_segments, vid_segments

    def create_test_segment(self, vid_name, start_frame, end_frame, accident_frame, label, is_dada):
        label_info = torch.tensor([[label, end_frame, accident_frame]], dtype=torch.float32)
        vid_info = [[vid_name, start_frame, end_frame, is_dada]]
        return label_info, vid_info

    def prepare_data(self):
        for i in tqdm(range(self.length_videos)):           
            # 讀取 .txt 解析出來的對應欄位資料
            label = int(self.df['label'].iloc[i])           
            toa = float(self.df['toa'].iloc[i])          
            start_frame = int(self.df['start_frame'].iloc[i])   
            end_frame = int(self.df['end_frame'].iloc[i])   
            vid_name = str(self.df['video_id'].iloc[i])        
            is_dada = self.is_dada

            # if self.set_name == 'test':
            #     temp = self.create_test_segment(vid_name, start_frame, end_frame, toa, label, is_dada)
            # else:
            #     temp = self.create_time_segments(vid_name, start_frame, end_frame, toa, label, is_dada)
            temp = self.create_test_segment(vid_name, start_frame, end_frame, toa, label, is_dada)
            if temp is not None:
                self.frames_list.extend(temp[1])
                self.labels.append(temp[0])
                    
        if len(self.labels) > 0:
            self.labels = torch.cat(self.labels)
        else:
            print("⚠️ 警告：沒有任何片段被成功切出，請檢查資料集範圍或 window_size 設定。")     
        print(f"任務準備完成！共切出 {len(self.labels)} 個 {int(self.segment_duration * 30)} 幀片段給 {self.set_name} 階段使用")

def load_frame_sequence(vid_info, root_dir='/home/wayne/Documents/MMAU/', target_frames=None):
    vid_name, start_frame, end_frame, is_dada = vid_info
    sub_class, vid_id = vid_name.split('/')
    
    frames = []
    last_valid_img = None
    
    for f_idx in range(int(start_frame), int(end_frame)+1):
        current_frame_idx = f_idx + 1 
        
        if is_dada:
            vid_id_padded = vid_id.zfill(3)
            img_name = f"{current_frame_idx:04d}.png"
            img_path = os.path.join(root_dir, 'DADA-DATA', sub_class, vid_id_padded, 'images', img_name)
        else:
            vid_id_padded = vid_id.zfill(6)
            img_name = f"{current_frame_idx:06d}.jpg"
            img_path = os.path.join(root_dir, 'CAP-DATA', sub_class, vid_id_padded, 'images', img_name)
            
        img = cv2.imread(img_path)
        
        if img is None:
            if last_valid_img is not None:
                img = last_valid_img.copy()
            else:
                return None
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            min_edge = min(h, w)
            target_min = 720
            if min_edge != target_min:
                scale = target_min / float(min_edge)
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                img = cv2.resize(img, (new_w, new_h))
                
            last_valid_img = img
            
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        frames.append(img_tensor)
        
    video_tensor = torch.stack(frames)
    if target_frames is not None:
        indices = torch.linspace(0, len(video_tensor) - 1, target_frames).long()
        video_tensor = video_tensor[indices]
    return video_tensor

class TrainCollator():
    def __init__(self, model_type, target_size, root_dir):
        self.model_type = model_type
        self.target_size = target_size
        self.root_dir = root_dir
        
    def __call__(self, batch):
        if self.model_type in ['VidNeXt', 'ConvNeXtVanillaTransformer', 'ResNetNSTtransformer', 'ViViT']:
            dims_shape = [0, 1, 2, 3, 4] 
        else:
            dims_shape = [0, 2, 1, 3, 4] 
            
        train_trans = video_transforms.Compose([
                video_transforms.RandomHorizontalFlip(),
                video_transforms.Resize(self.target_size, antialias=True),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        valid_videos = []
        valid_labels = []
        valid_times = []
        valid_toas = []
        segment_duration = None

        # 逐一檢查 Batch 裡面的每一筆資料
        for item in batch:
            vid_info, label, time_idx, toa = item
            if segment_duration is None:
                _, start_frame, end_frame, _ = vid_info
                segment_duration = int(end_frame) - int(start_frame) +1
            if self.model_type in ['TimeSformer', 'ViViT']:
                video_tensor = load_frame_sequence(vid_info, root_dir=self.root_dir, target_frames=16)
            else:
                video_tensor = load_frame_sequence(vid_info, root_dir=self.root_dir)

            # 【嚴格把關】：只有當影片不是 None 的時候，才把它加入訓練名單！
            if video_tensor is not None:
                valid_videos.append(video_tensor)
                valid_labels.append(label)
                valid_times.append(time_idx)
                valid_toas.append(toa)

        # 萬一這個 Batch 運氣太差，裡面所有的影片都壞掉了
        if len(valid_videos) == 0:
            print("⚠️ 警告：此 Batch 所有資料皆損壞，已跳過。")
            # 隨便回傳一個 shape 合法的 Dummy Tensor 防止程式當掉 (這個 batch 的 loss 會被 optimizer 忽視)
            dummy_video = torch.zeros((1, segment_duration, 3, self.target_size[0], self.target_size[1]))
            return dummy_video.permute(*dims_shape), torch.tensor([0]), torch.tensor([0]), torch.tensor([0])

        # 安全地把「乾淨的影片」做轉換與堆疊
        transformed_video = torch.stack([train_trans(video) for video in valid_videos])

        # 回傳過濾後乾淨的 Tensor
        return transformed_video.permute(*dims_shape), torch.tensor(valid_labels), torch.tensor(valid_times), torch.tensor(valid_toas)

class ValCollator():
    def __init__(self, model_type, target_size, root_dir):
        self.model_type = model_type
        self.target_size = target_size
        self.root_dir = root_dir
        
    def __call__(self, batch):
        if self.model_type in ['VidNeXt', 'ConvNeXtVanillaTransformer', 'ResNetNSTtransformer', 'ViViT']:
            dims_shape = [0, 1, 2, 3, 4] 
        else:
            dims_shape = [0, 2, 1, 3, 4] 
            
        val_trans = video_transforms.Compose([
            video_transforms.Resize(self.target_size, antialias=True),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        valid_videos = []
        valid_labels = []
        valid_times = []
        valid_toas = []
        segment_duration = None

        for item in batch:
            vid_info, label, time_idx, toa = item
            if segment_duration is None:
                _, start_frame, end_frame, _ = vid_info
                segment_duration = int(end_frame) - int(start_frame) + 1
            if self.model_type in ['TimeSformer', 'ViViT']:
                video_tensor = load_frame_sequence(vid_info, root_dir=self.root_dir, target_frames=16)
            else:
                video_tensor = load_frame_sequence(vid_info, root_dir=self.root_dir)

            # 【嚴格把關】
            if video_tensor is not None:
                valid_videos.append(video_tensor)
                valid_labels.append(label)
                valid_times.append(time_idx)
                valid_toas.append(toa)

        if len(valid_videos) == 0:
            print("⚠️ 警告：此測試 Batch 所有資料皆損壞，已跳過。")
            dummy_video = torch.zeros((1, segment_duration, 3, self.target_size[0], self.target_size[1]))
            return dummy_video.permute(*dims_shape), torch.tensor([0]), torch.tensor([0]), torch.tensor([0])

        transformed_video = torch.stack([val_trans(video) for video in valid_videos])
        return transformed_video.permute(*dims_shape), torch.tensor(valid_labels), torch.tensor(valid_times), torch.tensor(valid_toas)

class TestCollator():
    def __init__(self, model_type, target_size, root_dir):
        self.model_type = model_type
        self.target_size = target_size
        self.root_dir = root_dir
        
    def __call__(self, batch):
        if self.model_type in ['VidNeXt', 'ConvNeXtVanillaTransformer', 'ResNetNSTtransformer', 'ViViT']:
            dims_shape = [0, 1, 2, 3, 4] 
        else:
            dims_shape = [0, 2, 1, 3, 4] 
            
        test_trans = video_transforms.Compose([
            video_transforms.Resize(self.target_size, antialias=True),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        valid_videos = []
        valid_labels = []
        valid_times = []
        valid_toas = []
        segment_duration = None

        for item in batch:
            vid_info, label, time_idx, toa = item
            if segment_duration is None:
                _, start_frame, end_frame, _ = vid_info
                segment_duration = int(end_frame) - int(start_frame) + 1
            if self.model_type in ['TimeSformer', 'ViViT']:
                video_tensor = load_frame_sequence(vid_info, root_dir=self.root_dir, target_frames=16)
            else:
                video_tensor = load_frame_sequence(vid_info, root_dir=self.root_dir)

            if video_tensor is not None:
                valid_videos.append(video_tensor)
                valid_labels.append(label)
                valid_times.append(time_idx)
                valid_toas.append(toa)

        if len(valid_videos) == 0:
            print("⚠️ 警告：此測試 Batch 所有資料皆損壞，已跳過。")
            dummy_video = torch.zeros((1, segment_duration, 3, self.target_size[0], self.target_size[1]))
            return dummy_video.permute(*dims_shape), torch.tensor([0]), torch.tensor([0]), torch.tensor([0])
        transformed_video = torch.stack([test_trans(video) for video in valid_videos])
        return transformed_video.permute(*dims_shape), torch.tensor(valid_labels), torch.tensor(valid_times), torch.tensor(valid_toas)

if __name__ == "__main__":
    pass