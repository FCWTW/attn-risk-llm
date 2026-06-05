import cv2
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_timestamp(timestamp_str):
    # Split the timestamp string into components
    components = timestamp_str.split(':')
    # Extract and convert components to integers
    hours = int(components[0])
    minutes = int(components[1])
    seconds = int(components[2])
    centiseconds = int(components[3])
    return hours, minutes, seconds, centiseconds

def subtract_timestamps(timestamp_str1, timestamp_str2):
    # Parse both timestamps
    hours1, minutes1, seconds1, centiseconds1 = parse_timestamp(timestamp_str1)
    hours2, minutes2, seconds2, centiseconds2 = parse_timestamp(timestamp_str2)
    # Calculate the differences for each component
    hours_diff = hours2 - hours1
    minutes_diff = minutes2 - minutes1
    seconds_diff = seconds2 - seconds1
    centiseconds_diff = centiseconds2 - centiseconds1
    
    # Handle negative differences (borrowing)
    if centiseconds_diff < 0:
        centiseconds_diff += 100
        seconds_diff -= 1
    if seconds_diff < 0:
        seconds_diff += 60
        minutes_diff -= 1
    if minutes_diff < 0:
        minutes_diff += 60
        hours_diff -= 1
    result = hours_diff*3600 + minutes_diff*60 + seconds_diff + centiseconds_diff/100
    return abs(result)

def uniform_frame_sampling(frame_list, target_frames):
    # Calculate the step size to achieve uniform sampling
    step_size = len(frame_list) // (target_frames - 1)

    # Use list comprehension to select frames at uniform intervals
    sampled_list = [frame_list[i * step_size] for i in range(target_frames - 1)] + [frame_list[-1]]

    return sampled_list

def pad_tensor(video_tensor, target_frames=30):
    if video_tensor.shape[0] <target_frames:
        return torch.cat([video_tensor, torch.zeros(max(target_frames - video_tensor.size(0), 0), *video_tensor.shape[1:])])
    if video_tensor.shape[0] >target_frames:
        return video_tensor[:target_frames]
    return video_tensor

# Fucntion to Calculate the Duration of an epoch
def epoch_time(start_time, end_time):
    duration = end_time - start_time
    minutes = duration//60
    seconds = duration - (60*minutes)
    return int(minutes), int(seconds)


def train_model_acc_grad(model, loader, optimizer, loss_fn, device, scaler, metric, metric2, opt_step_size=1, task_type='binclass'):
    # Intializing starting Epoch loss as 0
    loss_for_epoch = 0.0 
    score = 0.0   
    score2 = 0.0   
    # Model to be used in Training Mode
    model.train()
    iters = 0
    # For every Input Image, Label Image in a Batch
    for x, y in tqdm(loader):

        # Storing the Images to the Device
        x = x.to(device)
        y = y.to(device)
        # Set Gradient of all parameters to 0
        # Using Unscaled Mixed Precision using half Bit for Faster Processing
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # Get Predictions from Model
            y_pred = model(x)

            # Calculate the Loss
            y_pred = torch.squeeze(y_pred, dim=1)
            # y = torch.unsqueeze(y, 1)
            loss = loss_fn(y_pred, y)/opt_step_size
            
            if task_type == 'multiclass':
                y_pred = torch.argmax(y_pred, dim=1)
            score += metric(y_pred, y)
            score2 += metric2(y_pred, y)

        # Scale Loss Backwards
        scaler.scale(loss).mean().backward()

        # Unscale the Gradients in Optimizer
        if iters % opt_step_size == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Add the Loss for every sample in a Batch
        loss_for_epoch += loss.item()
        iters+=1

    
    if iters % opt_step_size == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    # Calculating The Average Loss for the Epoch
    loss_for_epoch =  torch.div(loss_for_epoch, len(loader))
    score =  torch.div(score, len(loader))
    score2 =  torch.div(score2, len(loader))
    return loss_for_epoch, score, score2

class ExponentialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target, time, toa, fps=30.0):
        target_cls = target[:, 1]
        target_cls = target_cls.to(torch.long)
        penalty = -torch.max(torch.zeros_like(toa).to(toa.device, pred.dtype), (toa.to(pred.dtype)-time-1)/fps)
        pos_loss = -torch.mul(torch.exp(penalty), -self.ce_loss(pred, target_cls))
        neg_loss = self.ce_loss(pred, target_cls)
        loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
        return loss

def evaluate_model(model, loader, loss_fn, device, metric, metric2):
    total_loss = 0.0
    model.eval()

    # 【核心修正 1】在驗證開始前，強制清空計數器，切斷與訓練集或歷史輪次的污染
    metric.reset()
    metric2.reset()

    with torch.no_grad():
        for batch in tqdm(loader):
            x, y, time_idx, toa = batch
            time_idx = time_idx.to(device)
            toa = toa.to(device)

            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)

            # One-hot encoding for Loss
            y_onehot = F.one_hot(y.long(), num_classes=2).float()
            loss = loss_fn(y_pred, y_onehot, time_idx, toa)
            total_loss += loss.item()
            
            # 【核心修正 2】使用 .update() 餵入數據，讓指標在內部累積混淆矩陣的原始數值
            probs = torch.softmax(y_pred, dim=1)[:, 1]
            metric.update(probs, y)
            metric2.update(probs, y)
            
        # 【核心修正 3】整個 Epoch 結束後，呼叫 .compute() 算出唯一且正確的全域指標
        epoch_f1 = metric.compute()
        epoch_acc = metric2.compute()
        
        total_loss = torch.div(total_loss, len(loader))
        
    return total_loss, epoch_f1, epoch_acc

def train_model(model, loader, optimizer, loss_fn, device, scaler, metric, metric2, *args, **kwargs):
    # 初始化這一輪的總損失為 0
    loss_for_epoch = 0.0 
    
    # Model 進入訓練模式
    model.train()

    # 【核心修正 1】在這一輪訓練開始前，強制清空計數器，切斷歷史輪次的統計污染
    metric.reset()
    metric2.reset()

    for batch in tqdm(loader):
        x, y, time_idx, toa = batch
        time_idx = time_idx.to(device)
        toa = toa.to(device)

        # 儲存影像到設備
        x = x.to(device, dtype=torch.float16)
        y = y.to(device)
        optimizer.zero_grad()

        # 使用混合精度加速
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # 取得模型預測值 (Logits)
            y_pred = model(x)

            # 計算損失
            if len(y.shape) == 1:
                y_onehot = F.one_hot(y.long(), num_classes=2).float()
            else:
                y_onehot = y
            loss = loss_fn(y_pred, y_onehot, time_idx, toa)

            # y_pred 是 Logits -> Softmax -> 取第二個欄位 (事故機率)
            probs = torch.softmax(y_pred, dim=1)[:, 1]
            
            # 【核心修正 2】改用 .update() 餵入數據，讓 TorchMetrics 在內部老實累積混淆矩陣
            metric.update(probs, y)
            metric2.update(probs, y)

        # 梯度反向傳播
        scaler.scale(loss).mean().backward()
        scaler.unscale_(optimizer)

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        scaler.step(optimizer)
        scaler.update()

        # 累加 Batch 損失
        loss_for_epoch += loss.item()

    # 【核心修正 3】整個訓練集跑完後，呼叫 .compute() 算出數學上完全精準的全域 F1 與 Acc
    epoch_f1 = metric.compute()
    epoch_acc = metric2.compute()
    
    # 計算平均損失
    loss_for_epoch = torch.div(loss_for_epoch, len(loader))
    
    return loss_for_epoch, epoch_f1, epoch_acc

def test_model(model, loader, device):
    """
    用於測試期推論的函數。
    核心邏輯：將模型輸出的片段級機率 (B, 2) 廣播至影格級維度 (B, T)，以符合 Benchmark 指標對 (N, T) 的要求。
    
    :param model: 封裝了 DataParallel 的 VideoEncoder 模型
    :param loader: test_loader (測試集資料載入器)
    :param device: 運算設備 (cuda 或 cpu)
    :return: all_pred (N x T), all_labels (N,), all_toas (N,) 的 NumPy 陣列
    """
    model.eval()
    all_pred = []
    all_labels = []
    all_toas = []
    
    # 從 loader 的 test_collate_fn 中安全取得模型型態，用來判定時間軸 T 的維度位置
    model_type = getattr(loader.collate_fn, 'model_type', 'VidNeXt')
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Inference] Processing Batches"):
            x, y, time_idx, toa = batch
            x = x.to(device)
            
            # 模型前向傳播 (得到 Logits)
            y_pred = model(x)
            
            # 計算事故預測機率 (過 Softmax 並取 index 1)，形狀為 (B,)
            probs = torch.softmax(y_pred, dim=1)[:, 1].cpu().numpy()
            
            # 根據模型類型的維度排列方式，動態獲取當前影片的總影格數 T
            if model_type in ['VidNeXt', 'ConvNeXtVanillaTransformer', 'ResNetNSTtransformer', 'ViViT']:
                T = x.shape[2]  # 維度順序為 (B, C, T, H, W)
            else:
                T = x.shape[1]  # 維度順序為 (B, T, C, H, W)
                
            # 將片段級機率 (B,) 沿著時間軸複製 T 次，包裝成 Benchmark 需要的影格級機率 (B, T)
            pred_frames = np.repeat(probs[:, np.newaxis], T, axis=1)
            
            all_pred.append(pred_frames)
            all_labels.append(y.cpu().numpy())
            all_toas.append(toa.cpu().numpy())
            
    # 將所有 Batch 的資料在第 0 維度集計、拼接成全域的大矩陣
    all_pred = np.concatenate(all_pred, axis=0)    # 最終形狀: (N, T)
    all_labels = np.concatenate(all_labels, axis=0)  # 最終形狀: (N,)
    all_toas = np.concatenate(all_toas, axis=0)      # 最終形狀: (N,)
    
    return all_pred, all_labels, all_toas