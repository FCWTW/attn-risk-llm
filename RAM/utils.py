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

def train_model(model, loader, optimizer, loss_fn, device, scaler, metric, metric2, opt_step_size=1, model_type='vidnext'):
    # Intializing starting Epoch loss as 0
    loss_for_epoch = 0.0 
    score = 0.0   
    score2 = 0.0   
    # Model to be used in Training Mode
    model.train()

    for batch in tqdm(loader):
        x, y, time_idx, toa = batch
        time_idx = time_idx.to(device)
        toa = toa.to(device)

        # Storing the Images to the Device
        x = x.to(device, dtype=torch.float16)
        y = y.to(device)
        optimizer.zero_grad()

        # Using Unscaled Mixed Precision using half Bit for Faster Processing
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # Get Predictions from Model
            y_pred = model(x)

            # Calculate the loss
            # 假設修改後的 VidNeXt 的輸出是 [Batch, 1]，因此要將 y 變成 One-hot 格式
            if len(y.shape) == 1:
                y_onehot = F.one_hot(y.long(), num_classes=2).float()
            else:
                y_onehot = y
            # 確保 y_pred 是 [Batch, 2]
            if y_pred.shape[1] == 1: 
                # 如果模型只輸出 1 個值，我們需要把它變成 2 個值的 logits (這比較少見，通常設 num_classes=2)
                # 這裡假設你的 get_model 會因為 task='ex' 而設定 num_classes=2
                pass
            loss = loss_fn(y_pred, y_onehot, time_idx, toa)

            # y_pred 是 Logits -> Softmax -> 取第二個欄位 (事故機率)
            probs = torch.softmax(y_pred, dim=1)[:, 1]
            
            # Metric 庫通常吃 (preds, target)
            # BinaryAccuracy 吃 (prob, label_idx)
            score += metric(probs, y) 
            score2 += metric2(probs, y)
        # Scale Loss Backwards
        scaler.scale(loss).mean().backward()

        # Unscale the Gradients in Optimizer
        scaler.unscale_(optimizer)

        # Clip the Gradients to they dontreach inf
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        scaler.step(optimizer)
        
        # Update the Scaler
        scaler.update()

        # Add the Loss for every sample in a Batch
        loss_for_epoch += loss.item()
    # Calculating The Average Loss for the Epoch
    loss_for_epoch =  torch.div(loss_for_epoch, len(loader))
    score =  torch.div(score, len(loader))
    score2 =  torch.div(score2, len(loader))
    return loss_for_epoch, score, score2

# Function to Evaluate the Model
def evaluate_model(model, loader, loss_fn, device, metric, metric2):
    # Intializing starting Epoch loss as 0
    total_loss = 0.0
    score = 0.0
    score2 = 0.0
    # Model to be used in Evaluation Mode
    model.eval()

    # Gradients are not calculated
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
            # Metric
            probs = torch.softmax(y_pred, dim=1)[:, 1]
            score += metric(probs, y)
            score2 += metric2(probs, y)
            total_loss += loss.item()
        # Calculating The Average Loss for the Epoch
        total_loss =  torch.div(total_loss, len(loader))
        score =  torch.div(score, len(loader))
        score2 =  torch.div(score2, len(loader))
        
    return total_loss, score, score2
