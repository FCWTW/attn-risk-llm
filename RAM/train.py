import cv2
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader import FetchData, ValCollator, TrainCollator
from get_model import VideoEncoder
from utils import train_model, evaluate_model, epoch_time, train_model_acc_grad, ExponentialLoss
from torchmetrics.classification import Accuracy, MulticlassF1Score, BinaryAccuracy, BinaryF1Score
from torchvision import transforms as video_transforms
import time
import plotly.graph_objects as go
import os
import argparse
import yaml
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.manual_seed(42)
np.random.seed(42)

class Train:
    def __init__(self, config_file, dataset_dir):
        self.configs = self.load_config(config_file)
        self.configs_dir = os.path.dirname(config_file)
        self.model_type = self.configs['model_class']
        self.model_params = self.configs['model_params']
        self.train_params = self.configs['train_params']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join('train_run', timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.log_file_path = os.path.join(self.run_dir, 'train_log.csv')
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            f.write("Epoch,Train_Loss,Train_F1,Train_Acc,Val_Loss,Val_F1,Val_Acc,Duration\n")

        # Setup dataloader
        train_dataset = FetchData(self.configs_dir, set_name='train', segment_duration=self.model_params['segment_length'], segment_interval=self.model_params['segment_overlap'], target_size=self.model_params['img_size'], after_acc=True, strategy=self.train_params['strategy'])
        train_collate_fn = TrainCollator(model_type=self.model_type, target_size=self.model_params['img_size'], root_dir=dataset_dir)
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.train_params['batch_size'],
            shuffle=True,
            num_workers=self.train_params['no_workers'],
            drop_last=True,
            pin_memory=False,
            collate_fn=train_collate_fn,
        )
        self.num_classes = train_dataset.num_classes
        val_dataset = FetchData(self.configs_dir, set_name='val', segment_duration=self.model_params['segment_length'], segment_interval=self.model_params['segment_overlap'], target_size=self.model_params['img_size'], after_acc=True)
        val_collate_fn = ValCollator(model_type=self.model_type, target_size=self.model_params['img_size'], root_dir=dataset_dir)
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.train_params['batch_size'],
            shuffle=False,
            num_workers=self.train_params['no_workers'],
            drop_last=False,
            pin_memory=False,
            collate_fn=val_collate_fn,
        )

        # Setup model
        print(f'-> Building {self.model_type}')
        self.model = VideoEncoder(model_type=self.model_type, num_classes=self.num_classes, segment_length=self.model_params['segment_length'])
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.train_params['lr'], weight_decay=0.1)
        
        # 【核心修改 1】重新啟用並優化 Scheduler 設定
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            patience=self.train_params['learning_rate_patience']
        )

    def load_config(self, config_file):
        with open(config_file, 'r') as fid:
            configs = yaml.safe_load(fid)
        return configs
    
    def train(self):
        loss_function = ExponentialLoss()
        scaler = torch.cuda.amp.GradScaler()
        metric = BinaryF1Score().to(self.device)
        metric2 = BinaryAccuracy().to(self.device)
        highest_score = 0
        training_loss = []
        training_score = []
        training_score2 = []
        val_loss = []
        val_score = []
        val_score2 = []
        bad_epochs = 0
        early_stop = False
        end_epoch = self.train_params['no_epochs']
        for epoch in range(self.train_params['no_epochs']):
            # Start Counting time
            start = time.time()

            # Train the Model for Every epoch
            train_value = train_model(self.model, self.train_loader, self.optimizer, loss_function, self.device, scaler, metric, metric2)
            training_loss.append(train_value[0].detach().cpu())
            training_score.append(train_value[1].detach().cpu())
            training_score2.append(train_value[2].detach().cpu())

            # Evaluate the Model using the val Split
            eval_value = evaluate_model(self.model, self.val_loader, loss_function, self.device, metric, metric2)
            val_loss.append((eval_value[0]).detach().cpu())
            val_score.append((eval_value[1]).detach().cpu())
            val_score2.append((eval_value[2]).detach().cpu())
            
            # Save the Model If the Model is Performing better on val set while Training
            if val_score[-1] > highest_score:
                bad_epochs = 0
                print(f"Val F1-Score improved from {highest_score:.4f} to {val_score[-1]:.4f}")
                highest_score = val_score[-1]
                model_name = f'{self.model_type}_epoch{epoch+1}_F1_{highest_score:.4f}.pt'
                torch.save(self.model.module.state_dict(), os.path.join(self.run_dir, model_name))
            else:
                bad_epochs += 1
            end = time.time()
            minutes, seconds = epoch_time(start, end)

            # 【核心修改 2】更新學習率調度器狀態
            self.scheduler.step(val_score[-1])
            current_lr = self.optimizer.param_groups[0]['lr']

            # Report Training and Val Loss
            print(f"Epoch Number: {epoch+1}")
            print(f"Duration: {minutes}m {seconds}s")
            print(f"Training Loss: {training_loss[-1]:.4f}")
            print(f"Training Score (F1): {training_score[-1]:.4f}")
            print(f"Training Score (Acc): {training_score2[-1]:.4f}")
            print(f"Val Loss: {val_loss[-1]:.4f}")
            print(f"Val Score (F1): {val_score[-1]:.4f}")
            print(f"Val Score (Acc): {val_score2[-1]:.4f}")
            print(f"Current Learning Rate: {current_lr:.8f}") # 印出當前 LR 方便追蹤
            print()

            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"{epoch+1},{training_loss[-1]:.4f},{training_score[-1]:.4f},{training_score2[-1]:.4f},"
                        f"{val_loss[-1]:.4f},{val_score[-1]:.4f},{val_score2[-1]:.4f},{minutes}m {seconds}s\n")

            # If Patience Level reached for Model not Performing better
            if bad_epochs == self.train_params['early_stop']:
                print("Stopped Early. The Model is not improving over val loss")
                end_epoch = epoch
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='Path to config file in YAML format')
    parser.add_argument('--dataset_dir', default='/home/wayne/Documents/MMAU', type=str, help='Path to MM-AU dataset')
    args = parser.parse_args()
    train = Train(args.config, args.dataset_dir)
    train.train()