from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import date
import time

import torch
from torch.utils.data import DataLoader
from models.model import RiskyObject
from models.evaluation import evaluation, plot_auc_curve, plot_pr_curve, frame_auc
from dataloader import MyDataset
import argparse
from tqdm import tqdm
import os
import logging
import numpy as np
import csv
import yaml

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Train:
    def __init__(self, data_dir, config_file, output_dir):
        self.data_dir = data_dir
        self.configs = self.load_config(config_file)
        self.model_params = self.configs['model_params']
        self.train_params = self.configs['train_params']
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup Logging
        print('-> Initializing the log file...')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'train.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Config File: {config_file} | Output Dir : {os.path.abspath(self.output_dir)} | Learning Rate: {self.train_params['lr']}")

        self.x_dim = self.model_params['x_dim']
        self.h_dim = self.model_params['h_dim']
        self.n_frame = 150
        self.fps = 30
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        print('-> Building data loader...')
        train_data = MyDataset(self.data_dir, os.path.dirname(config_file), 'train', self.train_params['strategy'], toTensor=True, device=self.device)
        val_data = MyDataset(self.data_dir, os.path.dirname(config_file), 'val', self.train_params['strategy'], toTensor=True, device=self.device)
        
        self.train_dataloader = DataLoader(
            dataset=train_data, batch_size=self.train_params['batch_size'], shuffle=True, drop_last=True
        )
        self.val_dataloader = DataLoader(
            dataset=val_data, batch_size=self.train_params['batch_size'], shuffle=False, drop_last=True
        )

        print('-> Building model...')
        self.model = RiskyObject(self.x_dim, self.h_dim, self.n_frame, self.fps) 
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_params['lr'])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.train_params['lr'],
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )   
        self.model = self.model.to(device=self.device)    
        
        self.auc_max = 0
        self.ap_max = 0
        
        # Setup CSV log
        today = date.today().strftime("%b-%d-%Y")
        current_time = time.strftime("%H-%M-%S", time.localtime())
        self.result_csv = os.path.join(self.output_dir, f'result_{today}_{current_time}.csv')
        with open(self.result_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"data_path: {self.data_dir}"])
            writer.writerow([f"x_dim: {self.model.x_dim}, base_h_dim: {self.model.h_dim}"])
            writer.writerow(['epoch', 'loss_val', 'roc_auc', 'ap'])

    def load_config(self, config_file):
        with open(config_file, 'r') as fid:
            configs = yaml.safe_load(fid)
        return configs
    
    def _load_checkpoint(self, filename):
        start_epoch = -1
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logger.info(f"-> Checkpoint loaded from {filename} (epoch {start_epoch})")
        else:
            self.logger.warning(f"[ERROR] no checkpoint found at '{filename}'")
        return start_epoch

    def train_epoch(self, epoch):
        self.model.train()
        loop = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc=f"Epoch [{epoch}/{self.train_params['epoch']}]")
        epoch_losses = []
        
        for i, (batch_det, batch_toas, batch_flow, batch_vid) in loop:
            self.optimizer.zero_grad()
            losses, _, _ = self.model(
                x=batch_flow,
                y=batch_det,
                toa=batch_toas,
                flow=batch_flow
            )
            total_bboxes = (batch_det[0, :, :, 0] != 0).sum().item()

            loss_ce = losses['cross_entropy']
            if not isinstance(loss_ce, torch.Tensor):
                # This indicates that this clip does not contain any tracked objects; skip this training round.
                continue
            loss = loss_ce.mean()
            loss = loss / total_bboxes
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
            loop.set_postfix(loss=loss.item())
            
        avg_loss = np.mean(epoch_losses)
        lr = self.optimizer.param_groups[0]['lr']
        self.logger.info(f"Epoch [{epoch}] Train Loss: {avg_loss:.4f} | LR: {lr}")
        return avg_loss

    def val_epoch(self, epoch):
        self.model.eval()
        losses_all = []
        all_pred = []
        all_labels = []
        loop = tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader), desc=f"Epoch [{epoch}/{self.train_params['epoch']}]")

        with torch.no_grad():
            for i, (batch_det, batch_toas, batch_flow, batch_vid) in loop:
                losses, all_outputs, labels = self.model(
                    x=batch_flow,
                    y=batch_det,
                    toa=batch_toas,
                    flow=batch_flow
                )
                total_bboxes = (batch_det[0, :, :, 0] != 0).sum().item()
                loss_ce = losses['cross_entropy']
                if isinstance(loss_ce, torch.Tensor):
                    losses_all.append(loss_ce.mean().item() / total_bboxes)
                # losses_all.append(losses['cross_entropy'].mean().item())
                
                for t in range(len(all_outputs)):
                    frame = all_outputs[t]
                    if len(frame) == 0:
                        continue
                    for j in range(len(frame)):
                        score = np.exp(frame[j][:, 1]) / np.sum(np.exp(frame[j]), axis=1)
                        all_pred.append(float(score[0]))
                        all_labels.append(int(labels[t][j]))

        loss_val = np.mean(losses_all)
        fpr, tpr, roc_auc = evaluation(all_pred, all_labels, epoch)
        plot_auc_curve(fpr, tpr, roc_auc, epoch)
        ap = plot_pr_curve(all_labels, all_pred, epoch)

        self.logger.info(f"Epoch [{epoch}] Val Loss: {loss_val:.4f} | AUC: {roc_auc:.4f} | AP: {ap:.4f}")

        with open(self.result_csv, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss_val, roc_auc, ap])

        if roc_auc > self.auc_max:
            self.auc_max = roc_auc
            best_auc_file = os.path.join(self.output_dir, 'best_auc.pth')
            torch.save({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, best_auc_file)
            self.logger.info(f"Best AUC Model saved: {best_auc_file}")
            
        if ap > self.ap_max:
            self.ap_max = ap
            best_ap_file = os.path.join(self.output_dir, 'best_ap.pth')
            torch.save({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, best_ap_file)
            self.logger.info(f"Best AP Model saved: {best_ap_file}")

        return loss_val

    def train(self):
        start_epoch = -1
        
        if self.train_params.get('resume', False):
            print(f'-> Resume from checkpoint...')
            start_epoch = self._load_checkpoint(self.train_params.get('ckpt_file', ''))

        if self.train_params.get('tl', False):
            print('-> Applying transfer learning. Freezing base layers...')
            for name, param in self.model.named_parameters():
                if 'dense1' in name or 'dense2' in name or 'soft_attention' in name or 'soft_attention_cor' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for epoch in range(self.train_params['epoch']):
            if epoch <= start_epoch:
                continue
                
            train_loss = self.train_epoch(epoch)
            val_loss = self.val_epoch(epoch)
            
            self.scheduler.step(val_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/media/wayne/27CD255760735841/MMAU', help='Path to root dir of .npz files')
    parser.add_argument('--config', default='./config/RAM.yaml', type=str, help='Path to config file in YAML format')
    parser.add_argument('--output_dir', type=str, default='./result', help='Save dir')
    args = parser.parse_args()
    print(args)

    train = Train(args.data_dir, args.config, args.output_dir)
    train.train()