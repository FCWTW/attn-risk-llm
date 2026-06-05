import cv2
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader import FetchData, TestCollator
from get_model import VideoEncoder
from utils import epoch_time, ExponentialLoss, test_model 
from torchmetrics.classification import Accuracy, MulticlassF1Score, BinaryAccuracy, BinaryF1Score
from torchvision import transforms as video_transforms
from sklearn.metrics import roc_auc_score  # 引入 Benchmark 所需的 AUC 計算工具
import time
import plotly.graph_objects as go
import os
import argparse
import yaml
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.manual_seed(42)
np.random.seed(42)

def evaluate_earliness(all_pred, all_labels, time_of_accidents, fps=30.0, thresh=0.5):
    """
    Evaluate the earliness for true positive videos
    Modified from https://github.com/JWFanggit/LOTVS-CAP/blob/main/src/eval_tools.py
    """
    time = 0.0
    counter = 0
    for i in range(len(all_pred)):
        pred_bins = (all_pred[i] >= thresh).astype(int)
        inds_pos = np.where(pred_bins > 0)[0]
        if all_labels[i] > 0 and len(inds_pos) > 0:
            time += max((time_of_accidents[i] - inds_pos[0]) / fps, 0)
            counter += 1  
    mTTA = time / counter if counter > 0 else 0 
    return mTTA

def evaluation(all_pred, all_labels, time_of_accidents, fps=30.0):
    """
    Compute AP (Average Precision), mTTA, and TTA@R80
    Modified from https://github.com/JWFanggit/LOTVS-CAP/blob/main/src/eval_tools.py
    """
    preds_eval = []
    min_pred = np.inf
    n_frames = 0
    for idx, toa in enumerate(time_of_accidents):
        if all_labels[idx] > 0:
            pred = all_pred[idx, :int(toa)]  
        else:
            pred = all_pred[idx, :]  
        min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
        preds_eval.append(pred)
        n_frames += len(pred)
    total_seconds = all_pred.shape[1] / fps

    Precision = np.zeros((n_frames))
    Recall = np.zeros((n_frames))
    Time = np.zeros((n_frames))
    cnt = 0
    for Th in np.arange(max(min_pred, 0), 1.0, 0.1):
        Tp = 0.0
        Tp_Fp = 0.0
        time = 0.0
        counter = 0.0  
        for i in range(len(preds_eval)):
            tp =  np.where(preds_eval[i]*all_labels[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter+1
            Tp_Fp += float(len(np.where(preds_eval[i]>=Th)[0])>0)
        if Tp_Fp == 0:  
            continue
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) == 0: 
            continue
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1

    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    
    _, rep_index = np.unique(Recall, return_index=1)
    rep_index = rep_index[1:]
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])
    
    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    
    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1, len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    mTTA = np.mean(new_Time) * total_seconds
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))] * total_seconds
    return AP, mTTA, TTA_R80

class Test:
    def __init__(self, config_file, dataset_dir, model_path):
        self.configs = self.load_config(config_file)
        self.configs_dir = os.path.dirname(config_file)
        self.model_type = self.configs['model_class']
        self.model_params = self.configs['model_params']
        self.test_params = self.configs['test_params']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup dataloader
        test_dataset = FetchData(self.configs_dir, set_name='test', segment_duration=self.model_params['segment_length'], segment_interval=1, target_size=self.model_params['img_size'], after_acc=True, strategy=self.test_params['strategy'])
        test_collate_fn = TestCollator(model_type=self.model_type, target_size=self.model_params['img_size'], root_dir=dataset_dir)
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.test_params['batch_size'],
            shuffle=False,
            num_workers=self.test_params['no_workers'],
            drop_last=False,
            pin_memory=False,
            collate_fn=test_collate_fn,
        )

        # Setup model
        print(f'-> Building {self.model_type}')
        self.num_classes = test_dataset.num_classes
        self.model = VideoEncoder(model_type=self.model_type, num_classes=self.num_classes, segment_length=self.model_params['segment_length'])
        print(f'-> Loading weights from {model_path}')
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def load_config(self, config_file):
        with open(config_file, 'r') as fid:
            configs = yaml.safe_load(fid)
        return configs
    
    def test(self):
        print("-> Starting inference on Test Set...")
        # 調用未來在 utils.py 中實現的推論函數，獲取全域的預測矩陣、標籤與車禍發生幀
        all_pred, all_labels, all_toas = test_model(self.model, self.test_loader, self.device)
        
        print("-> Inference finished. Calculating benchmark metrics...")
        
        # 1. 計算 Earliness (mTTA@0.5 閾值)
        mTTA_05 = evaluate_earliness(all_pred, all_labels, all_toas, fps=30, thresh=0.5)
        
        # 2. 計算 AP, 全域 mTTA, 以及 TTA@Recall 80%
        AP, mTTA, TTA_R80 = evaluation(all_pred, all_labels, all_toas, fps=30)
        
        # 3. 計算影片級別的 v-AUC (取車禍發生後的預測極值作為影片特徵分數)
        all_vid_scores = [max(pred[int(toa):]) if len(pred[int(toa):]) > 0 else 0.0 for toa, pred in zip(all_toas, all_pred)]
        AUC = roc_auc_score(all_labels, all_vid_scores)
        
        # 4. 漂亮地打印出與 Benchmark 格式完全對齊的測試報告
        print("\n" + "="*50)
        print("BENCHMARK TEST REPORT".center(50))
        print("="*50)
        print(f"Target Model      : {self.model_type}")
        print(f"[Earliness]       : mTTA@0.5 = {mTTA_05:.4f} seconds")
        print(f"[Correctness]     : AP       = {AP:.4f}")
        print(f"                    mTTA     = {mTTA:.4f} seconds")
        print(f"                    TTA_R80  = {TTA_R80:.4f} seconds")
        print(f"[Video-level AUC] : v-AUC    = {AUC:.5f}")
        print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='Path to config file in YAML format')
    parser.add_argument('--dataset_dir', default='/home/wayne/Documents/MMAU', type=str, help='Path to MM-AU dataset')
    parser.add_argument('--model', required=True, type=str, help='Path to .pt')
    args = parser.parse_args()
    test = Test(args.config, args.dataset_dir, args.model)
    test.test()