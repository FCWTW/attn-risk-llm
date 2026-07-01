# import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, average_precision_score

import matplotlib.pyplot as plt
import os
import numpy as np

def evaluate_earliness(all_pred, all_labels, time_of_accidents, fps=30.0, thresh=0.5):
    """Evaluate the earliness for true positive videos"""
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

def official_evaluation(all_pred, all_labels, time_of_accidents, fps=30.0):
    """
    對手論文官方評估指標解算器
    """
    preds_eval = []
    min_pred = np.inf
    n_frames = 0
    for idx, toa in enumerate(time_of_accidents):
        if all_labels[idx] > 0:
            # 確保切片索引不越界
            slice_idx = min(int(toa), all_pred.shape[1])
            pred = all_pred[idx, :slice_idx]  
        else:
            pred = all_pred[idx, :]  
        
        if len(pred) > 0:
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
            if i >= len(all_labels): continue
            tp = np.where(preds_eval[i] * all_labels[i] >= Th)
            Tp += float(len(tp[0]) > 0)
            if float(len(tp[0]) > 0) > 0 and time_of_accidents[i] > 0:
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter + 1
            Tp_Fp += float(len(np.where(preds_eval[i] >= Th)[0]) > 0)
            
        if Tp_Fp == 0:  
            continue
        else:
            Precision[cnt] = Tp / Tp_Fp
        if np.sum(all_labels) == 0: 
            continue
        else:
            Recall[cnt] = Tp / np.sum(all_labels)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1 - time / counter)
        cnt += 1
        
    # 限幅防禦
    Precision = Precision[:cnt]
    Recall = Recall[:cnt]
    Time = Time[:cnt]

    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    
    _, rep_index = np.unique(Recall, return_index=True)
    if len(rep_index) <= 1:
        return 0.0, 0.0, 0.0
        
    rep_index = rep_index[1:]
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    
    for i in range(len(rep_index) - 1):
        new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
        new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])
        
    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    
    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0] * (new_Recall[0] - 0)
    for i in range(1, len(new_Precision)):
        AP += (new_Precision[i-1] + new_Precision[i]) * (new_Recall[i] - new_Recall[i-1]) / 2

    mTTA = np.mean(new_Time) * total_seconds
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    TTA_R80 = sort_time[np.argmin(np.abs(sort_recall - 0.8))] * total_seconds

    return AP, mTTA, TTA_R80

def evaluation(all_pred, all_labels, epoch):
    fpr, tpr, thresholds = metrics.roc_curve(np.array(all_labels), np.array(all_pred), pos_label=1)
    # np.savez('auc.npz', fpr=fpr, tpr=tpr, thresholds=thresholds)
    roc_auc = metrics.auc(fpr, tpr)
    return fpr, tpr, roc_auc


def plot_auc_curve(fpr, tpr, roc_auc, epoch):
    curve_dir = 'charts/auc/'
    if not os.path.exists(curve_dir):
        os.makedirs(curve_dir)
    auc_curve_file = os.path.join(curve_dir, 'auc_%02d.png' % (epoch))

    plt.title(f'Receiver Operating Characteristic at epoch: {epoch}')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(auc_curve_file)
    plt.close()


def plot_pr_curve(all_labels, all_pred, epoch):
    pr_dir = 'charts/pr/'
    if not os.path.exists(pr_dir):
        os.makedirs(pr_dir)
    pr_curve_file = os.path.join(pr_dir, 'pr_%02d.png' % (epoch))
    precision, recall, thresholds = precision_recall_curve(np.array(all_labels), np.array(all_pred))
    # np.savez('ap_attention_bbox_flow.npz', precision=precision,
    #          recall=recall, thresholds=thresholds)
    ap = average_precision_score(np.array(all_labels), np.array(all_pred))

    plt.title(f'Precision-Recall Curve at epoch: {epoch}')
    plt.plot(recall, precision, 'b', label='AP = %0.2f' % ap)
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig(pr_curve_file)
    plt.close()
    return ap


def frame_auc(output, labels):
    # print(output)
    output = np.array(output)
    labels = np.array(labels)
    # print(output)
    all_pred = []
    all_labels = []

    for t in range(len(output)):
        frame = output[t]
        frame_score = []
        frame_label = []
        print(frame)

        if len(frame) == 0:
            continue
        else:
            for j in range(len(frame)):
                score = np.exp(frame[j][:, 1])/np.sum(np.exp(frame[j]), axis=1)
                frame_score.append(score)
                frame_label.append(labels[t][j]+0)
            all_pred.append(max(frame_score))
            all_labels.append(sum(frame_label))

    new_labels = []
    for i in all_labels:
        if i > 0.0:
            new_labels.append(1.0)
        else:
            new_labels.append(0.0)

    fpr, tpr, thresholds = metrics.roc_curve(np.array(new_labels), np.array(all_pred), pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc