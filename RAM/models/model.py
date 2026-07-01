from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import inspect
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
from torch.autograd import Variable


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, output_cor_dim):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = [0.5, 0.5]
        self.output_cor_dim = output_cor_dim
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
        self.dense1 = torch.nn.Linear(hidden_dim+output_cor_dim, 256)
        self.dense2 = torch.nn.Linear(256, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h, output_cor):

        out, h = self.gru(x, h)
        out = torch.cat([out, output_cor], dim=-1)
        out = F.dropout(out[:, -1], self.dropout[0])  # optional
        out = self.relu(self.dense1(out))
        out = F.dropout(out, self.dropout[1])
        out = self.dense2(out)

        return out, h


class CorGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(CorGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        return out, h


class flow_GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(flow_GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        return out, h


class SpatialAttention(nn.Module):
    """
    Applied soft attention on the hidden representation of all the objects in a frame.
    """

    def __init__(self, h_dim):
        super(SpatialAttention, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(h_dim, 1))  # hidden representation dimension is 256
        self.softmax = nn.Softmax(dim=1)
        import math
        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))

    def forward(self, h_all_in):
        """
        :h_all_in - dictionary containing object tracking id and hidden representation of size 2 x 1 x 256 each
        :output - dictionary with the same shape of h_all_in
        """
        k = []
        v = []
        for key in h_all_in:
            v.append(h_all_in[key])
            k.append(key)

        if len(v) != 0:
            h_in = torch.cat([element for element in v], dim=1)
            m = torch.tanh(h_in)
            alpha = torch.softmax(torch.matmul(m, self.weight), 1)
            roh = torch.mul(h_in, alpha)
            list_roh = []
            for i in range(roh.size(1)):
                list_roh.append(roh[:, i, :].unsqueeze(1).contiguous())

            h_all_in = {}
            for ke, value in zip(k, list_roh):
                h_all_in[ke] = value

        return h_all_in


class RiskyObject(nn.Module):
    def __init__(self, x_dim, h_dim, n_frames=100, fps=20.0):
        super(RiskyObject, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.fps = fps
        self.n_frames = n_frames
        self.n_layers = 2
        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())  # rgb

        # for secondary GRU
        self.n_layers_cor = 1
        self.h_dim_cor = 32
        self.gru_net = GRUNet(h_dim+h_dim, h_dim, 2, self.n_layers, self.h_dim_cor)
        self.weight = torch.Tensor([0.5, 1]).cuda()  # TO-DO: find the correct weight

        # input dim 4
        self.gru_net_cor = CorGRU(4, self.h_dim_cor, self.n_layers_cor)
        self.soft_attention = SpatialAttention(h_dim)
        self.soft_attention_cor = SpatialAttention(self.h_dim_cor)
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=self.weight, reduction='mean')

    def forward(self, x, y, toa, flow, hidden_in=None, testing=False):
        """
        :param x (batchsize, nFrames, 1+maxBox, Xdim)
        :param y (batchsize, nFrames, maxBox, 6)
        :toa (batchsize, 1)
        :batchsize = 1, currently we support batchsize=1
        """
        losses = {'cross_entropy': 0}
        h = Variable(torch.zeros(self.n_layers, x.size(0),  self.h_dim)
                     )  # TO-DO: hidden_in like dsta
        h = h.to(x.device)
        h_all_in = {}
        h_all_out = {}

        # hidden representation for secondary gru
        h_all_in_cor = {}
        h_all_out_cor = {}
        # h_all_in_flow = {}
        # h_all_out_flow = {}

        all_outputs = []
        all_labels = []

        for t in range(x.size(1)):
            # projecting to a lower dimensional space
            inp = flow[:, t]  # 1 x31 x2048

            # Flow----------------
            x_val = self.phi_x(inp)  # 1 x 31 x 256  #rgb_d
            img_embed = x_val[:, 0, :].unsqueeze(1)  # 1 x 1 x 256
            img_embed = img_embed.repeat(1, 30, 1)  # 1 x 30 x 256
            obj_embed = x_val[:, 1:, :]   # 1 x 30 x 256
            x_t = torch.cat([obj_embed, img_embed], dim=-1)  # 1 x 30 x 512

            h_all_out = {}
            h_all_out_cor = {}
            h_all_out_flow = {}
            frame_outputs = []
            frame_labels = []
            for bbox in range(30):
                if y[0][t][bbox][0] == 0:  # ignore if there is no bounding box
                    continue
                else:
                    track_id = str(y[0][t][bbox][0].cpu().detach().numpy())
                    if track_id in h_all_in:

                        # secondary GRU-----------------------------------
                        # decoding the coordinate with a secondary GRU model
                        unnormalized_cor = y[0][t][bbox]  # unnormalized coordinate (1080,720)scale
                        # print(d[1]/1080)
                        norm_cor = torch.Tensor([unnormalized_cor[1]/1080, unnormalized_cor[2]/720, unnormalized_cor[3] /
                                                 1080, unnormalized_cor[4]/720])  # normalized coordinate

                        norm_cor = torch.unsqueeze(norm_cor, 0)
                        norm_cor = torch.unsqueeze(norm_cor, 0)
                        norm_cor = norm_cor.to(x.device)

                        # hidden representation for coordinate gru
                        h_in_cor = h_all_in_cor[track_id]
                        output_cor, h_out_cor = self.gru_net_cor(norm_cor, h_in_cor)

                        h_all_out_cor[track_id] = h_out_cor

                        # base GRU---------------------------------------
                        h_in = h_all_in[track_id]  # 1x1x256

                        x_obj = x_t[0][bbox]  # 4096 # x_t[batch][frame][bbox]
                        x_obj = torch.unsqueeze(x_obj, 0)  # 1 x 512
                        x_obj = torch.unsqueeze(x_obj, 0)  # 1 x 1 x 512

                        output, h_out = self.gru_net(
                            x_obj, h_in, output_cor)  # 1x1x256
                        target = y[0][t][bbox][5].to(torch.long)
                        target = torch.as_tensor([target], device=torch.device('cuda'))

                        # compute error per object
                        loss = self.ce_loss(output, target)
                        losses['cross_entropy'] += loss
                        frame_outputs.append(output.detach().cpu().numpy())
                        frame_labels.append(y[0][t][bbox][5].detach().cpu().numpy())
                        h_all_out[track_id] = h_out  # storing in a dictionary

                    else:  # If object was not found in the previous frame

                        # secondary GRU --------------------------------------
                        unnormalized_cor = y[0][t][bbox]  # unnormalized coordinate (1080,720)scale
                        norm_cor = torch.Tensor([unnormalized_cor[1]/1080, unnormalized_cor[2]/720, unnormalized_cor[3] /
                                                 1080, unnormalized_cor[4]/720])  # normalized coordinate

                        norm_cor = torch.unsqueeze(norm_cor, 0)
                        norm_cor = torch.unsqueeze(norm_cor, 0)
                        norm_cor = norm_cor.to(x.device)

                        # hidden representation for coordinate gru
                        h_in_cor = Variable(torch.zeros(
                            self.n_layers_cor, x.size(0),  self.h_dim_cor))

                        h_in_cor = h_in_cor.to(x.device)

                        output_cor, h_out_cor = self.gru_net_cor(norm_cor, h_in_cor)
                        # Base GRU------------------------------------------
                        h_in = Variable(torch.zeros(self.n_layers, x.size(0),  self.h_dim)
                                        )  # TO-DO: hidden_in like dsta
                        h_in = h_in.to(x.device)
                        x_obj = x_t[0][bbox]  # 512
                        x_obj = torch.unsqueeze(x_obj, 0)  # 1 x 512
                        x_obj = torch.unsqueeze(x_obj, 0)  # 1 x 1 x 512

                        output, h_out = self.gru_net(
                            x_obj, h_in, output_cor)  # 1x1x256
                        target = y[0][t][bbox][5].to(torch.long)
                        target = torch.as_tensor([target], device=torch.device('cuda'))
                        loss = self.ce_loss(output, target)
                        losses['cross_entropy'] += loss
                        frame_outputs.append(output.detach().cpu().numpy())
                        frame_labels.append(y[0][t][bbox][5].detach().cpu().numpy())
                        h_all_out[track_id] = h_out  # storing in a dictionary
                        h_all_out_cor[track_id] = h_out_cor

            all_outputs.append(frame_outputs)
            all_labels.append(frame_labels)
            h_all_in = {}
            h_all_in = h_all_out.copy()

            h_all_in = self.soft_attention(h_all_in)

            h_all_in_cor = {}
            h_all_in_cor = h_all_out_cor.copy()
            h_all_in_cor = self.soft_attention_cor(h_all_in_cor)

        return losses, all_outputs, all_labels

class RiskyObject_v2(nn.Module):
    def __init__(self, x_dim, h_dim, n_frames=100, fps=20.0):
        super(RiskyObject_v2, self).__init__()  # 修正了原始代碼中的類別名稱錯位

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.fps = fps
        self.n_frames = n_frames
        self.n_layers = 2
        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())  # rgb

        # 🔥 修改點 1：因為加入了 1 維的 Saliency_Prior，主 GRU 的輸入維度由 512 變更為 513
        self.gru_net = GRUNet(h_dim + h_dim + 1, h_dim, 2, self.n_layers, self.h_dim_cor)
        self.weight = torch.Tensor([0.5, 1]).cuda()  

        # input dim 4
        self.gru_net_cor = CorGRU(4, self.h_dim_cor, self.n_layers_cor)
        self.soft_attention = SpatialAttention(h_dim)
        self.soft_attention_cor = SpatialAttention(self.h_dim_cor)
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=self.weight, reduction='mean')

    def forward(self, x, y, toa, flow, hidden_in=None, testing=False):
        """
        :param x (batchsize, nFrames, 1+maxBox, Xdim) -> 借用為元數據
        :param y (batchsize, nFrames, maxBox, 7)    -> [Track_ID, x1, y1, x2, y2, Risk_Label, Saliency_Prior]
        :toa (batchsize, 1)
        :batchsize = 1
        """
        losses = {'cross_entropy': 0}
        h = Variable(torch.zeros(self.n_layers, x.size(0),  self.h_dim))
        h = h.to(x.device)
        h_all_in = {}
        h_all_out = {}

        h_all_in_cor = {}
        h_all_out_cor = {}

        all_outputs = []
        all_labels = []

        for t in range(x.size(1)):
            inp = flow[:, t]  # 1 x 31 x 2048

            # Flow----------------
            x_val = self.phi_x(inp)  # 1 x 31 x 256
            img_embed = x_val[:, 0, :].unsqueeze(1)  # 1 x 1 x 256
            img_embed = img_embed.repeat(1, 30, 1)  # 1 x 30 x 256
            obj_embed = x_val[:, 1:, :]   # 1 x 30 x 256
            x_t = torch.cat([obj_embed, img_embed], dim=-1)  # 1 x 30 x 512

            h_all_out = {}
            h_all_out_cor = {}
            h_all_out_flow = {}
            frame_outputs = []
            frame_labels = []
            
            for bbox in range(30):
                if y[0][t][bbox][0] == 0:  # ignore if there is no bounding box
                    continue
                else:
                    track_id = str(y[0][t][bbox][0].cpu().detach().numpy())
                    if track_id in h_all_in:

                        # secondary GRU-----------------------------------
                        unnormalized_cor = y[0][t][bbox]  
                        norm_cor = torch.Tensor([unnormalized_cor[1]/1080, unnormalized_cor[2]/720, unnormalized_cor[3]/1080, unnormalized_cor[4]/720])  

                        norm_cor = torch.unsqueeze(norm_cor, 0)
                        norm_cor = torch.unsqueeze(norm_cor, 0)
                        norm_cor = norm_cor.to(x.device)

                        h_in_cor = h_all_in_cor[track_id]
                        output_cor, h_out_cor = self.gru_net_cor(norm_cor, h_in_cor)

                        h_all_out_cor[track_id] = h_out_cor

                        # base GRU---------------------------------------
                        h_in = h_all_in[track_id]  

                        x_obj = x_t[0][bbox]  
                        x_obj = torch.unsqueeze(x_obj, 0)  # 1 x 512
                        x_obj = torch.unsqueeze(x_obj, 0)  # 1 x 1 x 512

                        # 🔥 修改點 2-A：在既有軌跡分支中，切出第 7 欄先驗純量，對齊維度並執行 Early Fusion
                        s_prior = y[0][t][bbox][6].view(1, 1, 1).to(x.device)
                        x_obj = torch.cat([x_obj, s_prior], dim=-1)  # 1 x 1 x 513

                        output, h_out = self.gru_net(x_obj, h_in, output_cor)  
                        
                        target = y[0][t][bbox][5].to(torch.long)
                        target = torch.as_tensor([target], device=torch.device('cuda'))

                        loss = self.ce_loss(output, target)
                        losses['cross_entropy'] += loss
                        frame_outputs.append(output.detach().cpu().numpy())
                        frame_labels.append(y[0][t][bbox][5].detach().cpu().numpy())
                        h_all_out[track_id] = h_out  

                    else:  # If object was not found in the previous frame

                        # secondary GRU --------------------------------------
                        unnormalized_cor = y[0][t][bbox]  
                        norm_cor = torch.Tensor([unnormalized_cor[1]/1080, unnormalized_cor[2]/720, unnormalized_cor[3]/1080, unnormalized_cor[4]/720])  

                        norm_cor = torch.unsqueeze(norm_cor, 0)
                        norm_cor = torch.unsqueeze(norm_cor, 0)
                        norm_cor = norm_cor.to(x.device)

                        h_in_cor = Variable(torch.zeros(self.n_layers_cor, x.size(0),  self.h_dim_cor))
                        h_in_cor = h_in_cor.to(x.device)

                        output_cor, h_out_cor = self.gru_net_cor(norm_cor, h_in_cor)
                        
                        # Base GRU------------------------------------------
                        h_in = Variable(torch.zeros(self.n_layers, x.size(0),  self.h_dim))  
                        h_in = h_in.to(x.device)
                        
                        x_obj = x_t[0][bbox]  # 512
                        x_obj = torch.unsqueeze(x_obj, 0)  # 1 x 512
                        x_obj = torch.unsqueeze(x_obj, 0)  # 1 x 1 x 512

                        # 🔥 修改點 2-B：在新出現物件分支中，同步切出第 7 欄先驗純量並執行 Early Fusion
                        s_prior = y[0][t][bbox][6].view(1, 1, 1).to(x.device)
                        x_obj = torch.cat([x_obj, s_prior], dim=-1)  # 1 x 1 x 513

                        output, h_out = self.gru_net(x_obj, h_in, output_cor)  
                        
                        target = y[0][t][bbox][5].to(torch.long)
                        target = torch.as_tensor([target], device=torch.device('cuda'))
                        loss = self.ce_loss(output, target)
                        losses['cross_entropy'] += loss
                        frame_outputs.append(output.detach().cpu().numpy())
                        frame_labels.append(y[0][t][bbox][5].detach().cpu().numpy())
                        h_all_out[track_id] = h_out  
                        h_all_out_cor[track_id] = h_out_cor

            all_outputs.append(frame_outputs)
            all_labels.append(frame_labels)
            h_all_in = {}
            h_all_in = h_all_out.copy()

            h_all_in = self.soft_attention(h_all_in)

            h_all_in_cor = {}
            h_all_in_cor = h_all_out_cor.copy()
            h_all_in_cor = self.soft_attention_cor(h_all_in_cor)

        return losses, all_outputs, all_labels

if __name__ == '__main__':
    # 檢查是否支援 CUDA，因為模型代碼中強制使用了 .cuda()
    if not torch.cuda.is_available():
        print("錯誤：此模型代碼中包含硬編碼的 .cuda() 調用，必須在支援 GPU 的環境下運行。")
        sys.exit(1)

    device = torch.device('cuda')
    print(f"使用裝置: {device}")

    # --- 1. 定義超參數 (Hyperparameters) ---
    # 根據你的代碼邏輯推斷的維度
    BATCH_SIZE = 1        # 代碼限制 batchsize=1
    N_FRAMES = 10         # 模擬 10 個 Frame
    MAX_BOX = 30          # 代碼中迴圈範圍是 range(30)
    X_DIM = 2048          # 假設輸入特徵維度 (例如 ResNet 的輸出)
    H_DIM = 256           # Hidden layer 維度

    # --- 2. 初始化模型 ---
    print("初始化模型...")
    model = RiskyObject(x_dim=X_DIM, h_dim=H_DIM, n_frames=N_FRAMES)
    model = model.to(device) # 將模型搬移至 GPU

    # --- 3. 建立 Dummy Inputs ---
    print("生成測試數據...")

    # (1) x: 用於獲取序列長度，代碼中似乎未直接用於特徵提取，主要用 flow
    # 形狀: (batch, frames, 1+max_box, feature_dim)
    # 這裡 1+max_box 是因為代碼中有 img_embed (index 0) 和 obj_embed (index 1~30) 的區分
    x_dummy = torch.randn(BATCH_SIZE, N_FRAMES, 1 + MAX_BOX, X_DIM).to(device)

    # (2) flow: 實際放入 phi_x 進行特徵提取的輸入
    flow_dummy = torch.randn(BATCH_SIZE, N_FRAMES, 1 + MAX_BOX, X_DIM).to(device)

    # (3) y: 包含 Bounding Box 資訊與 Label
    # 形狀: (batch, frames, max_box, 6)
    # Channel 6 的定義: [Track_ID, x1, y1, x2, y2, Label]
    y_dummy = torch.zeros(BATCH_SIZE, N_FRAMES, MAX_BOX, 6).to(device)

    # 填入一些假數據以觸發邏輯
    for t in range(N_FRAMES):
        for b in range(5): # 假設每個 frame 前 5 個物件是存在的
            # Index 0: Track ID (必須大於 0 才會被處理)
            y_dummy[0, t, b, 0] = float(b + 1) 
            
            # Index 1-4: 座標 (Unnormalized, 假設 1920x1080)
            y_dummy[0, t, b, 1] = 100.0 # x1
            y_dummy[0, t, b, 2] = 100.0 # y1
            y_dummy[0, t, b, 3] = 200.0 # x2
            y_dummy[0, t, b, 4] = 200.0 # y2
            
            # Index 5: Label (0 或 1)
            y_dummy[0, t, b, 5] = float(np.random.randint(0, 2))

    # (4) toa: Time of Accident (在此測試中可能未被深度使用，給個 dummy)
    toa_dummy = torch.tensor([[N_FRAMES - 1]]).to(device)

    # --- 4. 執行 Forward Pass ---
    print("開始 Forward Pass...")
    try:
        # 根據 forward 定義: def forward(self, x, y, toa, flow, hidden_in=None, testing=False):
        losses, all_outputs, all_labels = model(x_dummy, y_dummy, toa_dummy, flow_dummy)
        
        print("\n--- 測試成功 ---")
        print(f"Loss Cross Entropy: {losses['cross_entropy'].item():.4f}")
        
        # 簡單檢查輸出格式
        print(f"輸出 Frame 數: {len(all_outputs)}")
        if len(all_outputs) > 0:
            print(f"第一個 Frame 的物件輸出數量: {len(all_outputs[0])}")
            if len(all_outputs[0]) > 0:
                print(f"單個物件輸出形狀: {all_outputs[0][0].shape}")

    except Exception as e:
        print(f"\n執行錯誤: {e}")
        import traceback
        traceback.print_exc()