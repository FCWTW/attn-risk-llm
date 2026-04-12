import torch
from torch import nn
import torch.nn.functional as F
import math
import os
from collections import OrderedDict
from video_swin_transformer import SwinTransformer3DBackbone
from einops import rearrange
import time
from utils import get_task_attribute_dict
import logging
import datetime

class SCOUT_task(nn.Module):
    def __init__(self, 
                  num_encoder_layers=4,
                  use_task=False,
                  task_attributes=None,
                  train_backbone=False,
                  pretrained_backbone=False,
                  transformer_params=None,
                  **kwargs
                ):

        super(SCOUT_task, self).__init__()

        self.backbone = SwinTransformer3DBackbone(pretrained=pretrained_backbone,
                                                train_backbone=train_backbone)
        self.task_attributes = get_task_attribute_dict(task_attributes)
        self.num_encoder_layers = num_encoder_layers
        if num_encoder_layers in [1, 2, 3, 4]:
            self.decoder = DecoderSwin(num_encoder_layers)
        else:
            raise ValueError(f'ERROR: unsupported num_encoder_layers={num_encoder_layers}, \
                            should be between 1 and 4.')

        self.use_task = use_task

        self.add_and_norm = transformer_params['add_and_norm']
        self.fuse_idx = transformer_params['fuse_idx']
        self.num_att_heads = transformer_params['num_att_heads']

        if self.use_task:
            embed_dims = [768, 384, 192, 96]

            self.task_context_model = TaskContextEncoder(clip_size=16, 
                                    task_attributes=self.task_attributes,
                                    fuse_idx=self.fuse_idx)

            multihead_attn_layers = [None, None, None, None]
            norm_layers = [None, None, None, None]
            
            for idx in self.fuse_idx:
                multihead_attn_layers[idx] = nn.MultiheadAttention(embed_dim=embed_dims[idx],
                                                        num_heads=self.num_att_heads[idx], 
                                                        bias=True)
                norm_layers[idx] = nn.LayerNorm(embed_dims[idx])


            self.multihead_attn = nn.ModuleList(multihead_attn_layers)
            self.norm = nn.ModuleList(norm_layers)



    def forward(self, x, task_input):

        b_out = self.backbone(x)
        b_s = [b.shape for b in b_out]

        if self.use_task:
            task_enc = self.task_context_model(task_input)
            for idx, b in enumerate(b_out):
                if idx in self.fuse_idx:
                    task_enc[idx] = task_enc[idx].flatten(2).permute((2, 0, 1))
                    b = b.flatten(2).permute((2, 0, 1))

                    fused_out, _ = self.multihead_attn[idx](task_enc[idx], b, b)
                
                    if self.add_and_norm:
                        fused_out += task_enc[idx]
                        fused_out = self.norm[idx](fused_out)

                    fused_out = fused_out.permute((1, 2, 0))
                    b_out[idx] = fused_out.view(*b_s[idx])
        
        return self.decoder(b_out[:self.num_encoder_layers])


class SCOUT_map_v1(nn.Module):
    def __init__(self, 
                  num_encoder_layers=4,
                  use_map=True,
                  map_params=None,
                  train_backbone=False,
                  pretrained_backbone=False,
                  transformer_params=None,
                  **kwargs
                ):

        super(SCOUT_map_v1, self).__init__()

        self.backbone_3d = SwinTransformer3DBackbone(pretrained=pretrained_backbone,
                                                train_backbone=train_backbone)

        self.num_encoder_layers = num_encoder_layers

        if num_encoder_layers in [1, 2, 3, 4]:
            self.decoder = DecoderSwin(num_encoder_layers)
        else:
            raise ValueError(f'ERROR: unsupported num_encoder_layers={num_encoder_layers}, \
                            should be between 1 and 4.')

        self.use_map = use_map

        self.add_and_norm = transformer_params['add_and_norm']
        self.fuse_idx = transformer_params['fuse_idx']
        self.num_att_heads = transformer_params['num_att_heads']

        if self.use_map:
            num_channels = 1 + len([x for x in map_params.keys() if isinstance(map_params[x], bool) and map_params[x]])

            self.map_encoder = MapEncoder(input_size=(num_channels, *map_params['img_size']), fuse_idx=self.fuse_idx)


            multihead_attn_layers = [None, None, None, None]
            norm_layers = [None, None, None, None]

            embed_dims = [768, 384, 192, 96]
            
            for idx in self.fuse_idx:
                multihead_attn_layers[idx] = nn.MultiheadAttention(embed_dim=embed_dims[idx],
                                                        num_heads=self.num_att_heads[idx], 
                                                        bias=True)
                norm_layers[idx] = nn.LayerNorm(embed_dims[idx])


            self.multihead_attn = nn.ModuleList(multihead_attn_layers)
            self.norm = nn.ModuleList(norm_layers)

    def forward(self, x, map_input):

        b_out = self.backbone_3d(x)
        b_s = [b.shape for b in b_out]

        if self.use_map:
            map_enc = self.map_encoder(map_input)

            for idx, b in enumerate(b_out):
                if idx in self.fuse_idx:
                    
                    #print('map_enc', idx, map_enc[idx].shape)
                    #print('b', idx, b.shape)

                    map_enc[idx] = map_enc[idx].flatten(2).permute((2, 0, 1))					
                    b = b.flatten(2).permute((2, 0, 1))

                    fused_out, _ = self.multihead_attn[idx](map_enc[idx], b, b)
                
                    if self.add_and_norm:
                        fused_out += map_enc[idx]
                        fused_out = self.norm[idx](fused_out)

                    fused_out = fused_out.permute((1, 2, 0))
                    b_out[idx] = fused_out.view(*b_s[idx])
        
        return self.decoder(b_out[:self.num_encoder_layers])


# model with only map input
class SCOUT_map_v2(nn.Module):
    def __init__(self, 
                  num_encoder_layers=4,
                  use_map=True,
                  map_params=None,
                  train_backbone=False,
                  pretrained_backbone=False,
                  transformer_params=None,
                  **kwargs
                ):

        super(SCOUT_map_v2, self).__init__()

        #self.backbone_3d = SwinTransformer3DBackbone(pretrained=pretrained_backbone,
        #											train_backbone=train_backbone)

        self.num_encoder_layers = num_encoder_layers

        if num_encoder_layers in [1, 2, 3, 4]:
            self.decoder = DecoderSwin(num_encoder_layers)
        else:
            raise ValueError(f'ERROR: unsupported num_encoder_layers={num_encoder_layers}, \
                            should be between 1 and 4.')

        self.use_map = use_map

        self.add_and_norm = transformer_params['add_and_norm']
        self.fuse_idx = transformer_params['fuse_idx']
        self.num_att_heads = transformer_params['num_att_heads']

        num_channels = 1 + map_params['obs_traj'] + map_params['coords'] + map_params['dist']

        self.map_encoder = MapEncoder(input_size=(num_channels, *map_params['img_size']), fuse_idx=self.fuse_idx)

    def forward(self, x, map_input):

        b_out = self.map_encoder(map_input)		
        return self.decoder(b_out[:self.num_encoder_layers]) 


class TaskContextEncoder(nn.Module):
    def __init__(self,
                 clip_size=16,
                 dict_len=30,
                 fuse_idx=(0, 1, 2, 3),
                 task_attributes=None):
        super(TaskContextEncoder, self).__init__()
        self.embedding = nn.Embedding(dict_len, clip_size//4)
        self.relu = nn.ReLU()
        self.task_attributes = get_task_attribute_dict(task_attributes)
        self.fuse_idx = fuse_idx
        self.emb_dims = (768, 384, 192, 96)
        self.repl_dims = (49, 196, 784, 3136)
        num_features = len([k for k, v in self.task_attributes.items() if v]) # number of task and context features
        if num_features == 0:
            raise ValueError('ERROR: no task attributes provided')

        dense_layers = [None, None, None, None]

        for idx in fuse_idx:
            dense_layers[idx] = nn.Linear(num_features, self.emb_dims[idx]) 
        self.dense = nn.ModuleList(dense_layers)


    def forward(self, task_context):

        for k in task_context.keys():	
            if len(task_context[k].shape) == 2:
                task_context[k] = self.embedding(task_context[k])
            else:
                task_context[k] = task_context[k][:, :, ::4]
        #print(task_context.keys())
        task_context_enc = torch.stack([v for k,v in task_context.items()], dim=1)
        #print('task_context_enc', task_context_enc.shape)

        task_context_enc = task_context_enc.permute((0, 3, 1, 2)).flatten(2)
        #print('task_context_enc', task_context_enc.shape)
        task = [None, None, None, None]
        for idx in self.fuse_idx:
            task[idx] = self.relu(self.dense[idx](task_context_enc)).permute((0, 2, 1))
            task[idx] = task[idx][:,:,:, None]
            task[idx] = task[idx].repeat(1, 1, 1, self.repl_dims[idx])
            #print('task', idx, task[idx].shape)
        return task

# Map encoder modified from
# https://github.com/StanfordASL/Trajectron-plus-plus/blob/1031c7bd1a444273af378c1ec1dcca907ba59830/trajectron/model/components/map_encoder.py
class MapEncoder(nn.Module):
    def __init__(self,
                 input_size=(1, 128, 128),
                 fuse_idx=(0, 1, 2, 3)):
        super(MapEncoder, self).__init__()

        hidden_channels = [10, 20, 10, 1]
        kernel_size = (5, 3, 3, 1)
        strides = (2, 2, 1, 1)
        output_size = (56, 56)

        self.fuse_idx = fuse_idx
        self.repl_dims = ([1, 768, 4, 1, 1], [1, 384, 4, 1, 1], [1, 192, 4, 1, 1], [1, 96, 4, 1, 1])

        self.convs = nn.ModuleList()
        self.post = nn.ModuleList()

        x_dummy = torch.ones(input_size).unsqueeze(0) * torch.tensor(float('nan'))

        for i, hidden_size in enumerate(hidden_channels):
            self.convs.append(nn.Conv2d(input_size[0] if i == 0 else hidden_channels[i-1],
                                        hidden_channels[i], kernel_size[i],
                                        stride=strides[i]))
            x_dummy = self.convs[i](x_dummy)
            #print(x_dummy.shape)
        
        self.post.append(nn.AvgPool2d(kernel_size=1, stride=4))
        self.post.append(nn.AvgPool2d(kernel_size=1, stride=2))
        self.post.append(nn.Identity())
        self.post.append(nn.Upsample(size=output_size))


    def forward(self, x):
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)
        
        map_enc = [None, None, None, None]
        for i in range(len(self.post)):
            if i in self.fuse_idx:
                #print('x', i, x.shape)
                map_enc[i] = self.post[i](x)[:,:,None,:,:].repeat(self.repl_dims[i])
                #print('map_enc', i, map_enc[i].shape)
        return map_enc

class DecoderSwin(nn.Module):
    def __init__(self, num_layers=4):
        super(DecoderSwin, self).__init__()
        
        self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)
        
        self.convtsp1 = nn.Sequential(
            nn.Conv3d(768, 384, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
            nn.ReLU(),
            self.upsampling
        )

        x = 1 if num_layers == 1 else 3

        self.convtsp2 = nn.Sequential(
            nn.Conv3d(384, 192, kernel_size=(x, 3, 3), stride=(x, 1, 1), padding=(0,1,1), bias=False),
            nn.ReLU(),
            self.upsampling
        )

        x = 1 if num_layers < 4 else 5

        self.convtsp3 = nn.Sequential(
            nn.Conv3d(192, 96, kernel_size=(x,3,3), stride=(x,1,1), padding=(0,1,1), bias=False),
            nn.ReLU(),
            self.upsampling
        )

        x = 1 if num_layers < 3 else 5
        layers = [('conv3_1', nn.Conv3d(96, 64, kernel_size=(x,3,3), stride=(x,1,1), padding=(0,1,1), bias=False)),
                              ('relu_1', nn.ReLU()),
                              ('up_1', self.upsampling),
                              ('conv3_2', nn.Conv3d(64, 32, kernel_size=(1,3,3), stride=(2,1,1), padding=(0,1,1), bias=False)),
                              ('relu_2', nn.ReLU()),
                              ('up_2', self.upsampling)
                              
                ]
        if num_layers == 1:
            layers.append(('conv3_3', nn.Conv3d(32, 1, kernel_size=(1,1,1), stride=(2,1,1), bias=True)))
        else:
            layers.append(('conv3_3', nn.Conv3d(32, 1, kernel_size=(1,1,1), stride=(1,1,1), bias=True)))

        layers.append(('sigm', nn.Sigmoid()))

        self.convtsp4 = nn.Sequential(OrderedDict(layers))

    def forward(self, y):
        if not isinstance(y, list):
            raise ValueError(f'ERROR: input to decoder should be a list!')

        if len(y) >= 1:
            z = self.convtsp1(y[0])

        if len(y) >= 2:
            z = torch.cat((z,y[1]), 2)
        
        z = self.convtsp2(z)

        if len(y) >= 3:
            z = torch.cat((z,y[2]), 2)
        
        z = self.convtsp3(z)

        if len(y) == 4:
            z = torch.cat((z,y[3]), 2)
        
        z = self.convtsp4(z)
        
        z = z.view(z.size(0), z.size(3), z.size(4))
        return z

class GraphConvolution(nn.Module):
    """
    Modify from https://github.com/JWFangit/LOTVS-DADA/blob/master/SCAFNet/nets.py#L103
    Basic graph convolution layer (GCN) as in https://arxiv.org/abs/1609.02907
    Input: features=[batch, node, C_in], adj = [batch, node, node]
    Output: [batch, node, C_out]
    """
    def __init__(self, in_features, out_features, activation=None, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        # Glorot uniform initialization for weights (類似 Keras 的 glorot_uniform)
        # nn.init.kaiming_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, features, basis):
        # features: (B, N, C_in)
        # basis:    (B, N, N)
        
        # 1. K.batch_dot(basis, features) -> torch.bmm(basis, features)
        # B x N x N @ B x N x C_in -> B x N x C_in
        supports = torch.bmm(basis, features)
        
        # 2. K.dot(supports, self.kernel) -> supports @ self.weight
        # B x N x C_in @ C_in x C_out -> B x N x C_out
        output = supports @ self.weight

        if self.bias is not None:
            output = output + self.bias
        
        if self.activation is not None:
            output = self.activation(output)
            
        return output
    
class SGcn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SGcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Embedding layers for similarity calculation
        self.sim_embed1 = nn.Linear(in_channels, in_channels)
        self.sim_embed2 = nn.Linear(in_channels, in_channels)
        
        # GCN Layers
        self.graph1 = GraphConvolution(in_channels, in_channels, activation=F.relu)
        self.graph2 = GraphConvolution(in_channels, in_channels, activation=F.relu)
        self.graph3 = GraphConvolution(in_channels, out_channels, activation=F.relu)

        self.ln1 = nn.LayerNorm(in_channels)
        self.ln2 = nn.LayerNorm(in_channels)

    def get_adj(self, x):
        # x: (Batch, Nodes, Channels)
        sim1 = self.sim_embed1(x)
        sim2 = self.sim_embed2(x)
        
        # adj: (Batch, Nodes, Nodes)
        adj = torch.bmm(sim1, sim2.transpose(1, 2))
        scale = self.in_channels ** -0.5
        adj = adj * scale
        adj = F.softmax(adj, dim=-1)
        return adj

    def forward(self, x):
        # Input x is expected to be (Batch, Channels, H, W)
        b, c, h, w = x.size()
        
        # Reshape to (Batch, Nodes, Channels) where Nodes = H*W
        x_reshaped = x.view(b, c, -1).permute(0, 2, 1)
        
        adj = self.get_adj(x_reshaped)
        
        outs = self.graph1(x_reshaped, adj)
        outs = self.ln1(outs)
        outs = self.graph2(outs, adj)
        outs = self.ln2(outs)
        outs = self.graph3(outs, adj)
        
        # Reduce mean over nodes (dim 1) -> (Batch, OutChannels)
        outs = torch.mean(outs, dim=1)
        
        # Expand dims to match (Batch, OutChannels, 1, 1) for broadcasting
        outs = outs.view(b, self.out_channels, 1, 1)
        
        return outs

class Seg_encoder(nn.Module):
    """
    Modify from https://github.com/JWFangit/LOTVS-DADA/blob/master/SCAFNet/nets.py#L103
    """
    def __init__(self):
        super(Seg_encoder, self).__init__()
        
        # Block 1
        self.conv1a = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm3d(64)
        self.conv1b = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d((1, 2, 2))

        # Block 2
        self.conv2a = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm3d(128)
        self.conv2b = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d((1, 2, 2))

        # Block 3
        self.conv3a = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm3d(256)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm3d(256)
        self.conv3c = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.bn3c = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d((1, 2, 2))

        # Block 4
        self.conv4a = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn4a = nn.BatchNorm3d(512)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn4b = nn.BatchNorm3d(512)
        self.conv4c = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn4c = nn.BatchNorm3d(512)

        # SGcn
        self.sgcn = SGcn(in_channels=512, out_channels=512)
        self.final_norm = nn.BatchNorm3d(512)

    def forward(self, x):
        # Input: (Batch, Channels, Time, Height, Width)
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = F.relu(self.bn3c(self.conv3c(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4a(self.conv4a(x)))
        x = F.relu(self.bn4b(self.conv4b(x)))
        x = F.relu(self.bn4c(self.conv4c(x)))

        b, c, t, h, w = x.size()
        x = self.sgcn(x.permute(0, 2, 1, 3, 4).contiguous().view(b*t, c, h, w))
        x_out = self.final_norm(x.view(b, t, c ,1, 1).permute(0, 2, 1, 3, 4))
        return x_out
    
class Seg_encoder_v2(nn.Module):
    """
    Modify from https://github.com/JWFangit/LOTVS-DADA/blob/master/SCAFNet/nets.py#L103
    """
    def __init__(self):
        super(Seg_encoder_v2, self).__init__()
        
        # Block 1
        self.conv1a = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm3d(64)
        self.conv1b = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d((1, 2, 2))

        # Block 2
        self.conv2a = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm3d(128)
        self.conv2b = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d((1, 2, 2))

        # SGcn
        self.sgcn = SGcn(in_channels=128, out_channels=128)
        self.final_norm = nn.BatchNorm3d(128)

    def forward(self, x):
        # Input: (Batch, Channels, Time, Height, Width)
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)

        b, c, t, h, w = x.size()
        x = self.sgcn(x.permute(0, 2, 1, 3, 4).contiguous().view(b*t, c, h, w))
        x_out = self.final_norm(x.view(b, t, c ,1, 1).permute(0, 2, 1, 3, 4))
        return x_out
    
class Seg_encoder_v3(nn.Module):
    def __init__(self):
        super(Seg_encoder_v3, self).__init__()
        
        # Block 1 (保持不變)
        self.conv1a = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm3d(64)
        self.conv1b = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d((1, 2, 2))

        # Block 2 (保持不變)
        self.conv2a = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm3d(128)
        self.conv2b = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d((1, 2, 2))

        # [修改點] Adapter 定義改變
        # 移除了 Pooling 層，因為我們要在 forward 裡面手動做 Spatial Pooling
        # 這裡只負責通道數的映射與數值正規化
        self.adapter = nn.Sequential(
            nn.Linear(128, 512),             # 投影到 512
            nn.LayerNorm(512),               # 穩定數值
            nn.Tanh()                        # 限制輸出範圍
        )

    def forward(self, x):
        # Input: (Batch, 3, Time, Height, Width)
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x) 
        # 此時 x 的形狀為: (B, 128, T, H', W')

        # [修改點] 關鍵邏輯修正
        
        # 1. Spatial Global Average Pooling (只平均 H 和 W，保留 T)
        # dim=(3, 4) 對應 H, W 維度
        x = x.mean(dim=(3, 4), keepdim=True) 
        # 形狀變為: (B, 128, T, 1, 1)

        # 2. 調整形狀以通過 Linear 層
        b, c, t, _, _ = x.size()
        x = x.view(b, c, t)       # (B, 128, T)
        x = x.permute(0, 2, 1)    # (B, T, 128) -> Linear 預設對最後一維做運算

        # 3. 通過 Adapter (對每個時間點獨立運算)
        x = self.adapter(x)       # (B, T, 512)

        # 4. 還原形狀給 SCOUT 模型使用
        x = x.permute(0, 2, 1)    # (B, 512, T)
        x = x.view(b, 512, t, 1, 1) # (B, 512, T, 1, 1)
        return x

class SCOUT_seg_v1(nn.Module):
    def __init__(self, 
                  num_encoder_layers=4,
                  train_backbone=False,
                  pretrained_backbone=False,
                  transformer_params=None,
                  **kwargs
                ):

        super(SCOUT_seg_v1, self).__init__()

        self.attn_logger = logging.getLogger('SCOUT_Attn')
        self.attn_logger.setLevel(logging.INFO)
        if not self.attn_logger.handlers:
            filename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            fh = logging.FileHandler(f'cache/{filename}', mode='w')
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            fh.setFormatter(formatter)
            self.attn_logger.addHandler(fh)
        self.log_step_counter = 0
        self.log_frequency = 100 # 設定每 10 個 batch 紀錄一次 (設為 1 則紀錄所有)

        self.backbone_3d = SwinTransformer3DBackbone(pretrained=pretrained_backbone,
                                                train_backbone=train_backbone)

        self.num_encoder_layers = num_encoder_layers

        if num_encoder_layers in [1, 2, 3, 4]:
            self.decoder = DecoderSwin(num_encoder_layers)
        else:
            raise ValueError(f'ERROR: unsupported num_encoder_layers={num_encoder_layers}, \
                            should be between 1 and 4.')

        self.seg_encoder = Seg_encoder()

        # self.add_and_norm = transformer_params['add_and_norm']
        # self.fuse_idx = transformer_params['fuse_idx']
        # self.num_att_heads = transformer_params['num_att_heads']
        self.add_and_norm = True
        self.fuse_idx = [0, 1, 2]
        self.num_att_heads = [8, 4, 2, 2]

        # 準備空的 ModuleList
        self.multihead_attn = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.seg_projectors = nn.ModuleList() # 用於存放 512 -> C 的映射層

        embed_dims = [768, 384, 192, 96]
        
        # 我們必須遍歷所有可能的層數 (0 到 3)，填滿 ModuleList
        # 即使該層不在 fuse_idx 中，也要放一個佔位符 (Identity)，保持索引對齊
        for i in range(4):
            if i in self.fuse_idx:
                # 如果這層需要融合，則實例化真實的層
                self.multihead_attn.append(
                    nn.MultiheadAttention(embed_dim=embed_dims[i],
                                          num_heads=self.num_att_heads[i], 
                                          bias=True)
                )
                self.norm.append(nn.LayerNorm(embed_dims[i]))
                
                # 線性映射層：將 Seg_encoder 的 512 維轉為 Backbone 對應維度
                self.seg_projectors.append(nn.Linear(512, embed_dims[i]))
            else:
                # 如果這層不融合，填入 Identity 防止索引錯誤，且滿足 ModuleList 要求
                self.multihead_attn.append(nn.Identity())
                self.norm.append(nn.Identity())
                self.seg_projectors.append(nn.Identity())
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, seg_input):

        # 保存原始形狀 (B, C, T, H, W)
        b_out = self.backbone_3d(x)
        b_s = [b.shape for b in b_out]

        # 輸出形狀: (B, 512, T, 1, 1)
        seg_out = self.seg_encoder(seg_input)

        # 更新計數器 (用於控制 log 頻率)
        if self.training:
            self.log_step_counter += 1

        for idx, b in enumerate(b_out):
            if idx in self.fuse_idx:
                
                #print('map_enc', idx, map_enc[idx].shape)
                #print('b', idx, b.shape)
                
                # Query
                b_flat = b.flatten(2).permute((2, 0, 1))

                # Key and Value
                curr_seg = seg_out.squeeze(-1).squeeze(-1) # (B, 512, T)
                curr_seg = curr_seg.permute(0, 2, 1)       # (B, T, 512) for Linear
                curr_seg = self.seg_projectors[idx](curr_seg) # (B, T, C)
                seg_flat = curr_seg.permute(1, 0, 2) # (T, B, C)

                # fused_out, _ = self.multihead_attn[idx](b_flat, seg_flat, seg_flat)
                fused_out, attn_weights = self.multihead_attn[idx](
                    b_flat, seg_flat, seg_flat, need_weights=True
                )

                '''
                # --- 暫時監控代碼 ---
                if self.training: # 只在訓練模式印出，避免測試時洗板
                    with torch.no_grad():
                        # attn_weights shape: (Batch, N_backbone_pixels, N_seg_frames)
                        # 我們觀察對每個像素來說，分配給不同時間幀的權重分佈
                        max_w = attn_weights.max().item()
                        mean_w = attn_weights.mean().item()
                        std_w = attn_weights.std(dim=-1).mean().item() # 每個像素權重的標準差
                        
                        # 理論上均勻分佈的標準差會接近 0
                        # 如果 std 遠小於 0.01，代表模型對所有語義資訊都一視同仁（即失敗的分佈）
                        if torch.isnan(attn_weights).any():
                            print(f"[Layer {idx}] ALERT: Attention weights contains NaN!")
                        else:
                            print(f"[Layer {idx}] Attn Stats -> Max: {max_w:.4f}, Mean: {mean_w:.4f}, Std: {std_w:.6f}")
                # ------------------
                '''
                
                # --- [修改] Log 監控代碼 ---
                # 條件：是訓練模式 且 (計數器符合頻率 或 第一個 Batch)
                if self.training and (self.log_step_counter % self.log_frequency == 0 or self.log_step_counter == 1):
                    with torch.no_grad():
                        max_w = attn_weights.max().item()
                        mean_w = attn_weights.mean().item()
                        std_w = attn_weights.std(dim=-1).mean().item()
                        
                        log_msg = f"[Step {self.log_step_counter}][Layer {idx}] Attn Stats -> Max: {max_w:.4f}, Mean: {mean_w:.4f}, Std: {std_w:.6f}"
                        
                        if torch.isnan(attn_weights).any():
                            self.attn_logger.warning(f"[Step {self.log_step_counter}][Layer {idx}] ALERT: Attention weights contains NaN!")
                        else:
                            # 寫入檔案，而不是印在 terminal
                            self.attn_logger.info(log_msg)
                # --------------------------
                
                fused_out = self.dropout(fused_out)
                if self.add_and_norm:
                    fused_out += b_flat
                    fused_out = self.norm[idx](fused_out)

                fused_out = fused_out.permute((1, 2, 0))
                b_out[idx] = fused_out.view(*b_s[idx])
        
        return self.decoder(b_out[:self.num_encoder_layers])

class SCOUT_seg_v2(nn.Module):
    def __init__(self, 
                  num_encoder_layers=4,
                  train_backbone=False,
                  pretrained_backbone=False,
                  transformer_params=None,
                  **kwargs
                ):

        super(SCOUT_seg_v2, self).__init__()

        self.attn_logger = logging.getLogger('SCOUT_Attn')
        self.attn_logger.setLevel(logging.INFO)
        if not self.attn_logger.handlers:
            filename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            fh = logging.FileHandler(f'cache/{filename}', mode='w')
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            fh.setFormatter(formatter)
            self.attn_logger.addHandler(fh)
        self.log_step_counter = 0
        self.log_frequency = 100 # 設定每 10 個 batch 紀錄一次 (設為 1 則紀錄所有)
        
        self.backbone_3d = SwinTransformer3DBackbone(pretrained=pretrained_backbone,
                                            train_backbone=train_backbone,
                                            drop_path_rate=0.2,  # 建議 0.2 ~ 0.3。這會隨機 "關掉" 某些層，強迫模型學習更魯棒的特徵
                                            attn_drop_rate=0.1,  # [選用] Attention 內部的 Dropout，通常 0.1 即可
                                            drop_rate=0.0)

        self.num_encoder_layers = num_encoder_layers

        if num_encoder_layers in [1, 2, 3, 4]:
            self.decoder = DecoderSwin(num_encoder_layers)
        else:
            raise ValueError(f'ERROR: unsupported num_encoder_layers={num_encoder_layers}, \
                            should be between 1 and 4.')

        self.seg_encoder = Seg_encoder_v3()

        self.add_and_norm = transformer_params['add_and_norm']
        self.fuse_idx = transformer_params['fuse_idx']
        self.num_att_heads = transformer_params['num_att_heads']

        # 準備空的 ModuleList
        self.multihead_attn = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.seg_projectors = nn.ModuleList() # 用於存放 512 -> C 的映射層

        embed_dims = [768, 384, 192, 96]
        
        # 我們必須遍歷所有可能的層數 (0 到 3)，填滿 ModuleList
        # 即使該層不在 fuse_idx 中，也要放一個佔位符 (Identity)，保持索引對齊
        for i in range(4):
            if i in self.fuse_idx:
                # 如果這層需要融合，則實例化真實的層
                self.multihead_attn.append(
                    nn.MultiheadAttention(embed_dim=embed_dims[i],
                                          num_heads=self.num_att_heads[i], 
                                          bias=True)
                )
                self.norm.append(nn.LayerNorm(embed_dims[i]))
                self.seg_projectors.append(nn.Linear(512, embed_dims[i]))
            else:
                # 如果這層不融合，填入 Identity 防止索引錯誤，且滿足 ModuleList 要求
                self.multihead_attn.append(nn.Identity())
                self.norm.append(nn.Identity())
                self.seg_projectors.append(nn.Identity())
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, seg_input):

        # 保存原始形狀 (B, C, T, H, W)
        b_out = self.backbone_3d(x)
        b_s = [b.shape for b in b_out]

        # 輸出形狀: (B, 512, T, 1, 1)
        seg_out = self.seg_encoder(seg_input)

        # 更新計數器 (用於控制 log 頻率)
        if self.training:
            self.log_step_counter += 1

        for idx, b in enumerate(b_out):
            if idx in self.fuse_idx:
                
                #print('map_enc', idx, map_enc[idx].shape)
                #print('b', idx, b.shape)
                
                # Query
                b_flat = b.flatten(2).permute((2, 0, 1))
                # b_flat = b_flat * 0.5

                # Key and Value
                curr_seg = seg_out.squeeze(-1).squeeze(-1) # (B, 512, T)
                curr_seg = curr_seg.permute(0, 2, 1)       # (B, T, 512) for Linear
                curr_seg = self.seg_projectors[idx](curr_seg) # (B, T, C)
                seg_flat = curr_seg.permute(1, 0, 2) # (T, B, C)

                # fused_out, _ = self.multihead_attn[idx](b_flat, seg_flat, seg_flat)
                fused_out, attn_weights = self.multihead_attn[idx](
                    b_flat, seg_flat, seg_flat, need_weights=True
                )
                
                # --- [修改] Log 監控代碼 ---
                # 條件：是訓練模式 且 (計數器符合頻率 或 第一個 Batch)
                if self.training and (self.log_step_counter % self.log_frequency == 0 or self.log_step_counter == 1):
                    with torch.no_grad():
                        max_w = attn_weights.max().item()
                        mean_w = attn_weights.mean().item()
                        std_w = attn_weights.std(dim=-1).mean().item()
                        
                        log_msg = f"[Step {self.log_step_counter}][Layer {idx}] Attn Stats -> Max: {max_w:.4f}, Mean: {mean_w:.4f}, Std: {std_w:.6f}"
                        
                        if torch.isnan(attn_weights).any():
                            self.attn_logger.warning(f"[Step {self.log_step_counter}][Layer {idx}] ALERT: Attention weights contains NaN!")
                        else:
                            # 寫入檔案，而不是印在 terminal
                            self.attn_logger.info(log_msg)
                # --------------------------
                
                fused_out = self.dropout(fused_out)
                if self.add_and_norm:
                    fused_out += b_flat
                    fused_out = self.norm[idx](fused_out)

                fused_out = fused_out.permute((1, 2, 0))
                b_out[idx] = fused_out.view(*b_s[idx])
        
        return self.decoder(b_out[:self.num_encoder_layers])
    
class SCOUT_seg_v3(nn.Module):
    def __init__(self, 
                  num_encoder_layers=4,
                  train_backbone=False,      # 確保這裡是 False
                  pretrained_backbone=False,
                  transformer_params=None,
                  **kwargs
                ):

        super(SCOUT_seg_v3, self).__init__()

        # --- Logger Setup ---
        self.attn_logger = logging.getLogger('SCOUT_Attn')
        self.attn_logger.setLevel(logging.INFO)
        if not self.attn_logger.handlers:
            filename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            fh = logging.FileHandler(f'cache/{filename}', mode='w')
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            fh.setFormatter(formatter)
            self.attn_logger.addHandler(fh)
        self.log_step_counter = 0
        self.log_frequency = 100
        # --------------------

        # 1. 骨幹網路 (完全凍結)
        self.backbone_3d = SwinTransformer3DBackbone(pretrained=pretrained_backbone, train_backbone=False)
        # 雙重保險：鎖死所有參數
        for param in self.backbone_3d.parameters():
            param.requires_grad = False
            
        self.num_encoder_layers = num_encoder_layers
        if num_encoder_layers in [1, 2, 3, 4]:
            self.decoder = DecoderSwin(num_encoder_layers)
        else:
            raise ValueError(f'ERROR: unsupported num_encoder_layers={num_encoder_layers}')

        self.seg_encoder = Seg_encoder_v3()
        self.add_and_norm = transformer_params['add_and_norm']
        self.fuse_idx = transformer_params['fuse_idx']
        self.num_att_heads = transformer_params['num_att_heads']

        # 2. 定義 Fusion 組件
        self.multihead_attn = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.seg_projectors = nn.ModuleList() # [保留] 用於映射 Seg 特徵
        self.adapters = nn.ModuleList()       # [新增] 用於適配 Backbone 特徵
        
        embed_dims = [768, 384, 192, 96]
        self.dropout = nn.Dropout(0.5)

        for i in range(4):
            if i in self.fuse_idx:
                # Attention 組件
                self.multihead_attn.append(
                    nn.MultiheadAttention(embed_dim=embed_dims[i],
                                          num_heads=self.num_att_heads[i], 
                                          bias=True)
                )
                self.norm.append(nn.LayerNorm(embed_dims[i]))
                
                # Seg Projector: 512 -> C
                self.seg_projectors.append(nn.Linear(512, embed_dims[i]))

                # Adapter: C -> C
                bottleneck_dim = embed_dims[i] // 4 

                adapter_layer = nn.Sequential(
                    # 1. 降維 (壓縮特徵，強迫抓重點)
                    nn.Conv3d(embed_dims[i], bottleneck_dim, kernel_size=1, bias=False),
                    nn.BatchNorm3d(bottleneck_dim),
                    nn.ReLU(inplace=True),
                    
                    # 2. 升維 (還原回原本維度)
                    nn.Conv3d(bottleneck_dim, embed_dims[i], kernel_size=1, bias=False),
                    nn.BatchNorm3d(embed_dims[i]),
                    # 這裡最後不加 ReLU，保持線性輸出以便與原特徵融合
                )
                nn.init.xavier_uniform_(adapter_layer[0].weight)
                
                self.adapters.append(adapter_layer)
            else:
                self.multihead_attn.append(nn.Identity())
                self.norm.append(nn.Identity())
                self.seg_projectors.append(nn.Identity())
                self.adapters.append(nn.Identity())

    def forward(self, x, seg_input):
        # 1. Backbone (No Grad)
        with torch.no_grad():
            b_out = self.backbone_3d(x)
        
        b_s = [b.shape for b in b_out] # 記錄原始形狀

        # 2. Seg Encoder (Trainable)
        # Output: (B, 512, T, 1, 1)
        seg_out = self.seg_encoder(seg_input)

        if self.training:
            self.log_step_counter += 1

        for idx, b in enumerate(b_out):
            if idx in self.fuse_idx:
                # --- [A] Process Backbone Feature (Query) ---
                # 通過 Adapter (Trainable)
                b = self.adapters[idx](b) 
                # Flatten: (B, C, T, H, W) -> (T*H*W, B, C) for Attention
                b_flat = b.flatten(2).permute((2, 0, 1))
                b_flat = b_flat * 2.0

                # --- [B] Process Seg Feature (Key/Value) ---
                # 確保形狀正確: (B, 512, T)
                curr_seg = seg_out.view(seg_out.size(0), 512, -1) 
                curr_seg = curr_seg.permute(0, 2, 1) # (B, T, 512)
                
                # Projector: (B, T, 512) -> (B, T, C)
                curr_seg = self.seg_projectors[idx](curr_seg) 
                
                # Flatten: (T, B, C)
                seg_flat = curr_seg.permute(1, 0, 2) 

                # --- [C] Attention Fusion ---
                fused_out, attn_weights = self.multihead_attn[idx](
                    b_flat, seg_flat, seg_flat, need_weights=True
                )
                
                # Logger (保持原本邏輯)
                if self.training and (self.log_step_counter % self.log_frequency == 0 or self.log_step_counter == 1):
                    with torch.no_grad():
                        max_w = attn_weights.max().item()
                        mean_w = attn_weights.mean().item()
                        std_w = attn_weights.std(dim=-1).mean().item()
                        self.attn_logger.info(f"[Step {self.log_step_counter}][Layer {idx}] Attn Stats -> Max: {max_w:.4f}, Mean: {mean_w:.4f}, Std: {std_w:.6f}")

                # Residual & Norm
                fused_out = self.dropout(fused_out)
                if self.add_and_norm:
                    fused_out += b_flat
                    fused_out = self.norm[idx](fused_out)

                # Reshape back to (B, C, T, H, W)
                fused_out = fused_out.permute((1, 2, 0))
                b_out[idx] = fused_out.view(*b_s[idx])
        
        return self.decoder(b_out[:self.num_encoder_layers])
    
class Seg_encoder_v4(nn.Module):
    def __init__(self, out_channels=256):
        super(Seg_encoder_v4, self).__init__()
        self.conv1a = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm3d(64)
        self.conv1b = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d((1, 2, 2))

        self.conv2a = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm3d(128)
        self.conv2b = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d((1, 2, 2))

        # 【修改 1】移除 Linear，改用 1x1x1 Conv3d 來保留空間維度 (Fully Convolutional)
        self.adapter = nn.Sequential(
            nn.Conv3d(128, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Input: (Batch, 3, Time, Height, Width)
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x) 
        
        # 【修改 2】徹底移除 x.mean(dim=(3, 4)) GAP 操作
        # 目前 x 的形狀是 (B, 128, T, H/4, W/4)

        # 【修改 3】直接通過 3D 卷積進行通道轉換
        x = self.adapter(x)       
        
        # Output shape: (B, 256, T, H/4, W/4)
        return x
    
class SCOUT_seg_v4(nn.Module):
    def __init__(self, 
                  num_encoder_layers=4,
                  train_backbone=False,
                  pretrained_backbone=False,
                  transformer_params=None,
                  **kwargs
                ):

        super(SCOUT_seg_v4, self).__init__()
        self.backbone_3d = SwinTransformer3DBackbone(pretrained=pretrained_backbone, train_backbone=train_backbone)
        # self.backbone_3d = SwinTransformer3DBackbone(pretrained=pretrained_backbone,
        #                                         train_backbone=train_backbone,
        #                                         drop_path_rate=0,
        #                                         attn_drop_rate=0,
        #                                         drop_rate=0)
        self.num_encoder_layers = num_encoder_layers
        if num_encoder_layers in [1, 2, 3, 4]:
            self.decoder = DecoderSwin(num_encoder_layers)
        else:
            raise ValueError(f'ERROR: unsupported num_encoder_layers={num_encoder_layers}')

        self.seg_channels = 256 # 【修改 4】對應 Seg_encoder 的輸出通道
        self.seg_encoder = Seg_encoder_v4(out_channels=self.seg_channels)
        self.fuse_idx = transformer_params['fuse_idx']

        self.fusion_convs = nn.ModuleList()
        embed_dims = [768, 384, 192, 96]
        self.dropout = nn.Dropout(0.5)

        for i in range(4):
            if i in self.fuse_idx:
                # 拼接後的輸入通道數 = Backbone特徵維度 + Seg特徵維度(256)
                in_channels = embed_dims[i] + self.seg_channels
                out_channels = embed_dims[i]

                # Late Fusion Adapter
                fusion_layer = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True)
                )
                nn.init.xavier_uniform_(fusion_layer[0].weight)
                self.fusion_convs.append(fusion_layer)
            else:
                self.fusion_convs.append(nn.Identity())

    def forward(self, x, seg_input):
        with torch.no_grad():
            b_out = self.backbone_3d(x)

        # Output shape: (B, 256, T_seg, H_seg, W_seg) -> 保留了完整的空間幾何資訊
        seg_out = self.seg_encoder(seg_input)

        for idx, b in enumerate(b_out):
            if idx in self.fuse_idx:
                # b 的形狀為 (B, C, T_b, H, W)
                B, C, T_b, H, W = b.shape

                # --- 【修改 5】動態時空對齊 (Spatio-Temporal Alignment) ---
                # 不再使用暴力擴展 (expand)，而是使用 trilinear 插值
                # 直接將 Seg 特徵縮放對齊到當前 Backbone 層的時間與空間解析度
                aligned_seg = F.interpolate(
                    seg_out, 
                    size=(T_b, H, W), 
                    mode='trilinear', 
                    align_corners=False
                )

                # --- 通道拼接 (Concatenation) ---
                # 現在兩者的 T, H, W 都「真實」匹配了！左上角的車會對準左上角的特徵
                # fused 形狀變為 (B, C + 256, T_b, H, W)
                fused = torch.cat([b, aligned_seg], dim=1)
                
                # 特徵融合與降維
                fused = self.fusion_convs[idx](fused)
                b_out[idx] = self.dropout(fused)
        
        return self.decoder(b_out[:self.num_encoder_layers])
    
class SCAM3d(nn.Module):
    def __init__(self, channels, reduction_ratio=16, spatial_pool_kernel=(1, 4, 4), channel_pool_group_size=8):
        """
        初始化 3D SCAM 模組
        :param channels: 輸入特徵的通道數 (C)
        :param reduction_ratio: 通道壓縮比例 (r)，用於減少參數量
        :param spatial_pool_kernel: 空間局部池化的核大小 (k_t, k_h, k_w)。時間維度通常設為 1 避免過度壓縮。
        :param channel_pool_group_size: 通道局部池化的分組大小 (C_l)
        """
        super(SCAM3d, self).__init__()
        self.channels = channels
        self.cl = channel_pool_group_size
        
        # 確保通道數可以被 channel_pool_group_size 整除，否則無法完美分組 [cite: 89, 111]
        assert channels % self.cl == 0, "Channels must be divisible by channel_pool_group_size"
        
        # ==========================================
        # 1. 空間局部通道注意力 (Spatially Localized Channel Attention)
        # ==========================================
        self.spatial_pool_kernel = spatial_pool_kernel
        
        # 論文指出這裡包含兩個連續的 1x1 卷積層 
        reduced_channels = max(1, channels // reduction_ratio)
        self.conv1 = nn.Conv3d(channels, reduced_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(reduced_channels, channels, kernel_size=1, bias=False)
        self.sigmoid_c = nn.Sigmoid()
        
        # ==========================================
        # 2. 通道局部空間注意力 (Channel Localized Spatial Attention)
        # ==========================================
        # 這部分主要依賴張量操作 (Channel Local Average Pooling)，
        # 所以只需要一個 Sigmoid 函數，不需額外學習參數 [cite: 114]。
        self.sigmoid_s = nn.Sigmoid()

    def forward(self, x):
        B, C, T, H, W = x.size()
        
        # -----------------------------------------------------------
        # Phase 1: 空間局部通道注意力 (Spatially Localized Channel Attention)
        # -----------------------------------------------------------
        # 1. 空間局部平均池化 (Spatial Local Average Pooling) 
        # 形狀變化: (B, C, T, H, W) -> (B, C, T', H', W')
        pool_c = F.avg_pool3d(x, kernel_size=self.spatial_pool_kernel, stride=self.spatial_pool_kernel)
        
        # 2. 經過 MLP (1x1 Conv -> ReLU -> 1x1 Conv) 與 Sigmoid 獲得通道權重 [cite: 104]
        ac = self.conv1(pool_c)
        ac = self.relu(ac)
        ac = self.conv2(ac)
        ac = self.sigmoid_c(ac)
        
        # 3. 將權重上採樣回原尺寸，並與輸入特徵進行 Element-wise 相乘 
        ac_upsampled = F.interpolate(ac, size=(T, H, W), mode='trilinear', align_corners=False)
        f_prime = x * ac_upsampled
        
        # -----------------------------------------------------------
        # Phase 2: 通道局部空間注意力 (Channel Localized Spatial Attention)
        # -----------------------------------------------------------
        # 1. 通道局部平均池化 (Channel Local Average Pooling) 
        cg = C // self.cl  # 計算會有幾組薄片 (C_g = C / C_l) [cite: 89, 111]
        
        # 將通道維度拆解為兩維度 (B, C_g, C_l, T, H, W)
        ch_pool = f_prime.view(B, cg, self.cl, T, H, W)
        # 沿著 C_l (局部通道組) 維度取平均，壓縮成 (B, C_g, T, H, W)
        ch_pool = ch_pool.mean(dim=2)
        
        # 2. 經過 Sigmoid 獲得空間注意力地圖 [cite: 111, 114]
        as_weight = self.sigmoid_s(ch_pool) 
        
        # 3. 將這些空間權重重複/擴展，對應回原本的各個通道 [cite: 94]
        # 先增加一個維度 (B, C_g, 1, T, H, W)，擴展為 (B, C_g, C_l, T, H, W)，最後 reshape 回 (B, C, T, H, W)
        as_expanded = as_weight.unsqueeze(2).expand(B, cg, self.cl, T, H, W).reshape(B, C, T, H, W)
        attention_out = f_prime * as_expanded

        # 4. 第二次 Element-wise 相乘，輸出最終精煉特徵 [cite: 91]
        # out = f_prime * as_expanded
        # 4. 使用殘差連結，而非直接相乘
        out = x + attention_out
        return out

class Seg_encoder_v5(nn.Module):
    def __init__(self, out_channels=256):
        super(Seg_encoder_v5, self).__init__()
        self.conv1a = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm3d(64)
        self.conv1b = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d((1, 2, 2))

        self.conv2a = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm3d(128)
        self.conv2b = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d((1, 2, 2))

        # 【修改 1】移除 Linear，改用 1x1x1 Conv3d 來保留空間維度 (Fully Convolutional)
        self.adapter = nn.Sequential(
            nn.Conv3d(128, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Input: (Batch, 3, Time, Height, Width)
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x) 
        
        # 【修改 2】徹底移除 x.mean(dim=(3, 4)) GAP 操作
        # 目前 x 的形狀是 (B, 128, T, H/4, W/4)

        # 【修改 3】直接通過 3D 卷積進行通道轉換
        x = self.adapter(x)       
        
        # Output shape: (B, 256, T, H/4, W/4)
        return x

class SCOUT_seg_v5(nn.Module):
    def __init__(self, 
                  num_encoder_layers=4,
                  train_backbone=False,
                  pretrained_backbone=False,
                  transformer_params=None,
                  **kwargs
                ):

        super(SCOUT_seg_v5, self).__init__()
        self.backbone_3d = SwinTransformer3DBackbone(pretrained=pretrained_backbone, train_backbone=train_backbone)
        self.num_encoder_layers = num_encoder_layers
        if num_encoder_layers in [1, 2, 3, 4]:
            self.decoder = DecoderSwin(num_encoder_layers)
        else:
            raise ValueError(f'ERROR: unsupported num_encoder_layers={num_encoder_layers}')

        self.seg_channels = 256 # 【修改 4】對應 Seg_encoder 的輸出通道
        self.seg_encoder = Seg_encoder_v4(out_channels=self.seg_channels)
        self.fuse_idx = transformer_params['fuse_idx']

        self.fusion_convs = nn.ModuleList()
        self.scam_modules = nn.ModuleList()
        embed_dims = [768, 384, 192, 96]
        self.dropout = nn.Dropout(0.5)

        for i in range(4):
            if i in self.fuse_idx:
                # 拼接後的輸入通道數 = Backbone特徵維度 + Seg特徵維度(256)
                in_channels = embed_dims[i] + self.seg_channels
                out_channels = embed_dims[i]

                # Late Fusion Adapter
                fusion_layer = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True)
                )
                nn.init.xavier_uniform_(fusion_layer[0].weight)
                self.fusion_convs.append(fusion_layer)
                self.scam_modules.append(SCAM3d(channels=out_channels))
            else:
                self.fusion_convs.append(nn.Identity())
                self.scam_modules.append(nn.Identity())

    def forward(self, x, seg_input):
        with torch.no_grad():
            b_out = self.backbone_3d(x)

        # Output shape: (B, 256, T_seg, H_seg, W_seg) -> 保留了完整的空間幾何資訊
        seg_out = self.seg_encoder(seg_input)

        for idx, b in enumerate(b_out):
            if idx in self.fuse_idx:
                # b 的形狀為 (B, C, T_b, H, W)
                B, C, T_b, H, W = b.shape
                aligned_seg = F.interpolate(
                    seg_out, 
                    size=(T_b, H, W), 
                    mode='trilinear', 
                    align_corners=False
                )

                fused = torch.cat([b, aligned_seg], dim=1)
                
                # 特徵融合與降維
                fused = self.fusion_convs[idx](fused)
                fused = self.scam_modules[idx](fused)
                b_out[idx] = self.dropout(fused)
        
        return self.decoder(b_out[:self.num_encoder_layers])

class SCOUT_seg_v6(nn.Module):
    def __init__(self, 
                  num_encoder_layers=4,
                  train_backbone=False,
                  pretrained_backbone=False,
                  transformer_params=None,
                  **kwargs
                ):

        super(SCOUT_seg_v6, self).__init__()
        self.backbone_3d = SwinTransformer3DBackbone(pretrained=pretrained_backbone, train_backbone=train_backbone)
        self.num_encoder_layers = num_encoder_layers
        if num_encoder_layers in [1, 2, 3, 4]:
            self.decoder = DecoderSwin(num_encoder_layers)
        else:
            raise ValueError(f'ERROR: unsupported num_encoder_layers={num_encoder_layers}')

        self.seg_channels = 256 # 【修改 4】對應 Seg_encoder 的輸出通道
        self.seg_encoder = Seg_encoder_v5(out_channels=self.seg_channels)
        self.fuse_idx = transformer_params['fuse_idx']

        self.fusion_convs = nn.ModuleList()
        embed_dims = [768, 384, 192, 96]
        self.dropout = nn.Dropout(0.5)

        for i in range(4):
            if i in self.fuse_idx:
                # 拼接後的輸入通道數 = Backbone特徵維度 + Seg特徵維度(256)
                in_channels = embed_dims[i] + self.seg_channels
                out_channels = embed_dims[i]

                # Late Fusion Adapter
                fusion_layer = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True)
                )
                nn.init.xavier_uniform_(fusion_layer[0].weight)
                self.fusion_convs.append(fusion_layer)
            else:
                self.fusion_convs.append(nn.Identity())

    def forward(self, x, seg_input):
        with torch.no_grad():
            b_out = self.backbone_3d(x)

        # Output shape: (B, 256, T_seg, H_seg, W_seg) -> 保留了完整的空間幾何資訊
        seg_out = self.seg_encoder(seg_input)
        for idx, b in enumerate(b_out):
            if idx in self.fuse_idx:
                # b 的形狀為 (B, C, T_b, H, W)
                B, C, T_b, H, W = b.shape
                aligned_seg = F.interpolate(
                    seg_out, 
                    size=(T_b, H, W), 
                    mode='trilinear', 
                    align_corners=False
                )
                fused = torch.cat([b, aligned_seg], dim=1)
                fused = self.fusion_convs[idx](fused)
                b_out[idx] = self.dropout(fused)
        
        return self.decoder(b_out[:self.num_encoder_layers])

class SCATM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, spatial_pool_kernel=(1, 4, 4), channel_pool_group_size=8, temporal_kernel_size=3):
        """
        初始化包含時間注意力的 3D SCAM 模組
        :param temporal_kernel_size: 時間卷積核的大小，通常設為 3，用於捕捉相鄰影格的動態變化
        """
        super(SCATM, self).__init__()
        self.channels = channels
        self.cl = channel_pool_group_size
        assert channels % self.cl == 0, "Channels must be divisible by channel_pool_group_size"
        self.spatial_pool_kernel = spatial_pool_kernel
        reduced_channels = max(1, channels // reduction_ratio)
        
        # --- Phase 1: Channel Attention 參數 ---
        self.conv1 = nn.Conv3d(channels, reduced_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(reduced_channels, channels, kernel_size=1, bias=False)
        self.sigmoid_c = nn.Sigmoid()
        
        # --- Phase 2: Spatial Attention 參數 ---
        self.sigmoid_s = nn.Sigmoid()

        # --- Phase 3: Temporal Attention 參數 (NEW) ---
        # kernel_size=(3, 1, 1) 代表只在時間軸 (T) 上做長度為 3 的卷積
        # padding=(1, 0, 0) 確保卷積後的時間長度 T 維持不變
        self.conv_t = nn.Conv3d(channels, channels, kernel_size=(temporal_kernel_size, 1, 1), 
                                padding=(temporal_kernel_size // 2, 0, 0))
        self.sigmoid_t = nn.Sigmoid()

    def forward(self, x):
        B, C, T, H, W = x.size()
        
        # ==========================================
        # Phase 1: 空間局部通道注意力 (Channel Attention)
        # ==========================================
        pool_c = F.avg_pool3d(x, kernel_size=self.spatial_pool_kernel, stride=self.spatial_pool_kernel)
        ac = self.conv1(pool_c)
        ac = self.relu(ac)
        ac = self.conv2(ac)
        ac = self.sigmoid_c(ac)
        ac_upsampled = F.interpolate(ac, size=(T, H, W), mode='trilinear', align_corners=False)
        f_prime = x * ac_upsampled
        
        # ==========================================
        # Phase 2: 通道局部空間注意力 (Spatial Attention)
        # ==========================================
        cg = C // self.cl  
        ch_pool = f_prime.view(B, cg, self.cl, T, H, W)
        ch_pool = ch_pool.mean(dim=2)
        as_weight = self.sigmoid_s(ch_pool) 
        as_expanded = as_weight.unsqueeze(2).expand(B, cg, self.cl, T, H, W).reshape(B, C, T, H, W)
        f_prime_prime = f_prime * as_expanded

        # ==========================================
        # Phase 3: 時間注意力 (Temporal Attention) [NEW]
        # ==========================================
        # 1. 空間全局池化 (Spatial Global Average Pooling)
        # 把 (B, C, T, H, W) 沿著 H 和 W 壓扁，變成 (B, C, T, 1, 1)
        pool_t = f_prime_prime.mean(dim=(3, 4), keepdim=True)
        
        # 2. 透過時間卷積捕捉相鄰幀的關聯，並用 Sigmoid 轉換為權重
        at = self.conv_t(pool_t)
        at_weight = self.sigmoid_t(at)
        
        # 3. 與特徵 Element-wise 相乘 (PyTorch 會自動將 1x1 廣播到 HxW)
        attention_out = f_prime_prime * at_weight
        
        # ==========================================
        # 殘差連接 (Residual Connection)
        # ==========================================
        out = x + attention_out
        return out

class SCOUT_seg_v7(nn.Module):
    def __init__(self, 
                  num_encoder_layers=4,
                  train_backbone=False,
                  pretrained_backbone=False,
                  transformer_params=None,
                  **kwargs
                ):

        super(SCOUT_seg_v7, self).__init__()
        self.backbone_3d = SwinTransformer3DBackbone(pretrained=pretrained_backbone, train_backbone=train_backbone)
        self.num_encoder_layers = num_encoder_layers
        if num_encoder_layers in [1, 2, 3, 4]:
            self.decoder = DecoderSwin(num_encoder_layers)
        else:
            raise ValueError(f'ERROR: unsupported num_encoder_layers={num_encoder_layers}')

        self.seg_channels = 256 # 【修改 4】對應 Seg_encoder 的輸出通道
        self.seg_encoder = Seg_encoder_v4(out_channels=self.seg_channels)
        self.fuse_idx = transformer_params['fuse_idx']

        self.fusion_convs = nn.ModuleList()
        self.scam_modules = nn.ModuleList()
        embed_dims = [768, 384, 192, 96]
        self.dropout = nn.Dropout(0.5)

        for i in range(4):
            if i in self.fuse_idx:
                # 拼接後的輸入通道數 = Backbone特徵維度 + Seg特徵維度(256)
                in_channels = embed_dims[i] + self.seg_channels
                out_channels = embed_dims[i]

                # Late Fusion Adapter
                fusion_layer = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True)
                )
                nn.init.xavier_uniform_(fusion_layer[0].weight)
                self.fusion_convs.append(fusion_layer)
                self.scam_modules.append(SCATM(channels=out_channels))
            else:
                self.fusion_convs.append(nn.Identity())
                self.scam_modules.append(nn.Identity())

    def forward(self, x, seg_input):
        with torch.no_grad():
            b_out = self.backbone_3d(x)

        # Output shape: (B, 256, T_seg, H_seg, W_seg) -> 保留了完整的空間幾何資訊
        seg_out = self.seg_encoder(seg_input)

        for idx, b in enumerate(b_out):
            if idx in self.fuse_idx:
                # b 的形狀為 (B, C, T_b, H, W)
                B, C, T_b, H, W = b.shape
                aligned_seg = F.interpolate(
                    seg_out, 
                    size=(T_b, H, W), 
                    mode='trilinear', 
                    align_corners=False
                )

                fused = torch.cat([b, aligned_seg], dim=1)
                
                # 特徵融合與降維
                fused = self.fusion_convs[idx](fused)
                fused = self.scam_modules[idx](fused)
                b_out[idx] = self.dropout(fused)
        
        return self.decoder(b_out[:self.num_encoder_layers])

class SCOUT_seg_v8(nn.Module):
    def __init__(self, 
                 num_encoder_layers=4,
                 train_backbone=False,
                 pretrained_backbone=False,
                 transformer_params=None,
                 **kwargs
                ):

        super(SCOUT_seg_v8, self).__init__()
        self.backbone_3d = SwinTransformer3DBackbone(pretrained=pretrained_backbone, train_backbone=train_backbone)
        self.num_encoder_layers = num_encoder_layers
        if num_encoder_layers in [1, 2, 3, 4]:
            self.decoder = DecoderSwin(num_encoder_layers)
        else:
            raise ValueError(f'ERROR: unsupported num_encoder_layers={num_encoder_layers}')

        self.seg_channels = 256
        self.seg_encoder = Seg_encoder_v4(out_channels=self.seg_channels)
        self.fuse_idx = transformer_params['fuse_idx']

        self.fusion_convs = nn.ModuleList()
        self.scam_modules = nn.ModuleList()
        embed_dims = [768, 384, 192, 96]
        self.dropout = nn.Dropout(0.5)

        for i in range(4):
            if i in self.fuse_idx:
                # 拼接後的輸入通道數 = Backbone特徵維度 + Seg特徵維度(256)
                in_channels = embed_dims[i] + self.seg_channels
                out_channels = embed_dims[i]

                # Late Fusion Adapter
                fusion_layer = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True)
                )
                nn.init.xavier_uniform_(fusion_layer[0].weight)
                self.fusion_convs.append(fusion_layer)
                
                # SCAM3d 現在負責處理 Backbone 特徵，通道數對齊 out_channels (即 embed_dims[i])
                self.scam_modules.append(SCAM3d(channels=out_channels))
            else:
                self.fusion_convs.append(nn.Identity())
                self.scam_modules.append(nn.Identity())

    def forward(self, x, seg_input):
        with torch.no_grad():
            b_out = self.backbone_3d(x)

        # Output shape: (B, 256, T_seg, H_seg, W_seg) -> 保留了完整的空間幾何資訊
        seg_out = self.seg_encoder(seg_input)

        for idx, b in enumerate(b_out):
            if idx in self.fuse_idx:
                # b 的形狀為 (B, C, T_b, H, W)
                B, C, T_b, H, W = b.shape
                aligned_seg = F.interpolate(
                    seg_out, 
                    size=(T_b, H, W), 
                    mode='trilinear', 
                    align_corners=False
                )

                # ==========================================================
                # 方案 B：PEFT 並聯式旁支融合 (Side-Tuning Adapter)
                # ==========================================================
                
                # 步驟 1: 讓 SCAM 直接作用在預訓練的 RGB 特徵上 (滿足教授的要求)
                # 因為 SCAM3d 內部有寫 out = x + attention_out，所以不會破壞特徵
                b_scam = self.scam_modules[idx](b)

                # 步驟 2: 將強化後的 RGB 特徵與 Seg 進行拼接與降維融合
                fused = torch.cat([b_scam, aligned_seg], dim=1)
                fused = self.fusion_convs[idx](fused)
                fused = self.dropout(fused)
                
                # 步驟 3: 終極防護網 —— 殘差並聯 (Residual Add)
                # 將「融合後的混合物」視為一個 Adapter 的視覺提示 (Visual Prompt)
                # 並將它直接「加回」最原始、最純淨的 Backbone 特徵 b
                b_out[idx] = b + fused
                # ==========================================================
        
        return self.decoder(b_out[:self.num_encoder_layers])

if __name__ == "__main__":
    batch_size = 4
    time_steps = 16
    height = 128
    width = 128

    # (Batch size, channel, time step, height, width)
    x_dummy = torch.randn(batch_size, 3, time_steps, height, width)
    y_dummy = torch.randn(batch_size, 3, time_steps, height, width)

    model = SCOUT_seg_v2()
    if torch.cuda.is_available():
        model = model.cuda()
        x_dummy = x_dummy.cuda()
        y_dummy = y_dummy.cuda()
    
    out = model(x_dummy, y_dummy)
    print(f"Input 1 shape: {x_dummy.shape}")
    print(f"Input 2 shape: {y_dummy.shape}")
    print(f"Output shape: {out.shape}")