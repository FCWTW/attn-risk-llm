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
from network.filters import cfg
from network.seg_encoder import *

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
    
# SGCN Seg encoder + Cross-Attention
class DAM_v1(nn.Module):
    def __init__(self, 
                  num_encoder_layers=4,
                  train_backbone=False,
                  pretrained_backbone=False,
                  transformer_params=None,
                  **kwargs
                ):

        super(DAM_v1, self).__init__()
        self.attn_logger = logging.getLogger('Attention Weight')
        self.attn_logger.setLevel(logging.INFO)
        if not self.attn_logger.handlers:
            filename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            fh = logging.FileHandler(f'cache/{filename}', mode='w')
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            fh.setFormatter(formatter)
            self.attn_logger.addHandler(fh)
        self.log_step_counter = 0
        self.log_frequency = 100

        self.backbone_3d = SwinTransformer3DBackbone(pretrained=pretrained_backbone,
                                                train_backbone=train_backbone)

        self.num_encoder_layers = num_encoder_layers

        if num_encoder_layers in [1, 2, 3, 4]:
            self.decoder = DecoderSwin(num_encoder_layers)
        else:
            raise ValueError(f'ERROR: unsupported num_encoder_layers={num_encoder_layers}, \
                            should be between 1 and 4.')

        self.seg_encoder = Seg_encoder_v1()
        self.add_and_norm = transformer_params['add_and_norm']
        self.fuse_idx = transformer_params['fuse_idx']
        self.num_att_heads = transformer_params['num_att_heads']

        self.multihead_attn = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.seg_projectors = nn.ModuleList()
        embed_dims = [768, 384, 192, 96]

        for i in range(4):
            if i in self.fuse_idx:
                self.multihead_attn.append(
                    nn.MultiheadAttention(embed_dim=embed_dims[i],
                                          num_heads=self.num_att_heads[i], 
                                          bias=True)
                )
                self.norm.append(nn.LayerNorm(embed_dims[i]))
                self.seg_projectors.append(nn.Linear(512, embed_dims[i]))
            else:
                self.multihead_attn.append(nn.Identity())
                self.norm.append(nn.Identity())
                self.seg_projectors.append(nn.Identity())
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, seg_input):
        # b_out: (B, C, T, H, W)
        b_out = self.backbone_3d(x)
        b_s = [b.shape for b in b_out]

        # seg_out: (B, 512, T, 1, 1)
        seg_out = self.seg_encoder(seg_input)
        if self.training:
            self.log_step_counter += 1

        for idx, b in enumerate(b_out):
            if idx in self.fuse_idx:    
                # Query
                b_flat = b.flatten(2).permute((2, 0, 1))

                # Key and Value
                curr_seg = seg_out.squeeze(-1).squeeze(-1)      # (B, 512, T)
                curr_seg = curr_seg.permute(0, 2, 1)            # (B, T, 512) for Linear
                curr_seg = self.seg_projectors[idx](curr_seg)   # (B, T, C)
                seg_flat = curr_seg.permute(1, 0, 2)            # (T, B, C)

                fused_out, attn_weights = self.multihead_attn[idx](
                    b_flat, seg_flat, seg_flat, need_weights=True
                )
                
                # --- Monitoring code for attention weight ---
                if self.training and (self.log_step_counter % self.log_frequency == 0 or self.log_step_counter == 1):
                    with torch.no_grad():
                        max_w = attn_weights.max().item()
                        mean_w = attn_weights.mean().item()
                        std_w = attn_weights.std(dim=-1).mean().item()
                        log_msg = f"[Step {self.log_step_counter}][Layer {idx}] Attn Stats -> Max: {max_w:.4f}, Mean: {mean_w:.4f}, Std: {std_w:.6f}"
                        if torch.isnan(attn_weights).any():
                            self.attn_logger.warning(f"[Step {self.log_step_counter}][Layer {idx}] ALERT: Attention weights contains NaN!")
                        else:
                            self.attn_logger.info(log_msg)
                
                fused_out = self.dropout(fused_out)
                if self.add_and_norm:
                    fused_out += b_flat
                    fused_out = self.norm[idx](fused_out)

                fused_out = fused_out.permute((1, 2, 0))
                b_out[idx] = fused_out.view(*b_s[idx])
        return self.decoder(b_out[:self.num_encoder_layers])
    
# FCN Seg encoder + Concat
class DAM_v2(nn.Module):
    def __init__(self, 
                  num_encoder_layers=4,
                  train_backbone=False,
                  pretrained_backbone=False,
                  transformer_params=None,
                  **kwargs
                ):

        super(DAM_v2, self).__init__()
        self.backbone_3d = SwinTransformer3DBackbone(pretrained=pretrained_backbone, train_backbone=train_backbone)
        self.num_encoder_layers = num_encoder_layers
        if num_encoder_layers in [1, 2, 3, 4]:
            self.decoder = DecoderSwin(num_encoder_layers)
        else:
            raise ValueError(f'ERROR: unsupported num_encoder_layers={num_encoder_layers}')

        self.seg_channels = 256
        self.seg_encoder = Seg_encoder_v3(out_channels=self.seg_channels)
        self.fuse_idx = transformer_params['fuse_idx']

        self.fusion_convs = nn.ModuleList()
        embed_dims = [768, 384, 192, 96]
        self.dropout = nn.Dropout(0.5)

        for i in range(4):
            if i in self.fuse_idx:
                # Number of channels after concatenation = Backbone feature dimension + Seg feature dimension (256)
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
        b_out = self.backbone_3d(x)

        # Output shape: (B, 256, T_seg, H_seg, W_seg)
        seg_out = self.seg_encoder(seg_input)

        for idx, b in enumerate(b_out):
            if idx in self.fuse_idx:
                # b: (B, C, T_b, H, W)
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
    
def conv3d_downsample(in_filters, out_filters, normalization=False):
    # 3D downsampling: Time axis (T) step size of 1 (uncompressed), spatial axes (H, W) step size of 2
    layers = [nn.Conv3d(in_filters, out_filters, kernel_size=3, stride=(1, 2, 2), padding=1)]
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    if normalization:
        layers.append(nn.InstanceNorm3d(out_filters, affine=True))
    return layers

class CNN_PP_3D(nn.Module):
    """
    Modify from https://github.com/wenyyu/IA-Seg/blob/main/network/dip.py
    """
    def __init__(self, in_channels=3):
        super(CNN_PP_3D, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm3d(16, affine=True),
            *conv3d_downsample(16, 32, normalization=True),
            *conv3d_downsample(32, 64, normalization=True),
            *conv3d_downsample(64, 128, normalization=True),
            *conv3d_downsample(128, 128),
            nn.Dropout(p=0.5)
        )
        # Use global pooling to compress space and time into a single point
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Conv3d(128, cfg.num_filter_parameters, kernel_size=1)

    def forward(self, img_input):
        # Input shape: (B, 3, T, H, W)
        B, C, T, H, W = img_input.shape
        x_resized = F.interpolate(img_input, size=(T, 256, 256), mode='trilinear', align_corners=False)
        features = self.feature_extractor(x_resized) # (B, 128, T', 8, 8)
        features = self.global_pool(features)        # (B, 128, 1, 1, 1)

        Pr_3d = self.fc(features) # (B, 4, 1, 1, 1)
        Pr_2d = Pr_3d.squeeze(2)  # (B, 4, 1, 1)

        # (B, 3, T, H, W) -> (B, T, 3, H, W) -> (B*T, 3, H, W)
        img_folded = img_input.transpose(1, 2).contiguous().view(B*T, C, H, W)

        # (B, 4, 1, 1) -> (B, T, 4, 1, 1) -> (B*T, 4, 1, 1)
        Pr = Pr_2d.unsqueeze(1).expand(B, T, cfg.num_filter_parameters, 1, 1).contiguous().view(B*T, cfg.num_filter_parameters, 1, 1)
        self.filtered_image_batch = img_folded

        filters_op = [x(self, cfg) for x in cfg.filters]
        filter_parameters = []
        filtered_images = []

        for j, filter_op in enumerate(filters_op):
            self.filtered_image_batch, filter_parameter = filter_op.apply(self.filtered_image_batch, Pr)
            filter_parameters.append(filter_parameter)
            filtered_images.append(self.filtered_image_batch)
            
        # (B*T, 3, H, W) -> (B, T, 3, H, W) -> (B, 3, T, H, W)
        final_output_3d = self.filtered_image_batch.view(B, T, C, H, W).transpose(1, 2).contiguous()
        return final_output_3d, filtered_images, Pr, filter_parameters

class GatedFusionAdapter(nn.Module):
    def __init__(self, rgb_channels, seg_channels, out_channels):
        super(GatedFusionAdapter, self).__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv3d(rgb_channels + seg_channels, seg_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(rgb_channels + seg_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb_feat, seg_feat):
        concat_feat = torch.cat([rgb_feat, seg_feat], dim=1)
        gate = self.gate_conv(concat_feat)
        gated_seg_feat = seg_feat * gate
        final_concat = torch.cat([rgb_feat, gated_seg_feat], dim=1)
        out = self.fusion_conv(final_concat)
        return out

# FCN Seg encoder + Concat + Gate Fusion + Video IAPM (CNN_PP_3D & filters)
class DAM_v3(nn.Module):
    def __init__(self, 
                  num_encoder_layers=4,
                  train_backbone=False,
                  pretrained_backbone=False,
                  transformer_params=None,
                  **kwargs
                ):

        super(DAM_v3, self).__init__()
        self.iapm = CNN_PP_3D(in_channels=3)
        self.backbone_3d = SwinTransformer3DBackbone(pretrained=pretrained_backbone, train_backbone=train_backbone)
        self.num_encoder_layers = num_encoder_layers
        if num_encoder_layers in [1, 2, 3, 4]:
            self.decoder = DecoderSwin(num_encoder_layers)
        else:
            raise ValueError(f'ERROR: unsupported num_encoder_layers={num_encoder_layers}')

        self.seg_channels = 256
        self.seg_encoder = Seg_encoder_v3(out_channels=self.seg_channels)
        self.fuse_idx = transformer_params['fuse_idx']

        self.fusion_convs = nn.ModuleList()
        embed_dims = [768, 384, 192, 96]
        self.dropout = nn.Dropout(0.5)

        for i in range(4):
            if i in self.fuse_idx:
                rgb_dim = embed_dims[i]
                fusion_layer = GatedFusionAdapter(
                    rgb_channels=rgb_dim, 
                    seg_channels=self.seg_channels, 
                    out_channels=rgb_dim
                )
                self.fusion_convs.append(fusion_layer)
            else:
                self.fusion_convs.append(nn.Identity())

    def forward(self, x, seg_input):
        enhanced_x, _, _, _ = self.iapm(x)
        b_out = self.backbone_3d(enhanced_x)

        # Output shape: (B, 256, T_seg, H_seg, W_seg)
        seg_out = self.seg_encoder(seg_input)
        for idx, b in enumerate(b_out):
            if idx in self.fuse_idx:
                B, C, T_b, H, W = b.shape
                aligned_seg = F.interpolate(
                    seg_out, 
                    size=(T_b, H, W), 
                    mode='trilinear', 
                    align_corners=False
                )
                fused = self.fusion_convs[idx](b, aligned_seg)
                b_out[idx] = self.dropout(fused)
        return self.decoder(b_out[:self.num_encoder_layers])

if __name__ == "__main__":
    batch_size = 4
    time_steps = 16
    height = 128
    width = 128

    # (Batch size, channel, time step, height, width)
    x_dummy = torch.randn(batch_size, 3, time_steps, height, width)
    y_dummy = torch.randn(batch_size, 3, time_steps, height, width)

    model = DAM_2()
    if torch.cuda.is_available():
        model = model.cuda()
        x_dummy = x_dummy.cuda()
        y_dummy = y_dummy.cuda()
    
    out = model(x_dummy, y_dummy)
    print(f"Input 1 shape: {x_dummy.shape}")
    print(f"Input 2 shape: {y_dummy.shape}")
    print(f"Output shape: {out.shape}")