from functools import partial
from tkinter import X
from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .simple_resnet import ResNet, Bottleneck

'''
H = img height
W = img width
T = video time
'''


class ResNet2D(nn.Module):
    def __init__(self, vid_dim=(128,128,100,3), temporal_dialation=2, temporal_kernal_size=16, norm_layer=None, num_classes=26):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            in_chans (int): number of input channels, RGB videos have 3 chanels
            spatial_embed_dim (int): spatial patch embedding dimension
            sdepth (int): depth of spatial transformer
            tdepth(int):depth of temporal transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            spat_op(string): Spatial Transformer output type - pool(Global avg pooling of encded features) or cls(Just CLS token)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            tubelet_dim(tuple): tubelet size (ch,tt,th,tw)
            vid_dim: Original video (H , W, T)
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.vid_dim = vid_dim
        self.num_classes = num_classes

        H,W,T,c = vid_dim
        self.t_size = T - temporal_dialation * (temporal_kernal_size - 1)

        self.temporal_conv = torch.nn.Conv1d(c, c, kernel_size=temporal_kernal_size, dilation=temporal_dialation, groups=c)

        self.res_net = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

        #Classification head
        self.class_head = nn.Linear(num_classes*self.t_size, self.num_classes)


    def forward(self, x):
        #Input x: batch x img_height x img_width x num_frames x num_channels
        b, T, H, W, c = x.shape

        x = rearrange(x, 'b T H W c  -> (b H W) c T')
        x = self.temporal_conv(x)

        x = rearrange(x, '(b H W) c t -> (b t) c H W', b=b, H=H, W=W, t=self.t_size)
        x = self.res_net(x)

        x = rearrange(x, '(b t) cls -> b (t cls)', b=b, t=self.t_size)
        x = self.class_head(x)
        return x #F.log_softmax(x,dim=1)

'''
model=ViViT_FE()
inp=torch.randn((1, 250, 3, 128 , 128 ,4))
op=model(inp)
print("Op shape: ",op.shape)
'''
