import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.swin_transformer import swin_tiny_patch4_window7_224 as swin_tiny
from models.MCBO import MCBO_tiny
from models.MCBO import MCM_tity
from models.transformer import *


class MCBO(nn.Module):
   
    def __init__(self):
        super().__init__()
        self.model = MCBO_tiny()
        self.model.load_state_dict(torch.load('models/pretrained_weights/swin_tiny_patch4_window7_224.pth')['model'], strict=False)
        self.MLP_layer = nn.Sequential(
                        nn.Linear(768, 256),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, 2))

    def forward(self, left_tensor, right_tensor, **kwargs):
        feature = self.model(left_tensor, right_tensor, **kwargs)
        cls_output = self.MLP_layer(feature)
        return cls_output


class MCM(nn.Module):
   
    def __init__(self):
        super().__init__()
        self.model = MCM_tity()
        self.model.load_state_dict(torch.load('models/pretrained_weights/swin_tiny_patch4_window7_224.pth')['model'], strict=False)
        self.MLP_layer = nn.Sequential(
                        nn.Linear(768, 256),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, 2))

    def forward(self, left_tensor, right_tensor, **kwargs):
        feature = self.model(left_tensor, right_tensor, **kwargs)
        cls_output = self.MLP_layer(feature)
        return cls_output


class BFM(nn.Module):

    def __init__(self, attention_layer=1, attention_num_heads=1, dim_feedforward=128, return_attn_map=False):
        super().__init__()
        empty_layer = nn.Sequential()
        self.return_attn_map = return_attn_map
        self.model = swin_tiny()
        self.model.load_state_dict(torch.load('models/pretrained_weights/swin_tiny_patch4_window7_224.pth')['model'], strict=False)
        self.model.head = empty_layer
        self.transformer_encoder = SimpleTransformer(in_dim=768, linear_dim=dim_feedforward, num_head=attention_num_heads, num_attn=attention_layer, merge_token=True, dropout=0.0)

        self.MLP_layer = nn.Sequential(
                        nn.Linear(768, 256),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, 2))

    def forward(self, left_tensor, right_tensor, **kwargs):
        left_feature = self.model(left_tensor)
        right_feature = self.model(right_tensor)
        feature = torch.cat((left_feature, right_feature), dim=1)
        feature, attn_map = self.transformer_encoder(feature)
        cls_output = self.MLP_layer(feature)
        if self.return_attn_map:
            return cls_output, attn_map[:, 0, 0]
        return cls_output
    
