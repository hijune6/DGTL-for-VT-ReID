import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
    PART of the code is from the following link
    https://github.com/Diego999/pyGAT/blob/master/layers.py
"""


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class AVG(nn.Module):
    def __init__(self, channel, fuse = 'sum', reduction=16):
        super(AVG, self).__init__()
        self.fuse = fuse
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.bottleneck = nn.BatchNorm1d(channel) if fuse == 'sum' else nn.BatchNorm1d(channel*2)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        nn.init.normal_(self.bottleneck.weight.data, 1.0, 0.01)
        nn.init.zeros_(self.bottleneck.bias.data)

    def forward(self, x, feat):
        b, c, _, _ = x.size()
        y_pool = self.avg_pool(x).view(b, c)

        if self.fuse == 'sum': 
            feats = y_pool + feat
        elif self.fuse == 'cat':
            feats = torch.cat([y_pool,feat],dim=1)
        feats_bn = self.bottleneck(feats)
        return feats, feats_bn

class MAX(nn.Module):
    def __init__(self, channel, fuse = 'sum', reduction=16):
        super(MAX, self).__init__()
        self.fuse = fuse
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.bottleneck = nn.BatchNorm1d(channel) if fuse == 'sum' else nn.BatchNorm1d(channel*2)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        nn.init.normal_(self.bottleneck.weight.data, 1.0, 0.01)
        nn.init.zeros_(self.bottleneck.bias.data)

    def forward(self, x, feat):
        b, c, _, _ = x.size()
        y_pool = self.max_pool(x).view(b, c)

        if self.fuse == 'sum': 
            feats = y_pool + feat
        elif self.fuse == 'cat':
            feats = torch.cat([y_pool,feat],dim=1)
        feats_bn = self.bottleneck(feats)
        return feats, feats_bn

class GEM(nn.Module):
    def __init__(self, channel, fuse = 'sum', reduction=16):
        super(GEM, self).__init__()
        self.fuse = fuse
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.bottleneck = nn.BatchNorm1d(channel) if fuse == 'sum' else nn.BatchNorm1d(channel*2)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        nn.init.normal_(self.bottleneck.weight.data, 1.0, 0.01)
        nn.init.zeros_(self.bottleneck.bias.data)

    def forward(self, x, feat):
        b, c, _, _ = x.size()
        x_pool = x.view(b, c, -1)
        p = 3.0    
        y_pool = (torch.mean(x_pool**p, dim=-1) + 1e-12)**(1/p)

        if self.fuse == 'sum': 
            feats = y_pool + feat
        elif self.fuse == 'cat':
            feats = torch.cat([y_pool,feat],dim=1)
        feats_bn = self.bottleneck(feats)
        return feats, feats_bn

class IWPA(nn.Module):
    """
    Part attention layer, "Dynamic Dual-Attentive Aggregation Learning for Visible-Infrared Person Re-Identification"
    """
    def __init__(self, in_channels, part = 3, fuse = 'sum', inter_channels=None, out_channels=None):
        super(IWPA, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.l2norm = Normalize(2)
        self.fuse = fuse

        if self.inter_channels is None:
            self.inter_channels = in_channels

        if self.out_channels is None:
            self.out_channels = in_channels

        conv_nd = nn.Conv2d

        self.fc1 = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.fc2 = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.fc3 = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.out_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.out_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.bottleneck = nn.BatchNorm1d(in_channels) if fuse == 'sum' else nn.BatchNorm1d(in_channels*2)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        nn.init.normal_(self.bottleneck.weight.data, 1.0, 0.01)
        nn.init.zeros_(self.bottleneck.bias.data)

        # weighting vector of the part features
        self.gate = nn.Parameter(torch.FloatTensor(part))
        nn.init.constant_(self.gate, 1/part)
    def forward(self, x, feat, t=None, part=0):
        bt, c, h, w = x.shape
        b = bt // t

        # get part features
        part_feat = F.adaptive_avg_pool2d(x, (part, 1))
        part_feat = part_feat.view(b, t, c, part)
        part_feat = part_feat.permute(0, 2, 1, 3) # B, C, T, Part

        part_feat1 = self.fc1(part_feat).view(b, self.inter_channels, -1)  # B, C//r, T*Part
        part_feat1 = part_feat1.permute(0, 2, 1)  # B, T*Part, C//r

        part_feat2 = self.fc2(part_feat).view(b, self.inter_channels, -1)  # B, C//r, T*Part

        part_feat3 = self.fc3(part_feat).view(b, self.inter_channels, -1)  # B, C//r, T*Part
        part_feat3 = part_feat3.permute(0, 2, 1)   # B, T*Part, C//r

        # get cross-part attention
        cpa_att = torch.matmul(part_feat1, part_feat2) # B, T*Part, T*Part
        cpa_att = F.softmax(cpa_att, dim=-1)

        # collect contextual information
        refined_part_feat = torch.matmul(cpa_att, part_feat3) # B, T*Part, C//r
        refined_part_feat = refined_part_feat.permute(0, 2, 1).contiguous() # B, C//r, T*Part
        refined_part_feat = refined_part_feat.view(b, self.inter_channels, part) # B, C//r, T, Part

        gate = F.softmax(self.gate, dim=-1)
        weight_part_feat = torch.matmul(refined_part_feat, gate)
        #x = F.adaptive_avg_pool2d(x, (1, 1))
        # weight_part_feat = weight_part_feat + x.view(x.size(0), x.size(1))

        if self.fuse == 'sum': 
            feats = weight_part_feat + feat
        elif self.fuse == 'cat':
            feats = torch.cat([weight_part_feat,feat],dim=1)
        feats_bn = self.bottleneck(feats)

        return feats, feats_bn


