# -*- coding: utf-8 -*-
import torch.nn as nn

affine_par = True
import torch
from torch.nn import functional as F
from torch.autograd import Variable
# from lib.GatedConv import GatedConv2dWithActivation
import numpy as np



"""
    Ordinary Differential Equation (ODE)
"""


class getAlpha(nn.Module):
    def __init__(self, in_channels):
        super(getAlpha, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class ODE(nn.Module):
    def __init__(self, in_channels):
        super(ODE, self).__init__()
        self.F1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        self.F2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        self.getalpha = getAlpha(in_channels)

    def forward(self, feature_map):
        f1 = self.F1(feature_map)
        f2 = self.F2(f1 + feature_map)
        alpha = self.getalpha(torch.cat([f1, f2], dim=1))
        out = feature_map + f1 * alpha + f2 * (1 - alpha)

        return out

class REEDGE(nn.Module):
        def __init__(self, in_channels,enfchn):
            super(REEDGE, self).__init__()
            self.F1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            )
            self.F2 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            )
            self.F3 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            )
            self.getalpha1 = getAlpha(in_channels)
            self.getalpha2 = getAlpha(in_channels)
            self.alignchn = nn.Conv2d(in_channels=enfchn, out_channels=in_channels, kernel_size=1)


        def forward(self, feature_map,decoder_map):
            B, HW, C = feature_map.shape  # rgb_fea_1_8 [B, 28*28, 64]
            feature_map = feature_map.transpose(1, 2).contiguous().reshape(B, C, int(np.sqrt(HW)),
                                                                           int(np.sqrt(HW)))
            decoder_map = self.alignchn(decoder_map)
            f1 = self.F1(feature_map)
            f2 = self.F2(decoder_map + feature_map)
            alpha1 = self.getalpha1(torch.cat([f1, f2], dim=1))
            out1 = f1 * alpha1 + f2 * (1 - alpha1)
            f3 = f1
            f4 = self.F3(decoder_map + out1)
            alpha2 = self.getalpha2(torch.cat([f3, f4], dim=1))
            out = feature_map + f3 * alpha2 + f4 * (1 - alpha2)
            out = out.reshape(B, C, HW).transpose(1, 2).contiguous()
            return out