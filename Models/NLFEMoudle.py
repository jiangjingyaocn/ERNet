import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# -*- coding: utf-8 -*-
from .ICE import ICE

affine_par = True
from torch.autograd import Variable
# from lib.GatedConv import GatedConv2dWithActivation

class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False): # num_state=384 num_node=16
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x): # x [16,384,16]
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class NLFEnhence(nn.Module):
    def __init__(self, dim_in = 64, dim_temp=32, img_size=224, mids=4, tmp_chs = 14):
        super(NLFEnhence, self).__init__()

        self.img_size = img_size
        self.dim_in = dim_in
        self.dim_temp = dim_temp
        self.num_n = mids * mids
        self.tmp_chs = tmp_chs
        self.conv_fc = nn.Conv2d(self.dim_in, self.dim_temp, kernel_size=1)

        # f1
        self.norm_layer_f1 = nn.LayerNorm(dim_in)
        self.conv_f1_Q = nn.Conv2d(self.dim_in, self.dim_temp, kernel_size=1)
        self.conv_f1_K = nn.Conv2d(self.dim_in, self.dim_temp, kernel_size=1)
        self.ap_f1 = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))
        self.gcn_f1 = GCN(num_state=self.dim_temp, num_node=self.num_n)
        self.conv_f1_extend = nn.Conv2d(self.dim_temp, self.dim_in, kernel_size=1, bias=False)
        self.ice = ICE()
        # f2
        self.norm_layer_f2 = nn.LayerNorm(dim_in)

        self.align = nn.Sequential(
            nn.Linear(tmp_chs*tmp_chs * 5, tmp_chs*tmp_chs * 5),
            nn.GELU(),
            nn.Linear(tmp_chs*tmp_chs * 5, tmp_chs*tmp_chs * 4),
        )
    def forward(self, f_big,f_little):
        # f_big [11, 784, 64] f_littl [11, 196, 64]

        bs, num_token, chs = f_big.shape
        bs, num_token_little, chs = f_little.shape


        # print('f_big.shape',f_big.shape)
        # print('f_little.shape',f_little.shape)
        f1_ = self.norm_layer_f1(f_big)
        # print('f1_.shape',f1_.shape)
        f2_ = self.norm_layer_f2(f_little)
        # print('f2_.shape',f2_.shape)

        fea1 = f1_.permute(0, 2, 1).view(bs, chs, int(np.sqrt(num_token)), int(np.sqrt(num_token))).contiguous()
        fea2 = f2_.permute(0, 2, 1).view(bs, chs, int(np.sqrt(num_token_little)), int(np.sqrt(num_token_little))).contiguous()

        fc = self.ice(in1 = fea1,in2 = fea2)
        fc = self.conv_fc(fc)
        # print('fc', fc.shape)
        f1_ = f1_.permute(0, 2, 1).view(bs, chs, int(np.sqrt(num_token)), int(np.sqrt(num_token))).contiguous()
        # print('f1_',f1_.shape)
        # f1_ torch.Size([11, 64, 28, 28])
        f2_ = f2_.permute(0, 2, 1).view(bs, chs, int(np.sqrt(num_token_little)), int(np.sqrt(num_token_little))).contiguous()
        # print('f2_.shape',f2_.shape)
        f1, f2 = f1_, f2_
        fc_att = torch.nn.functional.softmax(fc, dim=1)[:, 1, :, :].unsqueeze(1)
        # print('fc_att.shape',fc_att.shape)
        f1_Q = self.conv_f1_Q(f1).view(bs, self.dim_temp, -1).contiguous()
        # print('f1_Q.shape',f1_Q.shape)
        f1_K = self.conv_f1_K(f1)
        # print('f1_K.shape',f1_K.shape)
        f1_masked = f1_K * fc_att
        # print('f1_masked.shape',f1_masked.shape)
        f1_V = self.ap_f1(f1_masked)[:, :, 1:-1, 1:-1].reshape(bs, self.dim_temp, -1)
        # print('f1_V.shape',f1_V.shape)
        f1_proj_reshaped = torch.matmul(f1_V.permute(0, 2, 1), f1_K.reshape(bs, self.dim_temp, -1))
        # print('f1_proj_reshaped',f1_proj_reshaped.shape)
        f1_proj_reshaped = torch.nn.functional.softmax(f1_proj_reshaped, dim=1)
        # print('f1_proj_reshaped',f1_proj_reshaped.shape)
        f1_rproj_reshaped = f1_proj_reshaped
        # print('f1_rproj_reshaped',f1_rproj_reshaped.shape)
        f1_n_state = torch.matmul(f1_Q, f1_proj_reshaped.permute(0, 2, 1))
        # print('f1_n_state',f1_n_state.shape)
        # f1_n_rel = f1_n_state
        f1_n_rel = self.gcn_f1(f1_n_state)
        # print('f1_n_rel',f1_n_rel.shape)
        f1_state_reshaped = torch.matmul(f1_n_rel, f1_rproj_reshaped)
        # print('f1_state_reshaped',f1_state_reshaped.shape)
        f1_state = f1_state_reshaped.view(bs, self.dim_temp, *f1.size()[2:])
        # print('f1_state',f1_state.shape)
        # f1_state_rereshaped = f1_state.view(bs,self.dim_temp, *f1.size()[2:])
        # print('f1_state_rereshaped',f1_state_rereshaped.shape)
        f1_out = f1_ +(self.conv_f1_extend(f1_state))
        # print('f1_out',f1_out.shape)
        f1_out = f1_out.view(bs,self.dim_in,-1).transpose(1,2)
        # print('f1_out',f1_out.shape)
        return f1_out