import torch.nn as nn
import torch
from .token_performer import Token_performer
from .Transformer import saliency_token_inference, contour_token_inference, token_TransformerEncoder
from . import ERMoudle
from . import NLFEMoudle
import numpy as np
from .ICE import ICE

class token_trans(nn.Module):
    def __init__(self, in_dim=64, embed_dim=512, depth=14, num_heads=6, mlp_ratio=3.):
        super(token_trans, self).__init__()

        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.encoderlayer = token_TransformerEncoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.saliency_token_pre = saliency_token_inference(dim=embed_dim, num_heads=1)
        self.contour_token_pre = contour_token_inference(dim=embed_dim, num_heads=1)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

        self.norm2_c = nn.LayerNorm(embed_dim)
        self.mlp2_c = nn.Sequential(
            nn.Linear(embed_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

    def forward(self, fea, saliency_tokens, contour_tokens):
        B, _, _ = fea.shape
        # fea [B, H*W, 64]
        # project to 512 dim
        fea = self.mlp(self.norm(fea))
        # fea [B, H*W, 512]

        fea = torch.cat((saliency_tokens, fea), dim=1)
        fea = torch.cat((fea, contour_tokens), dim=1)
        # [B, 1 + H*W + 1, 512]

        fea = self.encoderlayer(fea)
        # fea [B, 1 + H*W + 1, 512]
        saliency_tokens = fea[:, 0, :].unsqueeze(1)
        contour_tokens = fea[:, -1, :].unsqueeze(1)

        saliency_fea = self.saliency_token_pre(fea)
        # saliency_fea [B, H*W, 512]
        contour_fea = self.contour_token_pre(fea)
        # contour_fea [B, H*W, 512]

        # reproject back to 64 dim
        saliency_fea = self.mlp2(self.norm2(saliency_fea))
        contour_fea = self.mlp2_c(self.norm2_c(contour_fea))

        return saliency_fea, contour_fea, fea, saliency_tokens, contour_tokens


class decoder_module(nn.Module):
    def __init__(self, dim=512, token_dim=64, img_size=224, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True):
        super(decoder_module, self).__init__()

        self.project = nn.Linear(token_dim, token_dim * kernel_size[0] * kernel_size[1])
        self.upsample = nn.Fold(output_size=(img_size // ratio,  img_size // ratio), kernel_size=kernel_size, stride=stride, padding=padding)
        self.fuse = fuse
        if self.fuse:
            self.concatFuse = nn.Sequential(
                nn.Linear(token_dim*2, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )
            self.att = Token_performer(dim=token_dim, in_dim=token_dim, kernel_ratio=0.5)

            # project input feature to 64 dim
            self.norm = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )

    def forward(self, dec_fea, enc_fea=None):

        if self.fuse:
            # from 512 to 64
            dec_fea = self.mlp(self.norm(dec_fea))

        # [1] token upsampling by the proposed reverse T2T module
        dec_fea = self.project(dec_fea)
        # [B, H*W, token_dim*kernel_size*kernel_size]
        dec_fea = self.upsample(dec_fea.transpose(1, 2).contiguous())
        B, C, _, _ = dec_fea.shape
        dec_fea = dec_fea.view(B, C, -1).transpose(1, 2).contiguous()
        # [B, HW, C]

        if self.fuse:
            # [2] fuse encoder fea and decoder fea
            dec_fea = self.concatFuse(torch.cat([dec_fea, enc_fea], dim=2))
            dec_fea = self.att(dec_fea)

        return dec_fea


class Decoder(nn.Module):
    def __init__(self, embed_dim=512, token_dim=64, depth=2, img_size=224):

        super(Decoder, self).__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )

        self.norm_c = nn.LayerNorm(embed_dim)
        self.mlp_c = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, token_dim),
        )
        self.img_size = img_size
        # token upsampling and multi-level token fusion
        self.decoder1 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder2 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), fuse=True)
        self.decoder3 = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=1, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2), fuse=False)
        self.decoder3_c = decoder_module(dim=embed_dim, token_dim=token_dim, img_size=img_size, ratio=1, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2), fuse=False)

        self.reedge_1_16 = ERMoudle.REEDGE(token_dim,enfchn=1024 )
        self.reedge_1_8 = ERMoudle.REEDGE(token_dim,enfchn =512 )
        self.reedge_1_4 = ERMoudle.REEDGE(token_dim,enfchn = 256)
        self.ode_1_1 = ERMoudle.ODE(token_dim)
        self.reedge_s = ERMoudle.REEDGE(token_dim,enfchn = 64)
        self.reedge_c = ERMoudle.REEDGE(token_dim,enfchn = 64)
        self.edge = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3,stride=1, padding=1)
        )
        # feature fusion
        self.nlfea_1_8 = NLFEMoudle.NLFEnhence(dim_in=int(64),dim_temp=int(32),img_size=224)
        self.nlfea_1_4 = NLFEMoudle.NLFEnhence(dim_in=int(64),dim_temp=int(32),img_size=224,tmp_chs=28)
        # self.nlfea_1_2 = NLFEMoudle.NLFEnhence(dim_in=int(64),dim_temp=int(32),img_size=224,tmp_chs=56)
        # token based multi-task predictions
        self.token_pre_1_8 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)
        self.token_pre_1_4 = token_trans(in_dim=token_dim, embed_dim=embed_dim, depth=depth, num_heads=1)

        # predict saliency maps
        self.pre_1_16 = nn.Linear(token_dim, 1)
        self.pre_1_8 = nn.Linear(token_dim, 1)
        self.pre_1_4 = nn.Linear(token_dim, 1)
        self.pre_1_1 = nn.Linear(token_dim, 1)
        # predict contour maps
        self.pre_1_16_c = nn.Linear(token_dim, 1)
        self.pre_1_8_c = nn.Linear(token_dim, 1)
        self.pre_1_4_c = nn.Linear(token_dim, 1)
        self.pre_1_1_c = nn.Linear(token_dim, 1)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # def forward(self, saliency_fea_1_16, token_fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens, rgb_fea_1_8,
    #             rgb_fea_1_4):    #             rgb_fea_1_4):
    def forward(self, saliency_fea_1_16, token_fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens,
                    rgb_fea_1_8,rgb_fea_1_4, r2, r3, r4, image_Input):
        B, _, _, = token_fea_1_16.size()
        saliency_fea_1_16 = self.mlp(self.norm(saliency_fea_1_16))
        saliency_fea_1_16 = self.reedge_1_16(saliency_fea_1_16,r4)
        # print('saliency_fea_1_16',saliency_fea_1_16.shape)

        mask_1_16 = self.pre_1_16(saliency_fea_1_16)
        mask_1_16 = mask_1_16.transpose(1, 2).contiguous().reshape(B, 1, self.img_size // 16, self.img_size // 16)


        # # 特征融合
        # # 1/16 1/8 -> 1/8#[B, 14*14, 64]
        # print('rgb_fea_1_8',rgb_fea_1_8.shape)
        rgb_fea_1_8 = self.nlfea_1_8(rgb_fea_1_8, saliency_fea_1_16)
        # print('rgb_fea_1_8',rgb_fea_1_8.shape)


        contour_fea_1_16 = self.mlp_c(self.norm_c(contour_fea_1_16))
        # contour_fea_1_16 [B, 14*14, 64]
        contour_1_16 = self.pre_1_16_c(contour_fea_1_16)
        contour_1_16 = contour_1_16.transpose(1, 2).contiguous().reshape(B, 1, self.img_size // 16, self.img_size // 16)


        # 特征融合
        # # 1/8 1/4 -> 1/4
        rgb_fea_1_4 = self.nlfea_1_4(rgb_fea_1_4, rgb_fea_1_8)
        # reverse T2T and fuse low-level feature
        fea_1_8 = self.decoder1(token_fea_1_16[:, 1:-1, :], rgb_fea_1_8)
        # ([11, 196, 512] ([11, 784, 64])
        # token prediction
        saliency_fea_1_8, contour_fea_1_8, token_fea_1_8, saliency_tokens, contour_tokens = self.token_pre_1_8(fea_1_8, saliency_tokens,contour_tokens)
        # # 1/8的ODE模块
        saliency_fea_1_8 = self.reedge_1_8(saliency_fea_1_8,r3)
        # predict saliency maps and contour maps
        mask_1_8 = self.pre_1_8(saliency_fea_1_8)
        mask_1_8 = mask_1_8.transpose(1, 2).contiguous().reshape(B, 1, self.img_size // 8, self.img_size // 8)

        contour_1_8 = self.pre_1_8_c(contour_fea_1_8)
        contour_1_8 = contour_1_8.transpose(1, 2).contiguous().reshape(B, 1, self.img_size // 8, self.img_size // 8)

        # 1/8 -> 1/4
        fea_1_4 = self.decoder2(token_fea_1_8[:, 1:-1, :], rgb_fea_1_4)

        # token prediction
        saliency_fea_1_4, contour_fea_1_4, token_fea_1_4, saliency_tokens, contour_tokens = self.token_pre_1_4(fea_1_4,
                                                                                                               saliency_tokens,
                                                                                                               contour_tokens)
        # 1/4的ODE模块
        saliency_fea_1_4 = self.reedge_1_4(saliency_fea_1_4,r2)

        # predict saliency maps and contour maps
        mask_1_4 = self.pre_1_4(saliency_fea_1_4)
        mask_1_4 = mask_1_4.transpose(1, 2).contiguous().reshape(B, 1, self.img_size // 4, self.img_size // 4)

        contour_1_4 = self.pre_1_4_c(contour_fea_1_4)
        contour_1_4 = contour_1_4.transpose(1, 2).contiguous().reshape(B, 1, self.img_size // 4, self.img_size // 4)

        edge = self.edge(image_Input)
        decoder_edge = self.ode_1_1(edge)

        # 1/4 -> 1
        saliency_fea_1_1 = self.decoder3(saliency_fea_1_4)
        contour_fea_1_1 = self.decoder3_c(contour_fea_1_4)
        saliency_fea_1_1 = self.reedge_s(saliency_fea_1_1,decoder_edge)
        contour_fea_1_1 = self.reedge_s(contour_fea_1_1,decoder_edge)

        mask_1_1 = self.pre_1_1(saliency_fea_1_1)
        mask_1_1 = mask_1_1.transpose(1, 2).reshape(B, 1, self.img_size // 1, self.img_size // 1)

        contour_1_1 = self.pre_1_1_c(contour_fea_1_1)
        contour_1_1 = contour_1_1.transpose(1, 2).reshape(B, 1, self.img_size // 1, self.img_size // 1)

        return [mask_1_16, mask_1_8, mask_1_4, mask_1_1], [contour_1_16, contour_1_8, contour_1_4, contour_1_1]
