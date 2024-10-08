import torch
import timm
import torch.nn as nn
from .t2t_vit import T2t_vit_t_24
from .Transformer import Transformer
from .Transformer import token_Transformer
from .Decoder import Decoder
from .Res import resnet50

class ImageDepthNet(nn.Module):
    def __init__(self, args):
        super(ImageDepthNet, self).__init__()
         # VST Encoder
        self.rgb_backbone = T2t_vit_t_24(pretrained=True, args=args)
        self.resnet_backbone = resnet50()
        self.resnet_backbone.load_state_dict(torch.load('./pretrained_model/resnet50.pth'),strict=False)
        # VST Convertor
        self.transformer = Transformer(embed_dim=512, depth=4, num_heads=8, mlp_ratio=3.)

        # VST Decoder
        self.token_trans = token_Transformer(embed_dim=512, depth=4, num_heads=8, mlp_ratio=3.)
        self.decoder = Decoder(embed_dim=512, token_dim=64, depth=2, img_size=args.img_size)

    def forward(self, image_Input):

        B, _, _, _ = image_Input.shape
        # VST Encoder
        rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4 = self.rgb_backbone(image_Input)
        r2, r3, r4, _ = self.resnet_backbone(image_Input)
        # print('r2',r2.shape, r3.shape, r4.shape)
        # print('rgb_fea_1_16.shape',rgb_fea_1_16.shape)
        # VST Convertor
        rgb_fea_1_16 = self.transformer(rgb_fea_1_16)
        # # rgb_fea_1_16 [B, 14*14, 384]
        # print('rgb_fea_1_16.shape',rgb_fea_1_16.shape)

        # VST Decoder
        saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens = self.token_trans(rgb_fea_1_16)
        # saliency_fea_1_16 [B, 14*14, 384]
        # fea_1_16 [B, 1 + 14*14 + 1, 384]
        # saliency_tokens [B, 1, 384]
        # contour_fea_1_16 [B, 14*14, 384]
        # contour_tokens [B, 1, 384]

        # outputs = self.decoder(saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens, rgb_fea_1_8, rgb_fea_1_4,r2, r3, r4, r5,r2, r3, r4)
        outputs = self.decoder(saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens, rgb_fea_1_8, rgb_fea_1_4,r2, r3, r4,image_Input)

        return outputs
