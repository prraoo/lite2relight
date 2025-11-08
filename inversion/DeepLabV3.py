#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch_utils import persistence


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def replace_bn_with_identity(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            setattr(model, name, Identity())
        else:
            replace_bn_with_identity(module)


class DeepLabV3_encoder(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()

        self.segmentation_model = smp.DeepLabV3(encoder_name='resnet34', encoder_depth=5, encoder_weights='imagenet',
                                                decoder_channels=256, in_channels=in_ch, classes=1, activation=None,
                                                upsampling=8, aux_params=None)

        self.segmentation_model.segmentation_head = None
        replace_bn_with_identity(self.segmentation_model)
        
    def forward(self, input):
        x = self.segmentation_model.encoder(input)
        x = self.segmentation_model.decoder(*x)
        return x

if __name__ == '__main__':
    print("Debugging DeepLabV3 ...")
    model = DeepLabV3_encoder(in_ch=3)
    img = torch.randn(4, 3, 512, 512)
    encoder_out = model.segmentation_model.encoder(img)
    result = model.segmentation_model.decoder(*encoder_out)
    print(result.shape)
    print("Finished DeepLabV3 debugging!")
