#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

# Redefining the CNNEncoder model

class HighFeatEncoderNetwork(nn.Module):
    def __init__(self, img_ch, out_ch):
        super(HighFeatEncoderNetwork, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(img_ch, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(96, out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.01)
        )

    def forward(self, x):
        return self.encoder(x)

if __name__ == '__main__':
    # Create an instance of the Encoder
    encoder = HighFeatEncoderNetwork().cuda()

    # Testing the CNNEncoder with a random input
    input_tensor = torch.randn(1, 5, 64, 64).cuda()  # Batch size of 1, 5 channels, 64x64 image
    print('Input: ', input_tensor.shape)
    output = encoder(input_tensor)
    print('Output: ', output.shape)
