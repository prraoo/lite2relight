#!/usr/bin/python
# -*- encoding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch_utils.ops import bias_act
import numpy as np

"""
Reference: https://github.com/NVlabs/SegFormer/blob/65fa8cfa9b52b6ee7e8897a98705abf8570f9e32/mmseg/models/backbones/mix_transformer.py#L203
"""

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TransformerBlock(nn.Module):
    """
    Self-attention Block
    """
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        
        return x


class OverlapPatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    Source: https://github.com/NVlabs/SegFormer/blob/65fa8cfa9b52b6ee7e8897a98705abf8570f9e32/mmseg/models/backbones/mix_transformer.py#L160
    """

    def __init__(self, img_size=64, patch_size=3, stride=2, in_chans=256, embed_dim=1024):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

# ------------------------------------------------------------------------


def conv3x3(in_channels, out_channels, stride=1, kernel_size=3):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=1, bias=False)


# Reflectance Decoder
class UnetReflectanceBlock(nn.Module):
    """
    Reflectance network with input conditioned on OLAT directions
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_OLAT_dirs=False, n_OLATs=0, use_fp16=False, fp16_channels_last=False):
        super(UnetReflectanceBlock, self).__init__()
        self.use_OLAT_dirs = use_OLAT_dirs
        self.n_OLATs = n_OLATs
        self.use_fp16 = False
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.channels = 64

        # if self.use_OLAT_dirs:
        #     self.conv1 = conv3x3(in_channels+n_OLATs*3, out_channels, stride)
        #     self.bn1 = nn.BatchNorm2d(out_channels)
        #     if (stride != 1) or (in_channels != out_channels):
        #         downsample = nn.Sequential(conv3x3(in_channels+n_OLATs*3, out_channels, stride=stride), nn.BatchNorm2d(out_channels))
        # else:

        # Block 0
        self.conv0 = conv3x3(in_channels, self.channels, stride=2)
        self.conv1 = conv3x3(self.channels, self.channels*2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(self.channels*2, self.channels*4, stride=2)

        self.downsample = nn.Conv2d(in_channels, self.channels*4, kernel_size=8, stride=8, padding=0)
        self.conv3 = conv3x3(self.channels*8, self.channels*4, stride=1)
        # Block 1
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.conv4 = conv3x3(self.channels*4, self.channels*2, stride=1)
        self.conv5 = conv3x3(self.channels*2, self.channels, stride=1)
        self.conv6 = conv3x3(self.channels, out_channels, stride=1)

    def forward(self, x, force_fp32=False):
        # input
        residual = x

        out = self.conv0(x)
        out = self.relu(out)

        # Downsample
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)

        # Residual
        if self.downsample:
            residual = self.downsample(x)
        out = torch.cat([residual, out], dim=1)
        out = self.conv3(out)
        out = self.relu(out)

        # Upsample
        out = self.upsample(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.upsample(out)
        out = self.conv5(out)
        out = self.relu(out)

        # Final Layer
        out = self.upsample(out)
        out = self.conv6(out)

        return out


class UnetEncoder(nn.Module):
    def __init__(self, block:nn.Module, in_channels:int, out_channels:int, use_OLAT_dirs=True, n_OLATs=1):
        super().__init__()
        self.use_OLAT_dirs = use_OLAT_dirs
        self.n_OLATs = n_OLATs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resblock = self.make_layer(block, self.in_channels, self.out_channels, blocks=1)

    def make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, 64, stride=4),
                # nn.BatchNorm2d(out_channels //4)
            )
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample, self.use_OLAT_dirs, self.n_OLATs))
        in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, features, camera_pose=None, sampled_location=None):
        out = self.resblock(features)
        return out


# ------------------------------------------------------------------------


class ReflectanceTriplaneViTNetwork(nn.Module):
    def __init__(self, img_size=256, stride=2, in_chans=128, embed_dim=1024, num_heads=2, mlp_ratio=2, sr_ratio=2):
        super().__init__()

        # OverLapPatchEmbed
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.stride = stride
        self.in_chans = in_chans
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.sr_ratio = sr_ratio

        self.conv1 = nn.Conv2d(96+3, 128, kernel_size=3, stride=1, padding=1)
        self.patch_embed = OverlapPatchEmbed(img_size=img_size, stride=self.stride, in_chans=self.in_chans, embed_dim=self.embed_dim)
        # TransformerBlocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, sr_ratio=self.sr_ratio, qkv_bias=True, drop_path=0),
        ])
        # Upsampling and convolution layers
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.conv2 = nn.Conv2d(256+96, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1)
        self.Leaky_ReLU = nn.LeakyReLU(negative_slope=0.001)

    def forward(self, dirs, in_feat):
        x=torch.cat([dirs, in_feat], dim=1)
        x = self.Leaky_ReLU(self.conv1(x))
        x, H, W = self.patch_embed(x)
        for i, blk in enumerate(self.transformer_blocks):
            x = blk(x, H, W)

        x = x.view(-1, self.embed_dim, H, W)
        x = self.pixel_shuffle(x)
        x = self.Leaky_ReLU(self.conv2(torch.cat([x, in_feat], dim=1)))
        x = self.Leaky_ReLU(self.conv3(x))
        x = self.Leaky_ReLU(self.conv4(x))
        x = self.Leaky_ReLU(self.conv5(x))
        return x


# ------------------------------------------------------------------------


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True, negative_slope=0.01)
        )

    def forward(self, x):
        return self.conv(x)



class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # Adjusted number of input channels here
            nn.LeakyReLU(inplace=True, negative_slope=0.01)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # Concatenate along channel dimension
        x = self.conv(x)
        return x


class HourglassEncoder(nn.Module):
    def __init__(self, base_channels=64, depth=3, in_channels=99, out_channels=96):
        super(HourglassEncoder, self).__init__()
        self.first_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        # Down Blocks
        self.down_blocks = nn.ModuleList()
        channels = base_channels
        for i in range(depth):
            self.down_blocks.append(DownBlock(channels, channels * 2))
            channels *= 2
        # Up Blocks
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            self.up_blocks.append(UpBlock(channels, channels // 2))
            channels //= 2

        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.first_conv(x)
        skip_connections = []
        for i, block in enumerate(self.down_blocks):
            skip_connections.append(x)
            x = block(x)

        for block in self.up_blocks:
            skip = skip_connections.pop()
            x = block(x, skip)

        x = self.final_conv(x)
        return x


# ------------------------------------------------------------------------


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Reflectance Decoder
class CNNReflectanceBlock(nn.Module):
    """
    Reflectance network with input conditioned on OLAT directions
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_OLAT_dirs=False, n_OLATs=0, use_fp16=False, fp16_channels_last=False):
        super(CNNReflectanceBlock, self).__init__()
        self.use_OLAT_dirs = use_OLAT_dirs
        self.n_OLATs = n_OLATs
        self.use_fp16 = False
        self.channels_last = (use_fp16 and fp16_channels_last)

        # Block 1
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Block 2
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        # Block 3
        self.conv3 = conv3x3(out_channels, out_channels, stride)
        self.bn3 = nn.BatchNorm2d(out_channels)
        # Block 4
        self.conv4 = conv3x3(out_channels, out_channels)
        self.bn4 = nn.BatchNorm2d(out_channels)
        # Activation Function
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.01)

    def forward(self, in_feat, force_fp32=False):

        # Block 1
        out = self.conv1(in_feat)
        out = self.bn1(out)
        out = self.relu(out)
        # Block 2
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(in_feat)
        else:
            residual = in_feat
        out += residual
        out = self.relu(out)
        # Block 3
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        # Block 4
        out = self.conv4(out)
        out = self.bn4(out)
        # out = self.relu(out)

        return out


class CNNEncoder(nn.Module):
    def __init__(self, block: nn.Module, in_channels: int, out_channels: int, use_OLAT_dirs: bool, n_OLATs: int):
        super().__init__()
        self.use_OLAT_dirs = use_OLAT_dirs
        self.n_OLATs = n_OLATs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resblock = self.make_layer(block, self.in_channels, self.out_channels)

    def make_layer(self, block, in_channels, out_channels, blocks=1, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, self.out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample, self.use_OLAT_dirs, self.n_OLATs))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, features, camera_pose=None, sampled_location=None):
        out = self.resblock(features)
        return out

# ----------------------------------------------------------------------------

class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
                 in_features,  # Number of input features.
                 out_features,  # Number of output features.
                 bias=True,  # Apply additive bias before the activation function?
                 activation='linear',  # Activation function: 'relu', 'lrelu', etc.
                 lr_multiplier=1,  # Learning rate multiplier.
                 bias_init=0,  # Initial value for the additive bias.
                 ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

class ReflectanceDecoder(torch.nn.Module):
    def __init__(self, in_features, options, relight_mode=False, act_fn=None):
        super().__init__()
        self.hidden_dim = 64
        self.relight_mode = relight_mode
        if relight_mode:
            assert act_fn is not None
            self.act_fn = act_fn

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(in_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )

    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N * M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        if self.relight_mode:
            if self.act_fn == 'sigmoid':
                rgb = torch.sigmoid(x) * (1 + 2 * 0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
            elif self.act_fn == 'exp':
                x = torch.clamp_max(x, 88.)  # exp(89) = inf
                rgb = torch.exp(x)
            else:
                raise NotImplementedError
            return {'rgb': rgb}
        else:
            rgb = torch.sigmoid(x) * (1 + 2 * 0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
            return {'rgb': rgb}


# ------------------------------------------------------------------------
if __name__ == '__main__':
    print("Debugging Reflectance Encoder ...")
    # Create an instance of the network
    model = CNNEncoder(block=CNNReflectanceBlock, in_channels=96 + 3, out_channels=96, use_OLAT_dirs=True, n_OLATs=1).cuda()
    # model = UnetEncoder(block=ReflectanceUNetBlock, in_channels=96 + 3, out_channels=96, use_OLAT_dirs=True, n_OLATs=1).cuda()
    # model = ReflectanceTriplaneViTNetwork().cuda()
    # model = HourglassNetwork(depth=3).cuda()

    # Create a dummy input tensor with dimensions Nx256x64x64
    omega = torch.randn(4,3).cuda()
    OLAT_dirs_planes = omega[:, :, None, None].tile(1, 1, 256, 256) # N x dirs x H x W
    dummy_input = torch.randn(4, 96+3, 256, 256).cuda()
    dummy_F = torch.randn(4, 96, 256, 256).cuda()

    # Test the forward pass
    print(model)
    output = model(dummy_input)
    print(output.shape)
    print("Finished Reflectance Encoder!")
