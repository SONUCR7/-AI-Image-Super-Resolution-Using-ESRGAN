"""
RRDBNet Architecture Definition
Reference: "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"
Authors: Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Chen Change Loy, Yu Qiao, Xiaoou Tang
ECCV 2018 Workshops (PIRM 2018 Super-Resolution Challenge 1st Place)
"""

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_layer(basic_block, num_basic_block, **kwargs):
    """Stack identical basic blocks in sequence."""
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwargs))
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block with 5 Convolutional Layers.
    Features dense skip connections between all internal convolutional layers,
    allowing hierarchical feature reuse before scaling by a residual factor of 0.2.
    """
    def __init__(self, nf: int = 64, gc: int = 32, bias: bool = True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Residual scaling factor of 0.2 stabilizes training in deep networks without BN
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block (RRDB).
    Combines three Residual Dense Blocks (RDBs) with a higher-level residual connection.
    Eliminates Batch Normalization layers to avoid artifacts and improve dynamic range.
    """
    def __init__(self, nf: int = 64, gc: int = 32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """
    Complete Deep Generator Network for ESRGAN (4x Super-Resolution).
    Consists of:
      1. Shallow feature extraction (conv_first)
      2. Deep trunk: 23 RRDB blocks + trunk_conv with long skip connection
      3. Sub-pixel / Nearest-neighbor upsampling (2x * 2x = 4x)
      4. HR reconstruction convolution layers
    """
    def __init__(self, in_nc: int = 3, out_nc: int = 3, nf: int = 64, nb: int = 23, gc: int = 32):
        super(RRDBNet, self).__init__()
        rrdb_block = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(rrdb_block, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # 4x Upsampling pipeline (2 stages of 2x upsampling)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # High resolution reconstruction
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fea_first = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea_first))
        fea = fea_first + trunk

        # Stage 1: 2x upsampling
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # Stage 2: 2x upsampling (Total: 4x)
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out
