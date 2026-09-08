"""Unified Tasks 2-4 classifier — `class LE`.

Tasks 2–4 use the same LightM-UNet (nnU-Net v2) decoder-hook feature source
and 3D FPN formulation; each task has its own checkpoint.

This is the classifier whose probabilities match the manuscript's official
Task workbooks. It consumes the 3D decoder feature pyramid of the LightM-UNet
segmentation network (feat0/feat1/feat2), not an image and not an axial section.

The released implementation is the unified Tasks 2–4 `class LE` formulation.

`LE5` below is the five-class pathology variant: identical to
`LE` except for an extra `projection = conv3d(32, 32, 1)` before pooling and a
5-way head. It produces the pathology five-class column in the supplied Task 2 workbook
(labels 1-5; label 0 "normal" is supplied by Task 1). 10,574,862 params.

Every spatial operator is 3D (Conv3d / MaxPool3d / avg_pool3d). There is no 2D
stage anywhere in this path.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


def conv3d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv3d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm3d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([nn.MaxPool3d(pool_size, 1, pool_size // 2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv3d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        x = self.upsample(x)
        return x


class LE(nn.Module):
    """Deployed two-class classifier used separately for Tasks 2–4."""
    def __init__(self):
        super(LE, self).__init__()
        self.conv1 = conv3d(256, 64, 1)
        self.SPP = SpatialPyramidPooling()
        self.conv2 = conv3d(256, 128, 1)

        self.upsample1 = Upsample(128, 64)
        self.conv_for_P4 = conv3d(128, 64, 1)
        self.transfer1 = conv3d(128, 64, 1)

        self.upsample2 = Upsample(64, 32)
        self.conv_for_P3 = conv3d(64, 32, 1)
        self.transfer2 = conv3d(64, 32, 1)

        self.linear1 = nn.Linear(10240, 1024)
        self.linear2 = nn.Linear(1024, 2)

    def forward(self, feat0, feat1, feat2):
        P5 = self.conv1(feat0)
        P5 = self.SPP(P5)                              # [1, 256, 10, 16, 16]
        P5 = self.conv2(P5)                            # [1, 128, 10, 16, 16]

        P5_upsample = self.upsample1(P5)               # [1, 64, 20, 32, 32]
        P4 = self.conv_for_P4(feat1)
        P4 = torch.cat([P4, P5_upsample], axis=1)      # [1, 128, 20, 32, 32]
        P4 = self.transfer1(P4)                        # [1, 64, 20, 32, 32]

        P4_upsample = self.upsample2(P4)               # [1, 32, 40, 64, 64]
        P3 = self.conv_for_P3(feat2)                   # [1, 32, 40, 64, 64]
        P3 = torch.cat([P3, P4_upsample], axis=1)      # [1, 64, 40, 64, 64]
        P3 = self.transfer2(P3)                        # [1, 32, 40, 64, 64]

        out = F.avg_pool3d(P3, 8)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out


class LE5(nn.Module):
    """Five-class pathology variant.

    Identical to LE plus `projection = conv3d(32, 32, 1)` before pooling and a
    5-way head. Produces the pathology five-class column in the supplied Task 2 workbook.
    """
    def __init__(self):
        super(LE5, self).__init__()
        self.conv1 = conv3d(256, 64, 1)
        self.SPP = SpatialPyramidPooling()
        self.conv2 = conv3d(256, 128, 1)

        self.upsample1 = Upsample(128, 64)
        self.conv_for_P4 = conv3d(128, 64, 1)
        self.transfer1 = conv3d(128, 64, 1)

        self.upsample2 = Upsample(64, 32)
        self.conv_for_P3 = conv3d(64, 32, 1)
        self.transfer2 = conv3d(64, 32, 1)

        self.projection = conv3d(32, 32, 1)

        self.linear1 = nn.Linear(10240, 1024)
        self.linear2 = nn.Linear(1024, 5)

    def forward(self, feat0, feat1, feat2):
        P5 = self.conv1(feat0)
        P5 = self.SPP(P5)
        P5 = self.conv2(P5)

        P5_upsample = self.upsample1(P5)
        P4 = self.conv_for_P4(feat1)
        P4 = torch.cat([P4, P5_upsample], axis=1)
        P4 = self.transfer1(P4)

        P4_upsample = self.upsample2(P4)
        P3 = self.conv_for_P3(feat2)
        P3 = torch.cat([P3, P4_upsample], axis=1)
        P3 = self.transfer2(P3)

        P3 = self.projection(P3)

        out = F.avg_pool3d(P3, 8)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out

