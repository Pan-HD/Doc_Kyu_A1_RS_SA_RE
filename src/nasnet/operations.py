"""PyTorch operation factory for the NASNet search space."""

import torch
import torch.nn as nn


OPS = (
    "identity",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "sep_conv_7x7",
    "avg_pool_3x3",
    "max_pool_3x3",
    "dil_sep_conv_3x3",
    "conv_1x7_7x1",
)


BN_EPS = 1e-5
BN_MOMENTUM = 0.1


def make_bn(channels: int) -> nn.Module:
    """Build the BatchNorm layer shared by all operations."""
    return nn.BatchNorm2d(
        channels,
        eps=BN_EPS,
        momentum=BN_MOMENTUM,
        affine=True,
    )


class SepConvBlock(nn.Module):
    """ReLU followed by depthwise and pointwise convolutions and BatchNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        padding = dilation * (kernel_size - 1) // 2

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            make_bn(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class StackedSepConv(nn.Module):
    """Two stacked depthwise-separable convolution blocks."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        self.op = nn.Sequential(
            SepConvBlock(
                channels,
                channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
            ),
            SepConvBlock(
                channels,
                channels,
                kernel_size,
                stride=1,
                dilation=dilation,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class PoolOp(nn.Module):
    """NASNet 3x3 average or max pooling operation."""

    def __init__(
        self,
        kind: str,
        channels: int,
        stride: int,
    ) -> None:
        super().__init__()

        if kind == "avg":
            pool = nn.AvgPool2d(
                kernel_size=3,
                stride=stride,
                padding=1,
                count_include_pad=False,
            )
        elif kind == "max":
            pool = nn.MaxPool2d(
                kernel_size=3,
                stride=stride,
                padding=1,
            )
        else:
            raise ValueError(f"Unsupported pooling operation: {kind}")

        # The cell base has already projected every branch input to ``channels``.
        self.channels = channels
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class IdentityOp(nn.Module):
    """Identity for normal cells and strided projection for reduction cells."""

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()

        if stride == 1:
            self.op = nn.Identity()
        else:
            self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                make_bn(channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class AsymmetricConv(nn.Module):
    """Bottlenecked 1x7 then 7x1 asymmetric convolution operation."""

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()

        mid_channels = max(1, channels // 4)

        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                channels,
                mid_channels,
                kernel_size=1,
                bias=False,
            ),
            make_bn(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=(1, 7),
                stride=(1, stride),
                padding=(0, 3),
                bias=False,
            ),
            make_bn(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=(7, 1),
                stride=(stride, 1),
                padding=(3, 0),
                bias=False,
            ),
            make_bn(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                mid_channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            make_bn(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


def build_operation(name: str, channels: int, stride: int) -> nn.Module:
    """Build one operation from the canonical NASNet operation registry."""
    if name == "identity":
        return IdentityOp(channels, stride)

    if name == "sep_conv_3x3":
        return StackedSepConv(channels, 3, stride)

    if name == "sep_conv_5x5":
        return StackedSepConv(channels, 5, stride)

    if name == "sep_conv_7x7":
        return StackedSepConv(channels, 7, stride)

    if name == "dil_sep_conv_3x3":
        return StackedSepConv(
            channels,
            3,
            stride,
            dilation=2,
        )

    if name == "avg_pool_3x3":
        return PoolOp("avg", channels, stride)

    if name == "max_pool_3x3":
        return PoolOp("max", channels, stride)

    if name == "conv_1x7_7x1":
        return AsymmetricConv(channels, stride)

    raise ValueError(f"Unsupported NASNet operation: {name}")
