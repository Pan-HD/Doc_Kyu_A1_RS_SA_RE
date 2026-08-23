import torch
import torch.nn as nn

from src.search_space.architecture import Architecture


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def _pooling(name: str) -> nn.Module:
    if name == "max":
        return nn.MaxPool2d(kernel_size=2, stride=2)
    if name == "avg":
        return nn.AvgPool2d(kernel_size=2, stride=2)
    raise ValueError(f"Unsupported pooling: {name}")


class SearchCNN(nn.Module):
    def __init__(self, arch: Architecture, num_classes: int = 10):
        super().__init__()

        layers = []
        in_channels = 3
        out_channels = arch.initial_channels

        for block_idx in range(arch.num_conv_blocks):
            if block_idx > 0:
                out_channels *= arch.channel_multiplier

            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=arch.kernel_size,
                    padding=arch.kernel_size // 2,
                    bias=not arch.use_batchnorm,
                )
            )

            if arch.use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))

            layers.append(_activation(arch.activation))
            layers.append(_pooling(arch.pooling))

            if arch.dropout > 0:
                layers.append(nn.Dropout2d(p=arch.dropout))

            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_cnn(arch: Architecture, num_classes: int = 10) -> nn.Module:
    return SearchCNN(arch, num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
