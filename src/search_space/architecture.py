from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Architecture:
    num_conv_blocks: int
    initial_channels: int
    channel_multiplier: int
    kernel_size: int
    dropout: float
    use_batchnorm: bool
    activation: str
    pooling: str

    def to_dict(self):
        return asdict(self)
