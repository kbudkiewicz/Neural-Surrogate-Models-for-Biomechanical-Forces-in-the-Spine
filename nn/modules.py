from . import *


class LinearBlock(Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, output_size),
            nn.GELU()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
