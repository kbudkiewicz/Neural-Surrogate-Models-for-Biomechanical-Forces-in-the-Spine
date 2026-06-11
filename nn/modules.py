from . import *
from typing import Union, Tuple


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


class BasicStem(nn.Sequential):
    def __init__(self, in_features: int = 3):
        super().__init__(
            nn.Conv3d(in_features, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


class ConvBlock3D(Module):
    """Base class for a convolutional block with an activation function and BatchNorm.

    .. note:: No bias is added for convolution layers followed by a BatchNorm. See `this blog`_ for more details.

    .. _`this blog`:
        https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-
        followed-by-a-batch-norm
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]],
        padding: Union[int, Tuple[int, int, int]],
        groups: int = 1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=groups, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Shapes:
            - Input: (B, C_in, H, W, D)
            - Output: (B, C_out, H, W, D)
        """
        return self.net(x)


class ResBlock3D(Module):
    """
    Args:
        channels (int): Number of channels
        depth (int): Number of consecutive convolutional blocks
        kernel_size (int): Kernel size
        stride (int): Stride
        padding (int): Padding
    """
    def __init__(self, channels: int, depth: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.net = nn.Sequential(*nn.ModuleList([
            ConvBlock3D(channels, channels, kernel_size, stride, padding) for _ in range(depth)
        ]))

    def forward(self, x: Tensor) -> Tensor:
        """Expects a batched 5D input of size (B, C, D, H, W)"""
        y = self.net(x)
        return y + x


class Bottleneck3D(Module):
    """A bottleneck module for 3D convolutional blocks inspired by 2D `Deep Residual Learning for Image Recognition
    <http://arxiv.org/abs/1512.03385>`__.

    .. note:: Also take a look at the `2D ResNet`_ implementation.

    .. _`2D ResNet`:
        https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L108
    """
    def __init__(self, in_channels: int, compression_factor: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.width = int(in_channels // compression_factor)
        self.bottleneck = nn.Sequential(
            ConvBlock3D(in_channels, self.width, kernel_size=1, stride=1, padding=0),
            ConvBlock3D(self.width, self.width, kernel_size=3, stride=1, padding=1),
            ConvBlock3D(self.width, in_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Shapes:
            - Input, Output: (B, C, D, H, W)
        """
        return self.bottleneck(x) + x


class DepthwiseSeparableConvolution3D(Module):
    """
    Depth-wise Separable Convolution based on `MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
    Applications <http://arxiv.org/abs/1704.04861>`__.

    The number of groups for the depth-wise convolutional layer is equal to the number of input channels.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): kernel size
        stride (int): stride
        padding (int): padding
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.depthwise = ConvBlock3D(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class InvertedResidual3D(Module):
    """Implementation of an Inverted Residual block from `MobileNetV2: Inverted Residuals and Linear Bottlenecks
    <http://arxiv.org/abs/1801.04381>`__ for 3D inputs.

    This implementation was inspired by the `following implementation`_.

    .. _`following implementation`:
        https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py#L19
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expansion_ratio: int = 1):
        super().__init__()
        mid_channels = expansion_ratio * in_channels
        if stride not in [1, 2]:
            raise ValueError(f'Stride should be 1 or 2, but is {stride}')
        self.stride = stride
        self.res_allowed: bool = in_channels == out_channels and stride == 1

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.dwise = DepthwiseSeparableConvolution3D(mid_channels, out_channels, stride)

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(x)
        y = self.dwise(y)
        if self.res_allowed:
            return y + x
        else:
            return y


class UpscaleBlock(Module):
    """
    Args:
        channels (int): number of channels
        kernel_size (int): kernel size
        stride (int): stride
        padding (int): padding
    """
    def __init__(self, channels, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            ConvBlock3D(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding),
            ConvBlock3D(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding),
            ConvBlock3D(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DownscaleBlock(Module):
    """
    Args:
        channels (int): number of channels
        kernel_size (int): kernel size
        stride (int): stride
        padding (int): padding
    """
    def __init__(self, channels, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock3D(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding),
            ConvBlock3D(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding),
            ConvBlock3D(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.MaxPool3d(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
