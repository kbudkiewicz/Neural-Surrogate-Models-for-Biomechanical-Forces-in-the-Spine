import torchvision.models.video as video_models
from torchvision.models.video.resnet import Conv3DSimple, BasicBlock, VideoResNet
from . import *
from typing import Optional
from .modules import LinearBlock


class MultilayerPerceptron(Module):
    def __init__(self, *dims):
        super().__init__()
        modules = nn.ModuleList()
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            modules.append(LinearBlock(in_dim, out_dim))
        modules.append(nn.Linear(dims[-1], dims[-1]))
        self.net = nn.Sequential(*modules)

    def forward(self, x: Tensor, conditioning_embedding: Optional[Tensor] = None) -> Tensor:
        if conditioning_embedding is not None:
            x += conditioning_embedding
        return self.net(x)


class BasicStem(nn.Sequential):
    def __init__(self, in_features: int = 3):
        super().__init__(
            nn.Conv3d(in_features, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


class ResNet3DRegressor(nn.Module):
    """Basic Res3DNet regression network.

    .. note::
        basic_stem has to be included to enable inputs with different number of input channels.
        Otherwise, inputs with 3 channels only are accepted by the original implementation.
    """
    def __init__(self, out_features: int, in_features: int = 3):
        super().__init__()
        self.backbone = video_models.r3d_18(weights=None)
        self.backbone.stem = BasicStem(in_features=in_features)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)


class Chimera(Module):
    """Model combining already trained Res3DNet and a MultilayerPerceptron. The former predicts the muscle forces from
    an MRI image, while the former predicts the shear and compression forces based on those."""
    def __init__(self, conditioning_dim: int, out_features: int, in_features: int = 3):
        super().__init__()
        self.backbone = VideoResNet(
            BasicBlock,
            [Conv3DSimple] * 4,
            [2, 2, 2, 2],
            BasicStem
        )
        self.backbone.stem = BasicStem(in_features=in_features)

        self.embd_conditioning = nn.Sequential(
            nn.Linear(conditioning_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, self.backbone.fc.in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, out_features),
        )
        self.backbone.fc = nn.Identity()

    def forward(self, x: Tensor, conditioning: Optional[Tensor] = None) -> Tensor:
        if conditioning is not None:
            conditioning = self.embd_conditioning(conditioning)
        x = self.backbone(x)
        x = self.mlp(x + conditioning)
        return x
