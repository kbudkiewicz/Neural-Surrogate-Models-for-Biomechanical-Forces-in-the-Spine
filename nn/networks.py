import torchvision.models.video as video_models

from typing import Optional
from . import *
from .modules import LinearBlock, BasicStem
from torchvision.models.video.swin_transformer import SwinTransformer3d, PatchEmbed3d
from torchvision.models.video.resnet import Conv3DSimple, BasicBlock, VideoResNet


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


# VisionTransformer
class VideoSwinTransformer3DRegressor(Module):
    """Basic Video Swin Transformer (VST) regression network.

    .. note::
        The parameters of the small VST with approximately 49.8 million parameters are used.
    """
    def __init__(self, in_channels: int, out_features: int, model_config: Optional[str] = None):
        super().__init__()

        # Small VST as default
        configs = {
            'tiny': dict(
                patch_size=[2, 4, 4], embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                window_size=[8, 7, 7], stochastic_depth_prob=0.1
            ),
            'small': dict(
                patch_size=[2, 4, 4], embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24],
                window_size=[8, 7, 7], stochastic_depth_prob=0.1
            ),
            'big': dict(
                patch_size=[2, 4, 4], embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                window_size=[8, 7, 7], stochastic_depth_prob=0.1)
        }

        if isinstance(model_config, str) and model_config in configs.keys():
            config = configs[model_config]
        else:
            config = configs['small']

        self.backbone = SwinTransformer3d(**config)
        # Line 417 in swin_transformer.py
        self.backbone.patch_embed = PatchEmbed3d(
            in_channels=in_channels, patch_size=config['patch_size'], norm_layer=nn.LayerNorm
        )
        self.backbone.head = nn.Linear(self.backbone.num_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)
