import unittest
import torch
import torchvision.models.video as video_models

from nn.modules import *
from nn.networks import Chimera
from torchvision.models.video.resnet import Conv2Plus1D


class TestModules(unittest.TestCase):
    def setUp(self):
        self.batch_size = 5
        self.in_dim = 3
        self.out_dim = 12
        self.img_size = 32
        self.linblock = LinearBlock(self.in_dim, self.out_dim)
        self.convblock = ConvBlock3D(self.in_dim, self.out_dim, 3, 1, 1)
        self.resblock =  ResBlock3D(self.in_dim, 2, 3, 1, 1)
        self.upscaleblock = UpscaleBlock(self.in_dim)
        self.downscaleblock = DownscaleBlock(self.in_dim)
        self.x = torch.randn(self.batch_size, self.in_dim, self.img_size, self.img_size, self.img_size)  # (B, C, D, H, W)

    @staticmethod
    def count_parameters(module: nn.Module):
        return sum(p.numel() for p in module.parameters())

    def test_linblock(self):
        in_ = torch.randn(self.batch_size, self.in_dim)
        out = self.linblock(in_)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(in_.ndim, out.ndim)
        self.assertEqual(torch.Size([self.batch_size, self.out_dim]), out.shape)

    def test_convblock(self):
        out = self.convblock(self.x)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(5, out.ndim)
        self.assertEqual(torch.Size([self.batch_size, self.out_dim, *3 * [self.img_size]]), out.shape)

    def test_bottlneck3d(self):
        x = torch.empty([self.batch_size, 32, self.img_size, self.img_size, self.img_size])
        bottleneck = Bottleneck3D(32)
        out = bottleneck(x)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(x.shape, out.shape)

    def test_separable_depthwise(self):
        sdw_conv = DepthwiseSeparableConvolution3D(self.in_dim, self.out_dim, 1)
        out = sdw_conv(self.x)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(self.out_dim, out.ndim)

    def test_invertedresidual(self):
        invers = InvertedResidual3D(self.in_dim, self.in_dim, 1)
        out = invers(self.x)
        self.assertIsInstance(out, torch.Tensor)
        print(self.count_parameters(invers))

    def test_2plus1_conv(self):
        conv = Conv2Plus1D(self.in_dim, self.out_dim, 12)
        out = conv(self.x)
        self.assertIsInstance(out, torch.Tensor)
        print(self.count_parameters(conv))

    def test_resblock(self):
        out = self.resblock(self.x)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(5, out.ndim)
        self.assertEqual(torch.Size([self.batch_size, self.in_dim, *3 * [self.img_size]]), out.shape)

    def test_upscaleblock(self):
        out = self.upscaleblock(self.x)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(5, out.ndim)
        self.assertEqual(torch.Size([self.batch_size, self.in_dim, *3 * [self.img_size * 2]]), out.shape)

    def test_downscaleblock(self):
        out = self.downscaleblock(self.x)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(5, out.ndim)
        self.assertEqual(torch.Size([self.batch_size, self.in_dim, *3 * [self.img_size // 2]]), out.shape)

    def test_resnet3d(self):
        renset = video_models.r3d_18(pretrained=False)
        x = torch.empty([self.batch_size, 3, 128, 128, 128])
        out = renset(x)
        print(out.shape)

    def test_chimera(self):
        chimera = Chimera(128, 128, 10)
        conditioning = torch.randn(self.batch_size, 7)
        out = chimera(self.x, conditioning)
        self.assertEqual(out.shape, torch.Size([self.batch_size, chimera.out_dim]))


if __name__ == '__main__':
    unittest.main()
