import unittest
import torch
import torch.nn.functional as F

from torch.nn import Module, MSELoss
from torch.utils.data import DataLoader
from datasets import *
from nn.networks import ResNet3DRegressor, Chimera, VideoSwinTransformer3DRegressor  # MobileNet3D
from nn.losses import ScaledLoss


class TestNetworks(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.num_workers = 1
        self.n_batches = 2  # number of batches to test
        self.dataset = ShearComprDataset('../data/csv/conditional_half.csv')
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                     num_workers=self.num_workers)
        # self.mobilenet = MobileNet3D(self.dataset.dim)
        self.expected_shape = torch.Size([self.batch_size, self.dataset.dim])
        print('Setup complete.')

    def _test_batch(self, model: Module, batch: Tuple[Tensor, ...]):
        img, target, condition = batch
        if torch.isnan(condition).any() is True:
            out = model(img)
        else:
            out = model(img, condition)
        self.assertIsInstance(out, Tensor)
        self.assertEqual(out.shape, self.expected_shape)

    def test_nako_resnet3d(self):
        net = ResNet3DRegressor(in_features=3, out_features=self.dataset.dim)
        for batch, _ in zip(self.dataloader, range(self.n_batches)):
            self._test_batch(net, batch)

    def test_nako_chimera(self):
        net = Chimera(in_features=3, out_features=1, conditioning_dim=5)
        cond = torch.randn(self.batch_size, 5)
        x = torch.randn(self.batch_size, 3, 32, 32, 32)
        out = net.forward(x, cond)
        self.assertEqual(out.shape, torch.Size([self.batch_size, 1]))

    def test_swintransformer3d(self):
        net = VideoSwinTransformer3DRegressor(out_features=self.dataset.dim, in_channels=3)
        for batch, _ in zip(self.dataloader, range(self.n_batches)):
            self._test_batch(net, batch)

    @unittest.expectedFailure
    def test_mobilenet(self):
        for batch, _ in zip(self.dataloader, range(self.n_batches)):
            self._test_batch(self.mobilenet, batch)

    def test_scaled_loss(self):
        x, y = torch.zeros(self.batch_size, self.dataset.dim), torch.zeros(self.batch_size, self.dataset.dim)
        criterion = ScaledLoss(MSELoss, std=1.)
        loss = criterion(x, y)
        self.assertIsInstance(loss, Tensor)
        self.assertEqual(loss.size(), torch.Size([]))
        if torch.equal(x, y):
            self.assertEqual(loss, 0)


if __name__ == '__main__':
    unittest.main()
