import unittest
import torch

from typing import Tuple
from torch import Tensor
from torch.utils.data import DataLoader
from data.datasets import *


class TestDatasets(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_workers = 1
        self.rand_idx = torch.randint(0, 10, (1,)).item()
        self.df = '../data/csv/conditional.csv'
        self.df_nako = '../data/csv/nako_osim.csv'
        self.df_osim = '../data/csv/nako_osim_data.csv'

    def _test_batch(self, batch: Tuple[Tensor, ...], loader: DataLoader):
        img, target, cond = batch
        self.assertIsInstance(img, Tensor)
        self.assertIsInstance(target, Tensor)
        if cond.isnan().any():
            self.assertTrue(cond.isnan().all())
        else:
            self.assertEqual(cond.shape[-1], 7)  # 6 angles + weight
        self.assertEqual(target.size(), torch.Size([self.batch_size, loader.dataset.dim]))

    def test_slice(self):
        # ds = BaseDataset(self.df, 'compr')
        ds = NakoImagesDataset('../data/csv/nako_zeroed_.csv')
        std = ds.std
        print(ds.get_targets('compr'))
        print(ds.df[ds.cond_cols].values)

    def test_twoforces(self):
        twoforces = DataLoader(ShearComprDataset(df=self.df), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.assertEqual(twoforces.dataset.dim, 18)
        for batch, _ in zip(twoforces, range(10)):
            self._test_batch(batch, twoforces)

    def test_muscleforces(self):
        muscles = DataLoader(MuscleDataset(df=self.df), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.assertEqual(muscles.dataset.dim, 103)
        for batch, _ in zip(muscles, range(10)):
            self._test_batch(batch, muscles)

    def test_all_forces(self):
        allforces = DataLoader(AllForcesDataset(df=self.df), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.assertEqual(allforces.dataset.dim, 121)
        for batch, _ in zip(allforces, range(10)):
            self._test_batch(batch, allforces)

    def test_f2f(self):
        f2f = DataLoader(ForceToForceDataset(df=self.df), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.assertEqual(f2f.dataset.dim, 18)
        for batch, _ in zip(f2f, range(10)):
            self._test_batch(batch, f2f)

    def test_nako(self):
        dataset = NakoDataset(df=self.df_nako)
        nako = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        for batch, _ in zip(nako, range(10)):
            self._test_batch(batch, nako)

    def test_nako_w_osim(self):
        dataset = NakoImagesDataset(df=self.df_nako, osim=self.df_osim)
        self.assertEqual(len(dataset.target_cols), 36, 'Osim and Nako targets are not equal')
        t = dataset.osim.columns[3:]
        idx = slice(2, 4)
        data = dataset.osim.iloc[idx][t].values
        _ = torch.tensor(data, dtype=torch.float)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        for batch, _ in zip(loader, range(10)):
            self.assertEqual(len(batch), 3)
            self._test_batch(batch, loader)


if __name__ == '__main__':
    unittest.main()
