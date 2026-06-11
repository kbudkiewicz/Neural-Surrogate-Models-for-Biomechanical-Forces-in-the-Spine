import unittest
import torch
import numpy as np
import pandas as pd

from utils.eval import absolute_error, relative_error, rmse, nrmse, save_as_csv, calculate_metrics, get_splits
from utils.const import _SHEARCOMPR_D
from nn.networks import MultilayerPerceptron
from data.datasets import ForceToForceDataset
from torch.utils.data import DataLoader


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = ForceToForceDataset('../data/csv/Optim_0.csv')
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=256, shuffle=True, num_workers=1, drop_last=False)
        self.net = MultilayerPerceptron(103, 2, 18)
        self.net.to(self.device)
        self.METRICS = absolute_error, relative_error

    def test_rmse(self):
        test_set = pd.read_csv('../data/csv/split-Optim_0.csv')[_SHEARCOMPR_D.keys()]
        train_set = self.dataset.df[_SHEARCOMPR_D.keys()]
        self.assertEqual(len(train_set.columns), len(test_set.columns))
        self.assertEqual(round(len(train_set) * 0.1), len(test_set))   # assumes 8:1:1 data train-val-test split
        rmse_values = rmse(train_set)
        nrmse_values = nrmse(train_set, test_set)
        self.assertIsInstance(rmse_values, pd.Series, 'RMSE is not a Series')
        self.assertIsInstance(nrmse_values, pd.Series, 'nRMSE is not a Series')
        self.assertTrue(np.array_equal(nrmse_values, rmse_values / test_set.std().values))

    def test_evaluate(self):
        target_cols = self.dataloader.dataset.target_cols
        data_paths = self.dataloader.dataset.df[self.dataloader.dataset.paths_key]
        metrics = calculate_metrics(
            *self.METRICS, model=self.net, loader=self.dataloader, device=self.device
        )
        save_as_csv(metrics, data_paths=data_paths, target_cols=target_cols, csv_filename='test_evaluate.csv')


if __name__ == '__main__':
    unittest.main()
