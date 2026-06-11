import os
import argparse
import pandas as pd

from torch.utils.data import DataLoader
from utils.preprocessing import get_splits
from utils.eval import relative_error, absolute_error, absolute_error_by_std, evaluate, boxplot_metric
from nn.networks import *
from data.datasets import *

SUBSET_CHOICES = ['train', 'test', 'val']
METRICS = absolute_error, relative_error, absolute_error_by_std

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-file', type=str)
    parser.add_argument('--eval-name', type=str, default='eval')
    parser.add_argument('-w', '--weights', type=str, default=None, help='Relative path to model state dict.')
    parser.add_argument('-S', '--subset', type=str, default='val', choices=SUBSET_CHOICES, nargs='?')
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-n', '--num-workers', type=int, default=16)
    parser.add_argument('-s', '--stack', type=str, nargs='*')
    args = parser.parse_args()

    device = args.device
    df = pd.read_csv(args.csv_file)
    npz_name = args.csv_file.replace('.csv', '.npz')
    # NOTE: because of order of SUBSET_CHOICES, selecting 'val' returns the test indices, and 'test' the
    # validation indices
    indices = {k: v for k, v in zip(SUBSET_CHOICES, get_splits(npz_name, df))}
    dataset = ShearComprDataset(df.iloc[indices[args.subset]])
    loader = DataLoader(dataset, batch_size=8, num_workers=args.num_workers, generator=torch.manual_seed(0))
    model = ResNet3DRegressor(out_features=dataset.dim)
    model.to(device)

    eval_path = os.path.join('../data', 'eval_' + model.__class__.__name__.lower(), dataset.name)
    eval_name = args.eval_name + '_' + args.subset + '.csv'

    if not os.path.isdir(eval_path):
        os.mkdir(eval_path)
    eval_file = os.path.join(eval_path, eval_name)
    if not os.path.exists(eval_file):
        evaluate(*METRICS, model=model, loader=loader, device=device, csv_filename=eval_file, weights=args.weights)
    else:
        print('Evaluation file "{eval_file}" already exists. Plotting...')
    for metric in METRICS:
        plot_name = metric.__name__ + '_' + args.subset
        boxplot_metric(metric, eval_file, stack=args.stack, title=dataset.name, plot_name=plot_name)
