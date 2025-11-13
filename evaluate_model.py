import os
import argparse
import pandas as pd

from torch.utils.data import DataLoader
from utils.eval import evaluate, plot_metric
from utils.preprocessing import get_splits
from utils.eval import relative_error, absolute_error
from nn.networks import ResNet3DRegressor, MultilayerPerceptron
from data.datasets import CTDataset, MuscleDataset, ShearComprDataset, AllForcesDataset, ForceToForceDataset

SUBSET_CHOICES = ['train', 'test', 'val']
METRICS = absolute_error, relative_error

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default='compr_shear_muscle2.csv')
    parser.add_argument('--eval_name', type=str, default='eval.csv')
    parser.add_argument('--weights', type=str, default=None, help='Relative path to model state dict.')
    parser.add_argument('--subset', type=str, default='val', choices=SUBSET_CHOICES, nargs='?')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    csv_path = './data/eval'
    npz_name = args.csv_file.replace('csv', 'npz')
    df = pd.read_csv(os.path.join('./data', args.csv_file))

    # select subset
    device = args.device
    indices = {k: v for k, v in zip(SUBSET_CHOICES, get_splits(npz_name, df))}
    dataset = ForceToForceDataset(df.iloc[indices[args.subset]])
    loader = DataLoader(dataset, batch_size=16)
    # model = ResNet3DRegressor(out_features=dataset.dim).to(device)
    model = MultilayerPerceptron(103, 512, 512, 256, dataset.dim).to(device)

    csv_path = os.path.join(csv_path, dataset.name)
    if not os.path.isdir(csv_path):
        os.mkdir(csv_path)
    eval_file = os.path.join(csv_path, args.eval_name)

    evaluate(*METRICS, model=model, loader=loader, device=device, csv_filename=eval_file, weights=args.weights)
    for metric in METRICS:
        plot_name = metric.__name__ + '_' + args.subset
        plot_metric(metric, eval_file, title=dataset.name, plot_name=plot_name)
