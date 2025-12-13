import os
import argparse
import pandas as pd

from torch import load
from torch.utils.data import DataLoader
from utils.preprocessing import get_splits
from utils.eval import relative_error, absolute_error, evaluate, plot_metric
from nn.networks import ResNet3DRegressor, ResNet2Plus1DRegressor, MultilayerPerceptron, Chimera
from data.datasets import ComprDataset, MuscleDataset, ShearComprDataset, AllForcesDataset, ForceToForceDataset

SUBSET_CHOICES = ['train', 'test', 'val']
METRICS = absolute_error, relative_error

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-file', type=str)
    parser.add_argument('--eval-name', type=str, default='eval')
    parser.add_argument('-w', '--weights', type=str, default=None, help='Relative path to model state dict.')
    parser.add_argument('-s', '--subset', type=str, default='val', choices=SUBSET_CHOICES, nargs='?')
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-n', '--num-workers', type=int, default=16)
    args = parser.parse_args()

    device = args.device
    df = pd.read_csv(args.csv_file)
    npz_name = args.csv_file.replace('csv', 'npz')
    indices = {k: v for k, v in zip(SUBSET_CHOICES, get_splits(npz_name, df))}
    dataset = ShearComprDataset(df.iloc[indices[args.subset]])
    # dataset = ShearComprDataset(df) #
    # dataset._mask_df('task', 'Optim_0', True) #
    loader = DataLoader(dataset, batch_size=8, num_workers=args.num_workers)
    # model = ResNet3DRegressor(out_features=dataset.dim)
    model = ResNet2Plus1DRegressor(out_features=dataset.dim)
    # model = MultilayerPerceptron(103, 512, 512, 256, dataset.dim)
    # model = Chimera(512, 128, dataset.dim)
    model.to(device)

    # resnet = ResNet3DRegressor(out_features=103)
    # mlp = MultilayerPerceptron(103, 512, 512, 256, dataset.dim)
    # resnet.load_state_dict(torch.load('muscle.pth'))
    # mlp.load_state_dict(torch.load('forcetoforce_5.pth'))
    # model = Chimera(resnet=resnet, mlp=mlp).to(device)

    eval_path = os.path.join('./data', 'eval_' + model.__class__.__name__.lower(), dataset.name)
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
        plot_metric(metric, eval_file, title=dataset.name, plot_name=plot_name)
