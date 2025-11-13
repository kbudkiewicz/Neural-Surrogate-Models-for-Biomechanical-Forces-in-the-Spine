import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, Optional, Union, Iterable, Callable
from utils.preprocessing import wrap_dataloader
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader


def relative_error(pred: Tensor, target: Tensor) -> Tensor:
    return torch.abs(target - pred) / (torch.abs(target) + 1e-8)


def absolute_error(pred: Tensor, target: Tensor) -> Tensor:
    return torch.abs(target - pred)


# def mse(pred: Tensor, target: Tensor) -> Tensor:
#     return torch.square(target - pred) / (torch.square(target) + 1e-8)
#
#
# def mae(pred: Tensor, target: Tensor) -> Tensor:
#     return torch.abs(target - pred) / (torch.abs(target) + 1e-8)


def import_weights(model: Module, weights: str, device: Union[str, torch.device]):
    try:
        state_dict = torch.load(weights, map_location=device)
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise e
    except AttributeError or FileNotFoundError:
        print('No model was given or weight were not found. Proceeding with the provided model.')


def remove_zeros(x: np.array) -> np.array:
    return x[~np.all(x == 0, axis=1)]


@torch.no_grad()
def calculate_metrics(*metrics: Callable, model: Module, loader: DataLoader, device):
    expected_shape = l, bs, dim = len(loader), loader.batch_size, loader.dataset.dim
    loader = wrap_dataloader(loader, 'Evaluation')
    model.eval()

    d = {}
    for metric in metrics:
        d[metric.__name__] = np.empty(expected_shape)

    for i, batch in enumerate(loader):
        img, target = batch
        img, target = img.to(device), target.to(device)
        preds = model(img)

        for metric in metrics:
            value = metric(preds, target)
            if value.size() != torch.Size([*expected_shape[1:]]):
                delta = abs(bs - value.shape[0])
                value = torch.cat([value, torch.zeros([delta, dim], device=device)], dim=0)
            d[metric.__name__][i] = value.numpy(force=True)

    for metric in metrics:
        value = d[metric.__name__]
        d[metric.__name__] = remove_zeros(value.reshape(l * bs, dim))

    return d


def make_col_names(metric: np.array, target_col: pd.Index, label: str) -> pd.Index:
    names = []
    for i in range(metric.shape[-1]):
        names.append(label + '-' + target_col[i])
    return pd.Index(names)


def save_as_csv(
    metrics: dict[str, np.ndarray],
    nifti_paths: pd.DataFrame,
    target_cols: pd.Index,
    csv_filename: str,
) -> None:
    df = pd.DataFrame([])

    for metric, values in metrics.items():
        columns = make_col_names(values, target_cols, metric)
        dff = pd.DataFrame(values, columns=columns)
        if df.empty:
            df = dff
        else:
            df = df.join(dff)
    if len(nifti_paths) != len(df):
        nifti_paths = nifti_paths[:len(df)]
        df = pd.concat([nifti_paths, df], axis=1)
    df.index.name = 'id'

    try:
        df.to_csv(csv_filename, index=True)
    except OSError as e:
        print(e)
        csv_filename = 'eval.csv'
        df.to_csv(csv_filename, index=True)


def evaluate(
    *metrics: Callable,
    model: Module,
    loader: DataLoader,
    device: Union[str, torch.device],
    csv_filename: str,
    weights: Optional[str] = None
):
    """
    Args:
        csv_filename (str): CSV to which the evaluated metrics are saved into. Must be a viable relative path.
    """
    if weights:
        import_weights(model, weights, device=device)
    target_cols = loader.dataset.target_cols
    nifti_paths = loader.dataset.df['nifti_path']
    metrics = calculate_metrics(*metrics, model=model, loader=loader, device=device)
    save_as_csv(metrics, nifti_paths=nifti_paths, target_cols=target_cols, csv_filename=csv_filename)


# --- Plotting ---
def plot_metric(metric: Callable, eval_file: str, title: Optional[str] = None, plot_name: Optional[str] = None):
    df = pd.read_csv(eval_file)
    header = df.columns
    cols = header[header.str.contains(metric.__name__)]
    df = df[cols]
    tick_labels = df.columns.str.replace(metric.__name__ + '-', '')

    plt.figure(figsize=(df.shape[-1] // 6, 6))
    plt.boxplot(df.values, tick_labels=tick_labels)
    plt.ylabel(metric.__name__)
    if 'rel' in metric.__name__:
        # lower, upper = max(0, df.min().min()), min(2, df.max().max())
        plt.ylim([0, 0.4])
    if 'abs' in metric.__name__:
        plt.yscale('log')
        plt.ylabel(f'Log {metric.__name__}')
    plt.xticks(rotation=90)
    plt.title('Dataset: ' + title)
    plt.tight_layout()

    if plot_name:
        plot_name = os.path.join(os.path.dirname(eval_file), plot_name)
        plt.savefig(plot_name)
    else:
        plt.show()


# if __name__ == '__main__':
#     plot_metric('abs_err', '../data/eval/eval.csv', 'test')
