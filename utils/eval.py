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


def absolute_error_by_std(pred: Tensor, target: Tensor, std) -> Tensor:
    return torch.abs(target - pred) / (std + 1e-8)


# def mean_absolute_deviation(pred: Tensor, target: Tensor) -> Tensor:
#     return torch.abs(pred - torch.mean(target)) / len(target)


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
    expected_shape = loader_length, batch_size, dim = len(loader), loader.batch_size, loader.dataset.dim
    std = loader.dataset.std
    std = std.to(device)
    loader = wrap_dataloader(loader, 'Evaluation')
    model.eval()

    d = {metric.__name__: np.empty(expected_shape) for metric in metrics}

    for i, batch in enumerate(loader):
        img, target, conditioning = batch
        img, target = img.to(device), target.to(device)
        if conditioning.isnan().any():
            preds = model(img)
        else:
            conditioning = conditioning.to(device)
            preds = model(img, conditioning)

        for metric in metrics:
            if metric.__name__ == 'absolute_error_by_std':
                value = metric(preds, target, std)
            else:
                value = metric(preds, target)
            if value.size() != torch.Size([*expected_shape[1:]]):
                delta = abs(batch_size - value.shape[0])
                value = torch.cat([value, torch.zeros([delta, dim], device=device)], dim=0)
            d[metric.__name__][i] = value.numpy(force=True)

    for metric in metrics:
        value = d[metric.__name__]
        d[metric.__name__] = remove_zeros(value.reshape(loader_length * batch_size, dim))

    return d


def make_col_names(metric: np.array, target_col: pd.Index, label: str) -> pd.Index:
    names = []
    for i in range(metric.shape[-1]):
        names.append(label + '-' + target_col[i])
    return pd.Index(names)


def save_as_csv(
    metrics: dict[str, np.ndarray],
    data_paths: pd.DataFrame,
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
    if len(data_paths) != len(df):
        data_paths = data_paths[:len(df)]
        df = pd.concat([data_paths, df], axis=1)
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
    data_paths = loader.dataset.df[loader.dataset.paths_key]
    metrics = calculate_metrics(*metrics, model=model, loader=loader, device=device)
    save_as_csv(metrics, data_paths=data_paths, target_cols=target_cols, csv_filename=csv_filename)


def make_regex(strings: Iterable[str]) -> str:
    regex = ''
    for string in strings:
        regex += f'(?=.*{string})'
    return regex


def print_mean_std(csv_filename: str, keywords: str, stack: bool = False):
    """
    Args:
        csv_filename: path to eval.csv
        keywords: regular expression containing keywords. Only columns containing all keywords will be printed.
        stack: If ``True``, check for columns where the keyword is present via regex. Then stack all those columns, and
            calculate a common mean and standard deviation.
    """
    if not os.path.isfile(csv_filename):
        raise FileNotFoundError(f'{csv_filename} cannot be found.')
    df = pd.read_csv(csv_filename)
    keywords = make_regex(keywords)
    columns_with_keyword = df.columns.str.contains(keywords, regex=True)
    if not columns_with_keyword.any():
        print(f'WARNING: {csv_filename} does not contain regex "{keywords}".')
        return False
    valid_cols = df.columns[columns_with_keyword].values
    df = df[valid_cols]

    if stack:
        df = df.stack()
        print(f'{keywords}: {df.median():16.2f} with {df.mean():.2f}+-{df.std():.2f}')
    else:
        for column_name in valid_cols:
            df_slice = df[column_name]
            print(f'{column_name}: {df_slice.median():16.2f} with {df_slice.mean():.2f}+-{df_slice.std():.2f}')


# --- Plotting ---
def plot_metric(
    metric: Callable,
    eval_file: str,
    stack: Optional[Iterable[str]] = None,
    title: Optional[str] = None,
    plot_name: Optional[str] = None
):
    df = pd.read_csv(eval_file)
    header = df.columns
    cols = header[header.str.contains(metric.__name__)]
    if cols.empty:
        raise KeyError(f'{eval_file} does not contain metric {metric.__name__}.')

    if isinstance(stack, Iterable):
        temp_df = pd.DataFrame([], columns=pd.Index(stack))
        for regex in stack:
            c = cols[cols.str.contains(regex, regex=True)]
            temp_df[regex] = df[c].values.ravel()
        df = temp_df
        if df.isna().any().any():
            raise ValueError(f'NaN values in {eval_file}.')
        tick_labels = stack
    else:
        df = df[cols]
        tick_labels = df.columns.str.replace(metric.__name__ + '-', '')

    plt.figure(figsize=(max(df.shape[-1] // 6, 4), 6))
    plt.boxplot(df.values, tick_labels=tick_labels, meanline=True, showmeans=True)
    plt.ylabel(metric.__name__)
    if metric.__name__ == 'relative_error':
        # lower, upper = max(0, df.min().min()), min(2, df.max().max())
        plt.ylim([0, 2])
    elif metric.__name__ == 'absolute_error':
        plt.yscale('log')
        plt.ylabel(f'Log {metric.__name__}')
    else:
        pass
    plt.xticks(rotation=90)
    if title is not None:
        plt.title('Dataset: ' + title)
    plt.tight_layout()

    if plot_name:
        plot_name = os.path.join(os.path.dirname(eval_file), plot_name)
        plt.savefig(plot_name)
    else:
        plt.show()


# if __name__ == '__main__':
#     plot_metric('abs_err', '../data/eval/eval.csv', 'test')
