import os
import matplotlib
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.const import (_COMPRESSION_D, _SHEAR_D, _SHEARCOMPR_D, _MUSCLES_D, _MUSCLES_LATIN_D, _PRELIMINARIES_D,
                         _COORDS_D, _FORCES_D, _MOMENTS_D, _COORDS_ALL_D, sort_for_plot)
from typing import Tuple, Optional, Union, Iterable, Callable
from utils.preprocessing import wrap_dataloader, get_splits
from data.utilities import rename_if_exists
from scipy.stats import normaltest, shapiro
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader


NIFTI_DROP = ['id', 'task', 'nifti_path']
NAKO_DROP = ['id', 'nako_path']


def relative_error(pred: Tensor, target: Tensor) -> Tensor:
    return torch.abs(target - pred) / (torch.abs(target) + 1e-8)


def absolute_error(pred: Tensor, target: Tensor) -> Tensor:
    return torch.abs(target - pred)


def absolute_error_by_std(pred: Tensor, target: Tensor, std) -> Tensor:
    return torch.abs(target - pred) / (std + 1e-8)


def mse(pred: Tensor, target: Tensor) -> Tensor:
    return torch.square(target - pred) / (torch.square(target) + 1e-8)


def rmse(df: pd.DataFrame) -> pd.Series:
    return np.sqrt(df.pow(2).mean())


def nrmse(df: pd.DataFrame, dataset: pd.DataFrame, stack: Optional[dict] = None) -> pd.Series:
    dataset = drop_df_columns(dataset)
    if stack is not None:
        dataset = stack_df_columns(dataset, stack=stack)
        df = stack_df_columns(df, stack=stack)
    dataset = dataset[df.columns.str.replace('absolute_error-', '')]
    normalized_rmse = rmse(df).values / dataset.std().values
    return pd.Series(normalized_rmse, index=df.columns)


def import_weights(model: Module, weights: str, device: Union[str, torch.device]):
    r"""Try to import model weights to a model. Otherwise, evaluate with an untrained model.

    Args:
        model (Module): Model to import.
        weights (str): Absolute path to model state dictionary.
        device (Union[str, torch.device]): Device on which to perform the evaluation.
    """
    try:
        state_dict = torch.load(weights, map_location=device)
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise e
    except AttributeError or FileNotFoundError:
        print('No model was given or weight were not found. Proceeding with the provided model.')


@torch.no_grad()
def calculate_metrics(*metrics: Callable, model: Module, loader: DataLoader, device):
    std = loader.dataset.std
    std = std.to(device)
    expected_shape = (len(loader.dataset), loader.dataset.dim)
    loader = wrap_dataloader(loader, 'Evaluation')
    model.eval()

    d = {metric.__name__: np.empty(expected_shape) for metric in metrics}
    d['pred'] = np.empty(expected_shape)

    i = 0
    for batch in loader:
        img, target, conditioning = batch
        img, target = img.to(device), target.to(device)
        batch_len = img.size(0)
        window = slice(i, i + batch_len)

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
            d[metric.__name__][window] = value.numpy(force=True)
        d['pred'][window] = preds.numpy(force=True)
        i += batch_len

    return d


def make_col_names(metric: np.array, target_col: pd.Index, label: str) -> pd.Index:
    names = [label + '-' + target_col[i] for i in range(metric.shape[-1])]
    return pd.Index(names)


def make_regex(strings: Iterable[str]) -> str:
    regex = ''
    for string in strings:
        regex += f'(?=.*{string})'
    return regex


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
        csv_filename = rename_if_exists(csv_filename)
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

    def print_values(x: pd.DataFrame, col_name: str) -> None:
        print(f'{col_name}: {x.median():16.2f} with {x.mean():.2f}+-{x.std():.2f}')

    if stack:
        df = df.stack()
        print_values(df, keywords)
    else:
        for column_name in valid_cols:
            df_slice = df[column_name]
            print_values(df_slice, column_name)


def drop_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    if any('nako_' in c for c in df.columns):
        df = df.drop(columns=NAKO_DROP)
    elif any('nifti_' in c for c in df.columns):
        df = df.drop(columns=NIFTI_DROP)
    else:
        raise ValueError('NRMSE: Unknown type of dataset provided.')
    return df


def stack_df_columns(df: pd.DataFrame, stack: Iterable[str]) -> pd.DataFrame:
    """Stack all columns containing the same string or regex in stack to a single column."""
    stacked_series = []
    for string in stack:
        cols = df.columns[df.columns.str.contains(string)]
        df_copy = df[cols]
        if len(cols) > 1:
            melt = df_copy.melt(value_name=string)
            values = melt[string].reset_index(drop=True)
        else:
            values = df_copy
        stacked_series.append(values)

    return pd.concat(stacked_series, axis=1)


def filter_l6(df: pd.DataFrame, include: bool = True) -> Tuple[pd.DataFrame, pd.Index]:
    temp = df.filter(regex='L6')
    temp = temp[df == 0].all(axis=1)
    indices = temp[temp == include].index
    return df.iloc[indices], indices


def eval_to_latex(
    *metrics: Union[Callable, str],
    path: str,
    stack: Union[Iterable, str] = None,
    **kwargs
) -> str:
    df = pd.read_csv(path)
    summary = pd.DataFrame()
    original_columns = df.columns[df.columns.str.contains('absolute_error-')].str.replace('absolute_error-', '')

    def create_subframe(x: pd.DataFrame, metric: Union[Callable, str]) -> pd.DataFrame:
        if metric == 'rmse':
            index = pd.MultiIndex.from_product(
                [[metric.upper() + ' $\\downarrow$'], [' ']], names=['Force', ' ']
            )
            x = pd.DataFrame([rmse(x).values], index=index)
        else:
            index = pd.MultiIndex.from_product(
                [[metric.__name__.replace('_', ' ').capitalize() + ' $\\downarrow$'],
                 ['Median', 'Mean', 'Std']], names=['Force', ' ']
            )
            median = x.median(axis=0).values
            mean = x.mean(axis=0).values
            std = x.std(axis=0).values
            x = pd.DataFrame([median, mean, std], index=index)
        return x

    for metric in metrics:
        regex = metric.__name__ + '-' if isinstance(metric, Callable) else 'absolute_error-'
        x = df.filter(regex=regex)
        if stack is not None:
            x = stack_df_columns(x, stack=stack)
        x = create_subframe(x, metric)
        summary = pd.concat([summary, x])

    if stack is not None:
        if len(stack.values()) != len(summary.columns):
            summary.columns = pd.Index([v for v, col in zip(stack.values(), original_columns) if col in stack.keys()])
        else:
            summary.columns = stack.values()
    else:
        summary.columns = original_columns.str.replace('_', ' ')

    # filter high values
    summary = summary.where(summary.values < 1e3, '-')

    # Add styling
    summary = summary.T
    styler = summary.style.map_index(lambda x: 'font-weight: bold;', axis='columns')    # columns in bold
    styler.format(precision=2)                                                          # float precision
    column_format = 'l' + 'c' * len(summary.columns)
    return styler.to_latex(position='h', column_format=column_format, position_float='centering',
                           multicol_align='c', hrules=True, convert_css=True, **kwargs)


def normality_test(
    path: str,
    buf: Optional[str] = None,
    stack: Optional[dict] = None,
):
    df = pd.read_csv(path)
    columns = df.columns
    if stack is not None:
        df = stack_df_columns(df, stack)
        columns = pd.Index(stack.values())
    stat, p = normaltest(df, nan_policy='omit')  # omnibus formula
    m = df.apply(shapiro, axis=0, nan_policy='omit')
    m.columns = columns

    index_dagostino = pd.MultiIndex.from_product(
        [['D\'Agostino omnibus'], ['$K^2 \\times 10^3$', '$p$-value $\\times 10^{-6}$']], names=['Force', '']
    )
    index_shapiro = pd.MultiIndex.from_product(
        [['Shapiro-Wilk'], ['$W$', '$p$-value $\\times 10^{-6}$']], names=['Force', '']
    )
    dagostino_values = pd.DataFrame([stat / 1e3, p * 1e6], index=index_dagostino, columns=columns)
    shapiro_values = pd.DataFrame([m.iloc[0], m.iloc[1] * 1e6], index=index_shapiro)
    results = pd.concat([dagostino_values, shapiro_values])

    # Add styling
    results = results.T
    styler = results.style.map_index(lambda x: 'font-weight: bold;', axis='columns')  # columns in bold
    styler.format(precision=2)  # float precision
    column_format = 'l' + 'c' * len(results.columns)
    return styler.to_latex(position='h', column_format=column_format, position_float='centering',
                           multicol_align='c', hrules=True, convert_css=True, buf=buf)


def csv_to_latex(file: str, buf: str, is_nako: bool):
    df = pd.read_csv(file)
    df = drop_df_columns(df)

    # define key for sorting
    if is_nako:
        key = lambda col: col.lower().split('_')[-1][0] + col.lower().split('_')[0][1]
        columns = _COORDS_ALL_D
        df = df[sorted(columns.keys(), key=key)]
    else:
        columns = _PRELIMINARIES_D
        df = stack_df_columns(df, _PRELIMINARIES_D)
        df = df[columns.keys()]

    median = df.median(axis=0).values
    mean = df.mean(axis=0).values
    std = df.std(axis=0).values

    index = pd.Series(['Median', 'Mean', 'Std'])
    table = pd.DataFrame([median, mean, std], index=index, columns=columns.values()).T

    # Save the table to LaTeX and add styling to it
    table = table.style.map_index(lambda x: 'font-weight: bold;', axis='columns')  # columns in bold
    table.format(precision=2)  # float precision
    column_format = 'l' + 'c' * len(table.columns)
    table.to_latex(position='h', column_format=column_format, position_float='centering', hrules=True,
                   convert_css=True, buf=buf)


# --- Plotting ---
def stack_df_for_boxplot(df: pd.DataFrame, stack: Iterable[str]) -> pd.DataFrame:
    """
    Melts columns that contain any of the substrings in 'stack' into a long format.
    Adds a 'stack_group' column indicating which substring matched.
    """
    dfs = []
    for s in stack:
        # Filter columns containing the current stack string (e.g., 'coord_0')
        cols = [c for c in df.columns if s in c]
        if not cols:
            continue

        # Melt these columns into a single 'value' column
        subset = df[cols].melt(value_name='value')
        subset['stack_group'] = s
        dfs.append(subset)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def remove_outliers(df: pd.DataFrame, threshold: float = 1e2) -> pd.DataFrame:
    mask = df.values <= threshold
    return df[mask.all(axis=1)]


def set_ylabel(ax, metric: Callable):
    ylabel = metric.__name__.replace('_', ' ').capitalize()
    if ax.get_yscale() == 'log':
        ylabel = 'Log ' + ylabel.lower()
    ax.set_ylabel(ylabel)


def set_boxplot_bounds(metric: Callable, bottom: float = 0., top: float = 3.):
    if metric.__name__ == 'relative_error':
        plt.ylim([0, 5])
    elif metric.__name__ == 'absolute_error':
        plt.yscale('log')
    else:
        plt.ylim(bottom=bottom, top=top)


def plot_metric(
    metric: Callable,
    eval_file: str,
    stack: Optional[Iterable[str]] = None,
    title: Optional[str] = None,
    plot_name: Optional[str] = None
) -> None:
    """
    Plot previously calculated metrics from an evaluation file.

    Args:
        metric (Callable): Metric function to be evaluated. Should take two Tensors and return one.
        eval_file (str): Relative path to the evaluation file in csv format.
        stack (Iterable[str], optional): Iterable of column names to concatenate. If provided, the function will stack
            all values from columns containing a substring into a single column and calculate the metric over it.
        title (str, optional): Title of the plot.
        plot_name (str, optional): Name of the plot to be saved.
    """
    matplotlib.rcParams['boxplot.meanprops.color'] = 'k'
    matplotlib.rcParams['boxplot.meanprops.linestyle'] = '-.'
    df = pd.read_csv(eval_file)
    header = df.columns
    cols = header[header.str.contains(metric.__name__ + '-')]
    if cols.empty:
        raise KeyError(f'{eval_file} does not contain metric {metric.__name__}.')
    df = df[cols]

    # get the desired values from (stacked) columns
    if isinstance(stack, Iterable):
        df = stack_df_columns(df, stack)
        if df.isna().any().any():
            raise ValueError(f'NaN values in {eval_file}.')
        tick_labels = stack
    else:
        tick_labels = df.columns.str.replace(metric.__name__ + '-', '')

    plt.figure(figsize=(max(df.shape[-1] // 6, 4), 6))
    plt.boxplot(df.values, tick_labels=tick_labels, meanline=True, showmeans=True)
    plt.ylabel(metric.__name__.replace('_', ' ').capitalize())
    set_boxplot_bounds(metric=metric)
    set_ylabel(ax, metric=metric)
    plt.xticks(rotation=90)
    if title is not None:
        plt.title('Dataset: ' + title)
    plt.tight_layout()

    if plot_name:
        plot_name = os.path.join(os.path.dirname(eval_file), plot_name)
        plot_name = rename_if_exists(plot_name)
        plt.savefig(plot_name)
    else:
        plt.show()


def plot_compare_models_bar(
    *eval_files: Tuple[str, str],
    metric: Callable,
    width: float = 0.3,
    stack: Optional[Iterable[str]] = None,
    plot_name: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    factor = 0

    for file, label in eval_files:
        offset = width * factor
        df = pd.read_csv(file).drop(columns='id')
        df = df.filter(regex=metric.__name__ + '-')
        # df = remove_outliers(df)
        if isinstance(stack, Iterable):
            df = stack_df_columns(df, stack)
        x = np.arange(len(df.columns))
        b = ax.bar(x=x + offset, height=df.mean().round(2), width=width, label=label,
                   yerr=df.std(), ecolor='k', capsize=3)    # error bar kwargs
        ax.bar_label(b)
        factor += 1

    ax.set_xticks(x + width, df.columns, rotation=90)
    plt.ylabel('Mean ' + metric.__name__)
    plt.ylim(bottom=0)
    plt.legend(loc='best')  # ncols=len(eval_files))
    plt.tight_layout()

    if plot_name:
        plot_name = rename_if_exists(plot_name)
        plt.savefig(plot_name)
    else:
        plt.show()


def get_slices(stack: dict) -> tuple[slice, slice]:
    if stack == _COORDS_D:
        return slice(3), slice(3, len(_COORDS_D))
    elif stack == _COORDS_ALL_D or _PRELIMINARIES_D:
        return slice(18), slice(18, len(_PRELIMINARIES_D))
    else:
        raise ValueError('Unsupported stack type.')


def barplot_metric(
    *eval_files: Tuple[str, str],
    metric: Union[Callable, str],
    width: float = 0.25,
    figsize: tuple = (8, 5),
    remove: bool = False,
    log: bool = False,
    stack: Optional[dict] = None,
    dataset: Optional[pd.DataFrame] = None,
    plot_name: Optional[str] = None,
    **plt_kwargs
):
    def get_ylabels(metric, log: bool) -> Iterable[str]:
        if metric in ['rmse', 'nrmse']:
            l_ylabel, r_ylabel = 'RMSE [N]', 'RMSE [Nm]'
        else:
            l_ylabel = r_ylabel = metric.__name__.replace('_', ' ').capitalize() + ' [-]'
        if log:
            l_ylabel = 'Log ' + l_ylabel
            r_ylabel = 'Log ' + r_ylabel
        return l_ylabel, r_ylabel

    if stack == _COORDS_D:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax, ax2 = axes
    elif stack == _PRELIMINARIES_D or stack == _COORDS_ALL_D:
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        ax, ax2 = axes
    else:
        fig, ax = plt.subplots(figsize=figsize)
    factor = 0
    offset_ticks = (len(eval_files) * width - width) / 2

    for file, label in eval_files:
        offset = width * factor - offset_ticks
        df = pd.read_csv(file).drop(columns='id')
        regex = 'absolute_error-' if metric in ['rmse', 'nrmse'] else metric.__name__ + '-'
        df = df.filter(regex=regex)
        if remove:
            df = remove_outliers(df)
        if isinstance(stack, Iterable):
            df = stack_df_columns(df, stack)
        x = np.arange(len(df.columns)) * ((len(eval_files) + 1) * width)

        if metric == 'rmse':
            values = rmse(df)
        elif metric == 'nrmse':
            if dataset is None:
                raise ValueError('NRMSE: No reference test or validation set is given')
            values = nrmse(df, dataset, stack=stack)
        else:
            df.mean(axis=0)

        if stack == _COORDS_D or stack == _COORDS_ALL_D or stack == _PRELIMINARIES_D:
            left, right = get_slices(stack)
            torques, forces = values[left], values[right]
            ax.bar(x[left] + offset, torques, label=label, log=log, width=width, **plt_kwargs)
            ax2.bar(x[right] + offset, forces, label=label, log=log, width=width, **plt_kwargs)
        else:
            ax.bar(x + offset, values, label=label, log=log, width=width, **plt_kwargs)
        factor += 1

    if stack == _COORDS_D or stack == _COORDS_ALL_D or stack == _PRELIMINARIES_D:
        xlabels = list(stack.values())
        l_ylabel, r_ylabel = get_ylabels(metric, log=log, dataset=dataset)
        ax.set_xticks(x[left], xlabels[left], rotation=90)
        ax2.set_xticks(x[right], xlabels[right], rotation=90)
        ax.set_ylabel(l_ylabel)
        ax2.set_ylabel(r_ylabel)
        ax.grid(axis='y', linewidth=0.5, linestyle='--')
        ax2.grid(axis='y', linewidth=0.5, linestyle='--')
        if stack == _COORDS_D:
            legend_kwargs = {'loc': 'center right', 'bbox_to_anchor': (1.5, 0.5)}
            ax2.legend(**legend_kwargs)
        else:
            legend_kwargs = {'loc': 'upper center', 'ncols': len(eval_files), 'bbox_to_anchor': (0.5, 1.3)}
            ax.legend(**legend_kwargs)
    else:
        xlabels = list(stack.values()) if isinstance(stack, dict) else df.columns.str.replace('absolute_error-', '')
        _, ylabel = get_ylabels(metric, log=log)
        ax.set_xticks(x, xlabels, rotation=90)
        ax.grid(axis='y', linewidth=0.5, linestyle='--')
        plt.ylabel(ylabel)
        plt.legend(bbox_to_anchor=(1.1, 0.6))
    plt.tight_layout()

    if plot_name:
        plot_name = rename_if_exists(plot_name)
        plt.savefig(plot_name, format='pdf')
    else:
        plt.show()


def plot_correlation(
    path: str,
    figsize: tuple = (15, 12),
    use_mask: bool = True,
    stack: Optional[Iterable[str]] = None,
    plot_name: Optional[str] = None,
    **kwargs
) -> None:
    dataset = pd.read_csv(path)
    dataset = drop_df_columns(dataset)
    if isinstance(stack, Iterable):
        dataset = stack_df_columns(dataset, stack)
    else:
        dataset = dataset[dataset.columns[:121]]
    pearson = dataset.corr()
    spearman = dataset.corr(method='spearman')
    if use_mask:
        mask = np.triu(np.ones_like(pearson, dtype=bool), k=1)

    fig, axes = plt.subplots(figsize=figsize, nrows=1, ncols=2)
    cbar_ax = fig.add_axes([0.35, 0.92, 0.3, 0.02])
    xticklabels = yticklabels = 'auto' if stack is None else stack.values()
    sns.heatmap(pearson.round(2), ax=axes[0], mask=mask, cmap='coolwarm', square=True, vmin=-1, vmax=1,
                xticklabels=xticklabels, yticklabels=yticklabels, cbar=False, **kwargs)
    sns.heatmap(spearman.round(2), ax=axes[1], mask=mask, cmap='coolwarm', square=True, vmin=-1, vmax=1,
                xticklabels=xticklabels, yticklabels=yticklabels, cbar_kws={'orientation': 'horizontal'},
                cbar_ax=cbar_ax, **kwargs)
    plt.tight_layout()

    if plot_name:
        plt.savefig(plot_name, bbox_inches='tight', format='pdf')
    plt.show()


# COMPARING MODELS
def compare_distributions(
    data: Union[str, pd.DataFrame],
    difference: bool = False,
    scale: bool = False,
    predictions: Optional[str] = None,
    stack: Optional[dict] = None,
    transform: Optional[Callable] = None,
    plot_name: Optional[str] = None,
    **kwargs,
) -> None:
    if isinstance(data, str):
        dataset = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        dataset = data
    rect = None

    if isinstance(predictions, str):
        preds = pd.read_csv(predictions).filter(regex='pred-')
        preds.columns = preds.columns.str.replace('pred-', '')
        _, val_idx, test_idx = get_splits(data.replace('.csv', '.npz'), dataset)
        dataset = dataset.iloc[test_idx]
        preds.reset_index(inplace=True, drop=True)

    # Yeo-Johnson transformation has to be performed for each variable separately so that each distribution is normalized
    if transform is not None:
        dataset = dataset.apply(transform, axis=0)

    if stack is not None:
        dataset = stack_df_columns(dataset, stack=stack)
        if predictions is not None:
            preds = stack_df_columns(preds, stack=stack)
    dataset.reset_index(inplace=True, drop=True)

    # Histograms
    if difference:
        dataset -= preds  # unary operators act on index intersections
        if scale:
            dataset /= dataset.std()
        ax = dataset.hist(label='Difference', **kwargs)
    else:
        ax = dataset.hist(label='Ground Truth', **kwargs)
        if predictions is not None:
            preds.hist(ax=ax, label='Model', alpha=0.7, **kwargs)
            fig = ax.flat[0].get_figure()
            # Get handles and labels from just the first axis to avoid duplicates
            handles, labels = ax.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=2)
            rect = [0, 0, 1, 0.99]

    # Labeling and titles
    for idx, axis in enumerate(ax.flat):
        if stack == _MUSCLES_D or stack == _MUSCLES_LATIN_D:
            axis.set_yscale('log')
        if stack is not None and idx < len(stack):
            titles = list(stack.values())
            axis.set_title(titles[idx])
    # Add labels only on the left and at the bottom
    for axis in ax:
        axis[0].set_ylabel('Frequency')
    for axis in ax.T:
        axis = axis[-1]
        if 'M_' in axis.get_title():
            axis.set_xlabel('Moment [Nm]')
        else:
            axis.set_xlabel('Force [N]')

    plt.tight_layout(rect=rect)
    if plot_name:
        plt.savefig(plot_name, bbox_inches='tight', format='pdf')
    plt.show()


def compare_models(
    *eval_files: Tuple[str, str],
    metric: Callable = absolute_error,
    width: float = 0.25,
    figsize: tuple = (8, 5),
    stack: Optional[Iterable[str]] = None
):
    fig, ax = plt.subplots(figsize=figsize)
    factor = 0
    offset_ticks = (len(eval_files) * width - width) / 2
    labels = pd.Index([item[1] for item in eval_files])

    pivot = {l: pd.read_csv(p) for p, l in eval_files}
    for l, d in pivot.items():
        d = d.filter(regex=metric.__name__ + '-')
        d = stack_df_columns(d, stack=stack)
        # d.drop(columns='id', inplace=True)
        d.insert(0, 'label', l)
        pivot[l] = d
    df = pd.concat([d for d in pivot.values()], ignore_index=True)

    # define the first model in eval_files as the base reference
    base = df[df['label'] == labels[0]]
    base = base[base.columns[~base.columns.str.contains('label')]].mean()
    # base['label'] = labels[0]

    for label, df in pivot.items():
        if label == labels[0]:
            continue
        offset = width * factor
        if metric.__name__ == 'rmse':
            base = rmse(base)
        df = df.filter(regex=metric.__name__ + '-')
        df = df.mean()
        # scale values to the reference model
        # df = df.apply(scale, base=base)
        df /= base
        df *= 100
        # vals = rmse(df)
        x = np.arange(len(df.index)) * ((len(eval_files) + 1) * width) + offset
        bar = ax.bar(x, df.values, label=label, width=width, alpha=0.75)
        ax.bar_label(bar, fmt='{:.2f}%', label_type='edge')
        factor += 1

    labels = list(stack.values()) if isinstance(stack, dict) else stack
    ax.set_xticks(x - offset_ticks, labels, rotation=90)
    ax.grid(axis='y', linewidth=0.5, linestyle='--')
    plt.ylabel(metric.__name__.replace('_', ' ').capitalize())
    plt.legend(bbox_to_anchor=(1.1, 0.6))
    plt.tight_layout()
    plt.show()


# DEBUG
# if __name__ == '__main__':
#     plot_metric('abs_err', '../data/eval/eval.csv', 'test')
