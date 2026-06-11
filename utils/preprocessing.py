import os
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from typing import Optional, Tuple
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split


def _change_nifti_path(csv_file):
    df = pd.read_csv(csv_file)
    df['nifti_path'] = df['nifti_path'].apply(lambda x: x.replace('8tb_slot8/jonas/datasets/CT_nofx/', ''))
    df.to_csv(csv_file, index=False)


def get_splits(npz_filename: str, df: pd.DataFrame, allow_pickle: bool = False) -> Tuple:
    """Create a split 80/10/10 (train/val/test) if it doesn't exist already. Otherwise, load an existing split."""
    if not os.path.isfile(npz_filename):
        indices = np.arange(len(df))
        np.random.shuffle(indices)

        # Calculate split indices
        train_split = int(0.8 * len(indices))
        val_split = int(0.9 * len(indices))

        # Split indices
        train_idx = indices[:train_split]
        val_idx = indices[train_split:val_split]
        test_idx = indices[val_split:]

        # Save indices to file
        np.savez(npz_filename, train=train_idx, val=val_idx, test=test_idx)
        print(f"Created and saved new data split to {npz_filename}")
    else:
        loaded_split = np.load(npz_filename, allow_pickle=allow_pickle)
        train_idx = loaded_split['train']
        val_idx = loaded_split['val']
        test_idx = loaded_split['test']
        print(f"Loaded existing data split from {npz_filename}")
    return train_idx, val_idx, test_idx


def prepare_dataloaders(dataset: Dataset, batch_size: int, seed: Optional[int] = None):
    """Splits the data into Train-Val-Test sets with ratio 8:1:1"""
    if isinstance(seed, int):
        generator = torch.manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, test_loader


def wrap_dataloader(loader: DataLoader, desc) -> tqdm:
    return tqdm(loader, total=len(loader), ncols=100, desc=desc)


def tensor_to_dict(pred: Tensor, target: Tensor, tnames: iter, predicate: str) -> dict:
    """
    Args:
        tnames: iterator containing names of each target
        predicate: str put in front of tname
    """
    if pred.shape != target.shape:
        raise ValueError('Predictions and target tensors must have the same shape')
    d = {}
    pred, target = pred.detach().cpu().numpy(), target.detach().cpu().numpy()
    for p, t, name in zip(pred, target, tnames):
        pname, tname = predicate + name + '-output', predicate + name + '-target'
        d[pname] = p
        d[tname] = t
    return d


# DEBUG
# if __name__ == '__main__':
    # for csv in {'../ct_nifti_scalars.csv', '../ct_nifti_scalars_full.csv', '../data.csv', '../test_predictions.csv'}:
    #     _change_nifti_path(csv)
    # print('Done.')
