import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F

from typing import Tuple, Union, Optional
# from TPTBox import NII
from torch import Tensor
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, df, target_cols: str, resolution: int = 128, augment: bool = False):
        if isinstance(df, str):
            self.df = pd.read_csv(df)
        else:
            self.df = df.reset_index(drop=True)
        self.target_cols =  self.get_targets(target_cols)
        self.cond_cols = self.get_targets('Ang|weight')
        self.paths_key = self.get_targets('_path').values[0]
        self.resolution = resolution
        self.augment = augment

    @property
    def dim(self) -> int:
        return len(self.target_cols)

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower().replace('dataset', '')

    @property
    def std(self) -> Tensor:
        vals = self.df[self.target_cols].values
        return torch.std(torch.from_numpy(vals), dim=0)

    @staticmethod
    def _load_nifti(path: str) -> Tensor:
        img = nib.load(path, mmap=False).get_fdata(dtype=np.float32)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        img = torch.tensor(img).unsqueeze(0)
        return img

    @staticmethod
    def _augment(v: Tensor) -> Tensor:
        if torch.rand(1).item() < 0.5:
            v = torch.flip(v, dims=[2])
        if torch.rand(1).item() < 0.5:
            v = torch.flip(v, dims=[3])
        if torch.rand(1).item() < 0.2:
            v = v + 0.01 * torch.randn_like(v)
        return v

    @staticmethod
    def _normalize(x: np.array) -> Tensor:
        x = (x - x.mean()) / (x.std() + 1e-8)
        return x

    def _interpolate(self, x: Tensor, size: Optional[int] = None) -> Tensor:
        if isinstance(size, int):
            size = (size, size, size)
        else:
            size = (self.resolution, self.resolution, self.resolution)
        return F.interpolate(x.unsqueeze(0), size=size, mode='trilinear', align_corners=False).squeeze(0)

    def _mask_df(self, key: str, condition, negative: bool = False, inplace: bool = False) -> Union[None, pd.DataFrame]:
        mask = (self.df[key].values == condition)
        df = self.df[~mask] if negative else self.df[mask]
        df.reset_index(drop=True, inplace=True)
        if inplace:
            self.df = df
        else:
            return df

    def get_targets(self, regex: str) -> pd.Index:
        """Find columns fitting a regex pattern"""
        header = self.df.columns
        return header[header.str.contains(regex)]

    def import_nii(self, path: str) -> Tensor:
        img = self._load_nifti(path)
        img = self._interpolate(img)
        img = self._normalize(img)
        if self.augment:
            img = self._augment(img)
        return img

    def import_mask(self, path: str, suffix: str) -> Tensor:
        path = path.replace('_ct.nii.gz', suffix)
        mask = self._load_nifti(path)
        mask = self._interpolate(mask)
        return mask

    def import_img(self, idx: int) -> Tuple[Tensor, pd.DataFrame]:
        row = self.df.iloc[idx]
        img = self._load_nifti(row['nifti_path'])
        img = self._interpolate(img)
        img = self._normalize(img)
        if self.augment:
            img = self._augment(img)

        tissueSeg = self.import_mask(row, "_seg-tissue_msk.nii.gz")
        spineSeg = self.import_mask(row, "_seg-spine_msk.nii.gz")
        img = torch.cat([img, tissueSeg, spineSeg], dim=0)  # [3, D, H, W]
        return img, row
        # except EOFError:
        #     print(f"[WARNING] Skipping corrupted file: {row[self.paths_key]}")
        #     return self.__getitem__((idx + 1) % len(self.df))  # pick next one

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, float]:
        selected_row = self.df.iloc[idx]
        nifti_path = selected_row[self.paths_key]

        img = self.import_nii(nifti_path)
        tissue_mask = self.import_mask(nifti_path, '_seg-tissue_msk.nii.gz')
        spine_mask = self.import_mask(nifti_path, '_seg-spine_msk.nii.gz')
        img = torch.cat([img, tissue_mask, spine_mask], dim=0)
        target = torch.tensor(selected_row[self.target_cols].values.astype(np.float32))

        try:
            conditioning = torch.tensor(selected_row[self.cond_cols].values.astype(np.float32))
        except KeyError:
            conditioning = torch.nan

        return img, target, conditioning

    def __len__(self):
        return len(self.df)


class CTDataset(BaseDataset):
    """For training a ResNet: img -> shear forces"""
    def __init__(self, df):
        super().__init__(df, target_cols="y")   # -> y0, ..., y4


class ShearComprDataset(BaseDataset):
    """For training a ResNet: img -> shear, and compression forces"""
    def __init__(self, df, target_cols='compr|shear'):
        super().__init__(df, target_cols)


class MuscleDataset(BaseDataset):
    """For training a ResNet: img -> muscle forces"""
    def __init__(self, df, target_cols='G_|F_Ribcage'):
        super().__init__(df, target_cols)


class AllForcesDataset(BaseDataset):
    """For training a ResNet: img -> shear, compression, and muscle forces"""
    def __init__(self, df, target_cols='compr|shear|G_|F_Ribcage'):
        super().__init__(df, target_cols)


class ForceToForceDataset(BaseDataset):
    """For training an MLP: shear and compression -> muscle forces"""
    def __init__(self, df, target_cols='compr|shear'):
        super().__init__(df, target_cols)
        self.input_cols = self.get_targets('G_|F_Ribcage')    # muscle forces

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, torch.nan]:
        row = self.df.iloc[idx]
        input_ = torch.tensor(row[self.input_cols].values.astype(np.float32))
        target = torch.tensor(row[self.target_cols].values.astype(np.float32))
        return input_, target, torch.nan
