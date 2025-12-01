import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F

from typing import Tuple, Union
from torch import Tensor
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, df, target_cols: str, augment: bool = False):
        if isinstance(df, str):
            self.df = pd.read_csv(df)
        else:
            self.df = df.reset_index(drop=True)
        self.target_cols =  self.get_targets(target_cols)
        self.cond_cols = self.get_targets('Ang|weight')
        self.augment = augment

    @property
    def dim(self) -> int:
        return len(self.target_cols)

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower().replace('dataset', '')

    @staticmethod
    def _normalize(x: np.array) -> Tensor:
        x = (x - np.mean(x)) / (np.std(x) + 1e-8)
        x = torch.tensor(x).unsqueeze(0)
        x = F.interpolate(x.unsqueeze(0), size=(128, 128, 128), mode="trilinear", align_corners=False).squeeze(0)
        return x

    @staticmethod
    def _import_mask(row, path: str) -> Tensor:
        segmentation_mask = nib.load(
            row["nifti_path"].replace("_ct.nii.gz", path), mmap=False
        ).get_fdata().astype(np.float32)
        segmentation_mask = torch.tensor(segmentation_mask).unsqueeze(0)
        segmentation_mask = F.interpolate(
            segmentation_mask.unsqueeze(0), size=(128, 128, 128), mode="trilinear", align_corners=False
        ).squeeze(0)
        return segmentation_mask

    @staticmethod
    def _augment(v: Tensor) -> Tensor:
        if torch.rand(1).item() < 0.5:
            v = torch.flip(v, dims=[2])
        if torch.rand(1).item() < 0.5:
            v = torch.flip(v, dims=[3])
        if torch.rand(1).item() < 0.2:
            v = v + 0.01 * torch.randn_like(v)
        return v

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

    def import_img(self, idx: int) -> Tuple[Tensor, pd.DataFrame]:
        row = self.df.iloc[idx]
        try:
            img = nib.load(row["nifti_path"], mmap=False).get_fdata().astype(np.float32)
            # print(f"Loaded image shape: {img.shape} from {row['nifti_path']}")
            img = self._normalize(img)
            if self.augment:
                img = self._augment(img)
            tissueSeg = self._import_mask(row, "_seg-tissue_msk.nii.gz")
            spineSeg = self._import_mask(row, "_seg-spine_msk.nii.gz")
            img = torch.cat([img, tissueSeg, spineSeg], dim=0)  # [3, D, H, W]
            return img, row
        except EOFError:
            print(f"[WARNING] Skipping corrupted file: {row['nifti_path']}")
            return self.__getitem__((idx + 1) % len(self.df))  # pick next one

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        img, row = self.import_img(idx)
        target = torch.tensor(row[self.target_cols].values.astype(np.float32))
        try:
            conditioning = torch.tensor(row[self.cond_cols].values.astype(np.float32))
        except KeyError:
            conditioning = None
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
    """For training MLP: shear and compression -> muscle forces"""
    def __init__(self, df, target_cols='compr|shear'):
        super().__init__(df, target_cols)
        self.input_cols = self.get_targets('G_|F_Ribcage')    # muscle forces

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        row = self.df.iloc[idx]
        input_ = torch.tensor(row[self.input_cols].values.astype(np.float32))
        target = torch.tensor(row[self.target_cols].values.astype(np.float32))
        return input_, target
