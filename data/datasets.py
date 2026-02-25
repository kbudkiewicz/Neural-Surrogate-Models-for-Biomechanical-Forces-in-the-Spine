import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F

from typing import Tuple, Union, Optional
from TPTBox import NII
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
        return header[header.str.contains(regex, regex=True)]

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

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, float]:
        selected_row = self.df.iloc[idx]
        nifti_path = selected_row[self.paths_key]

        img = self.import_nii(nifti_path)
        tissue_mask = self.import_mask(nifti_path, '_seg-tissue_msk.nii.gz')
        spine_mask = self.import_mask(nifti_path, '_seg-spine_msk.nii.gz')
        img = torch.cat([img, tissue_mask, spine_mask], dim=0)
        target = torch.tensor(selected_row[self.target_cols].to_numpy(np.float32))

        try:
            conditioning = torch.tensor(selected_row[self.cond_cols].to_numpy(np.float32))
        except KeyError:
            conditioning = torch.nan
        return img, target, conditioning

    def __len__(self):
        return len(self.df)


class NakoDataset(BaseDataset):
    def __init__(self, df, target_cols: str = 'coord', augment: bool = False):
        super().__init__(df, target_cols, augment)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, float]:
        row = self.df.iloc[idx]
        t2w_path = row[self.paths_key]
        mask_path = t2w_path.replace('nako_data', 'nako_msk')

        img = self.import_nii(t2w_path)
        spine_mask = self.import_mask(mask_path, '-sag_mod-T2w_seg-spine_msk.nii.gz')
        vert_mask = self.import_mask(mask_path, '-sag_mod-T2w_seg-vert_msk.nii.gz')
        img = torch.cat([img, spine_mask, vert_mask], dim=0)
        target = torch.tensor(row[self.target_cols].to_numpy(np.float32))
        return img, target, torch.nan


class NakoBase(BaseDataset):
    def __init__(self, df, target_cols: str = 'coord', resolution: int = 128, augment: bool = False,
                 osim: Optional[Union[str, pd.DataFrame]] = None):
        super().__init__(df, target_cols, resolution, augment)
        if osim is not None:
            if isinstance(osim, str):
                self.osim = pd.read_csv(osim)
            else:
                self.osim = osim.reset_index(drop=True)
            self.cond_cols = self.osim.columns[3:]   # ignore 'id', 'nako_path', and 'osim_path'
            self.target_cols = self.get_targets('L[1-6]_coord')

    @staticmethod
    def _get_vibe_mask_path(path: str) -> str:
        path = path.replace('/rawdata_stitched', '/derivatives_Abdominal-Segmentation')
        _, tail = os.path.split(path)
        path = path.replace(tail, 'segmentation.nii.gz')
        return path

    @staticmethod
    def load_nifti(x: NII) -> Tensor:
        x = x.get_array()
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.tensor(x, dtype=torch.float).unsqueeze(0)
        return x

    def load_osim(self, idx: int) -> Tensor:
        x = self.osim.iloc[idx][self.cond_cols].to_numpy(dtype=np.float32)
        x = torch.tensor(x, dtype=torch.float).unsqueeze(0)
        return x

    def preprocess_image(self, x: NII) -> Tensor:
        """Convert a NIFTI image into a tensor.

        .. note::
            ``NII`` or ``Nifti1Images`` cannot be converted to tensor directly and must be first cast to np.array.
        """
        x = self.load_nifti(x)
        x = self._interpolate(x)
        x = self._normalize(x)
        if self.augment:
            x = self._augment(x)
        return x

    def preprocess_segmentation(self, x: NII) -> Tensor:
        x = self.load_nifti(x)
        x = self._interpolate(x)
        return x


class NakoImagesDataset(NakoBase):
    """
    Dataset containing T2w, in-phase and out-phase images. The T2w image is first upsampled to VIBES resolution and then
    preprocessed as usual.
    """
    def __init__(self, df, target_cols: str = 'coord', osim: Optional[str] = None):
        super().__init__(df, target_cols, osim=osim)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Union[Tensor, float]]:
        t2w_img_path = self.df.iloc[idx][self.paths_key]
        vibe_img_path = t2w_img_path.replace('/T2w', '/vibe')
        inphase_img_path = vibe_img_path.replace('-sag_T2w.nii.gz', '-ax_part-inphase_vibe.nii.gz')
        outphase_img_path = vibe_img_path.replace('-sag_T2w.nii.gz', '-ax_part-outphase_vibe.nii.gz')

        inphase_image = NII.load(inphase_img_path, False)
        outphase_image = NII.load(outphase_img_path, False)
        t2w_image = NII.load(t2w_img_path, False)
        t2w_image = t2w_image.resample_from_to(inphase_image, verbose=False)   # up-/resample to match VIBE resolution

        inphase_image = self.preprocess_image(inphase_image)
        outphase_image = self.preprocess_image(outphase_image)
        t2w_image = self.preprocess_image(t2w_image)

        img = torch.cat([t2w_image, inphase_image, outphase_image], dim=0)
        target = torch.tensor(self.df.iloc[idx][self.target_cols].to_numpy(np.float32))

        if self.osim is not None:
            conditioning = self.load_osim(idx)
        else:
            conditioning = torch.nan
        return img, target, conditioning


class NakoImagesSegDataset(NakoBase):
    """For training a network: 3 images + seg_mask -> STO. Uses images and segmentations masks from NAKO dataset. """
    def __init__(self, df, target_cols: str = 'coord'):
        super().__init__(df, target_cols)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, float]:
        t2w_img_path = self.df.iloc[idx][self.paths_key]
        t2w_seg_path = t2w_img_path.replace('/rawdata_stitched', '/derivatives_spine_inference_combination162_148')
        t2w_seg_path = t2w_seg_path.replace('-sag_T2w.nii.gz', '-sag_mod-T2w_seg-vert_msk.nii.gz')
        vibe_img_path = t2w_img_path.replace('/T2w', '/vibe')
        inphase_img_path = vibe_img_path.replace('-sag_T2w.nii.gz', '-ax_part-inphase_vibe.nii.gz')
        outphase_img_path = vibe_img_path.replace('-sag_T2w.nii.gz', '-ax_part-outphase_vibe.nii.gz')
        vibe_seg_path = self._get_vibe_mask_path(vibe_img_path)

        vibe_img_inphase = NII.load(inphase_img_path, False)
        vibe_img_outphase = NII.load(outphase_img_path, False)
        vibe_seg = NII.load(vibe_seg_path, True)
        t2w_img = NII.load(t2w_img_path, False)
        t2w_seg = NII.load(t2w_seg_path, True)
        t2w_img = t2w_img.resample_from_to(vibe_img_inphase, verbose=False)    # up-/resample to VIBE resolution
        t2w_seg = t2w_seg.resample_from_to(vibe_seg, verbose=False)            #

        vibe_img_inphase = self.preprocess_image(vibe_img_inphase)
        vibe_img_outphase = self.preprocess_image(vibe_img_outphase)
        vibe_seg = self.preprocess_segmentation(vibe_seg)
        t2w_img = self.preprocess_image(t2w_img)
        t2w_seg = self.preprocess_segmentation(t2w_seg)

        img = torch.cat([vibe_img_inphase, vibe_img_outphase, vibe_seg, t2w_img, t2w_seg], dim=0)
        target = torch.tensor(self.df.iloc[idx][self.target_cols].to_numpy(np.float32))
        return img, target, torch.nan


# CT Datasets
class ComprDataset(BaseDataset):
    """For training a ResNet: img -> compression forces

    .. note::
        For 'allResults.csv' use 'y' as target, so 'y0', ..., 'y4' are sampled.
    """
    def __init__(self, df, target_cols='compr'):
        super().__init__(df, target_cols)


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
        input_ = torch.tensor(row[self.input_cols].to_numpy(np.float32))
        target = torch.tensor(row[self.target_cols].to_numpy(np.float32))
        return input_, target, torch.nan
