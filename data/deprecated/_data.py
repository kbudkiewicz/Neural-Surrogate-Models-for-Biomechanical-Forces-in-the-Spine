from typing import List, Dict, Any
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class NiftiRegressionDataset(Dataset):
    def __init__(self, df, indices: List[int], target_size: int = 160, augment: bool = False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.indices = indices
        self.target_size = int(target_size)
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def _load_nifti(self, path: str) -> np.ndarray:
        img = nib.load(str(path))
        data = img.get_fdata(dtype=np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return data

    def _to_tensor_and_resize(self, vol: np.ndarray) -> torch.Tensor:
        v = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
        v = F.interpolate(v, size=(self.target_size, self.target_size, self.target_size),
                          mode="trilinear", align_corners=False)
        v = v.squeeze(0)  # [1,D,H,W]
        return v

    def _normalize(self, v: torch.Tensor) -> torch.Tensor:
        mean = v.mean()
        std = v.std()
        if float(std) < 1e-6:
            return v * 0.0
        return (v - mean) / std

    def _augment(self, v: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < 0.5:
            v = torch.flip(v, dims=[2])
        if torch.rand(1).item() < 0.5:
            v = torch.flip(v, dims=[3])
        if torch.rand(1).item() < 0.2:
            v = v + 0.01 * torch.randn_like(v)
        return v

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        
        row = self.df.iloc[self.indices[idx]]
        vol = self._load_nifti(row["nifti_path"])
        v = self._to_tensor_and_resize(vol)
        v = self._normalize(v)
        if self.augment:
            v = self._augment(v)
        target = torch.tensor([row["y0"], row["y1"], row["y2"], row["y3"], row["y4"]],
                              dtype=torch.float32)
        return {"image": v, "target": target, "id": row["id"]}

def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    targets = torch.stack([b["target"] for b in batch], dim=0)
    ids = [b["id"] for b in batch]
    return {"image": images, "target": targets, "id": ids}
