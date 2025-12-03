import random
import yaml
import torch
import numpy as np
from torch import Tensor


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_yaml(obj, path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_relative_error(preds: Tensor, targets: Tensor) -> float:
    return torch.mean(torch.abs(preds - targets) / (targets + 1e-8)).item()


def get_abs_error(preds: Tensor, targets: Tensor) -> float:
    return torch.mean(torch.abs(preds - targets)).item()
