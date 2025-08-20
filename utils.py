import random
import numpy as np
import torch
import yaml

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
