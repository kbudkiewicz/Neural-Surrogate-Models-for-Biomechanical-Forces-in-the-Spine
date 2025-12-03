#%% CONFIG
# Hardcoded inputs and W&B run that contains logged test_indices
CSV_PATH = "ct_nifti_scalars_full.csv"
WEIGHTS_PATH = "ct_resnet3d.pth"

WANDB_RUN_PATH = "tumcompimg/ct-3d-resnet/5tzdxktc"
OUT_CSV = "test_predictions.csv"



BATCH_SIZE = 8
NUM_WORKERS = 8
OUT_FEATURES = 5
TARGET_COLS = ["y0", "y1", "y2", "y3", "y4"]
RESAMPLE_SIZE = (128, 128, 128)  # D, H, W
DEVICE_FALLBACK = "cpu"  # used when no CUDA is available

#%% IMPORTS
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models.video as video_models
import nibabel as nib
import wandb
import seaborn as sns
import pandas as pd

#%% DATASET
class CTDataset(Dataset):
    def __init__(self, df, target_cols=None):
        if isinstance(df, str):
            self.df = pd.read_csv(df)
        else:
            self.df = df.reset_index(drop=True)
        self.target_cols = target_cols or ["y0", "y1", "y2", "y3", "y4"]

    def _safe_load_np(self, path):
        try:
            return nib.load(path, mmap=False).get_fdata().astype(np.float32)
        except EOFError:
            print(f"[WARNING] Corrupted file, skipping, {path}")
            return None
        except Exception as e:
            print(f"[WARNING] Failed to load {path}, {e}")
            return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_np = self._safe_load_np(row["nifti_path"])
        if img_np is None:
            # return an empty sample that the evaluation loop can filter out
            return None

        # z score normalize
        img_np = (img_np - np.mean(img_np)) / (np.std(img_np) + 1e-8)
        img = torch.tensor(img_np).unsqueeze(0)  # [1, D, H, W]
        img = F.interpolate(img.unsqueeze(0), size=RESAMPLE_SIZE, mode="trilinear", align_corners=False).squeeze(0)

        tissue_path = row["nifti_path"].replace("_ct.nii.gz", "_seg-tissue_msk.nii.gz")
        spine_path = row["nifti_path"].replace("_ct.nii.gz", "_seg-spine_msk.nii.gz")

        tissue_np = self._safe_load_np(tissue_path)
        spine_np = self._safe_load_np(spine_path)
        if tissue_np is None or spine_np is None:
            return None

        tissue = torch.tensor(tissue_np).unsqueeze(0)
        tissue = F.interpolate(tissue.unsqueeze(0), size=RESAMPLE_SIZE, mode="trilinear", align_corners=False).squeeze(0)

        spine = torch.tensor(spine_np).unsqueeze(0)
        spine = F.interpolate(spine.unsqueeze(0), size=RESAMPLE_SIZE, mode="trilinear", align_corners=False).squeeze(0)

        x = torch.cat([img, tissue, spine], dim=0)  # [3, D, H, W]
        y = torch.tensor(row[self.target_cols].values.astype(np.float32))

        meta = {"nifti_path": row["nifti_path"]}
        return x, y, meta

#%% MODEL
class ResNet3DRegressor(nn.Module):
    def __init__(self, out_features=5):
        super().__init__()
        self.backbone = video_models.r3d_18(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, out_features)

    def forward(self, x):
        # expects [N, C, T, H, W], we use D as T
        return self.backbone(x)

#%% METRICS AND EVAL
@torch.no_grad()
def evaluate(model, loader, device, out_features=5):
    model.eval()
    preds_all = []
    gts_all = []
    metas_all = []

    mae_acc = []
    rel_acc = []
    mse_acc = []

    for batch in loader:
        # filter out Nones from dataset
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            continue

        imgs, targets, metas = zip(*batch)
        imgs = torch.stack(imgs, dim=0).to(device)                    # [B, 3, D, H, W]
        targets = torch.stack(targets, dim=0).to(device)              # [B, F]
        outputs = model(imgs)                                         # [B, F]

        preds_all.append(outputs.cpu())
        gts_all.append(targets.cpu())
        metas_all.extend(metas)

        err = outputs - targets
        mae = torch.mean(torch.abs(err), dim=0)
        rel = torch.mean(torch.abs(err) / (targets.abs() + 1e-8), dim=0)
        mse = torch.mean(err ** 2, dim=0)

        mae_acc.append(mae.cpu())
        rel_acc.append(rel.cpu())
        mse_acc.append(mse.cpu())

    if len(preds_all) == 0:
        raise RuntimeError("No valid samples found in the test set")

    preds_np = torch.cat(preds_all, dim=0).numpy()
    gts_np = torch.cat(gts_all, dim=0).numpy()

    mae_mean = torch.stack(mae_acc).mean(dim=0).numpy()
    rel_mean = torch.stack(rel_acc).mean(dim=0).numpy()
    mse_mean = torch.stack(mse_acc).mean(dim=0).numpy()
    rmse_mean = np.sqrt(mse_mean)

    metrics = {"MAE": mae_mean, "RelErr": rel_mean, "MSE": mse_mean, "RMSE": rmse_mean}
    return preds_np, gts_np, metas_all, metrics

#%% FETCH TEST INDICES FROM W&B
api = wandb.Api()
run = api.run(WANDB_RUN_PATH)

if "test_indices" not in run.summary:
    raise ValueError("test_indices not found in W&B run summary, log them in training with wandb.log({'test_indices': test_idx.tolist()})")

test_idx = run.summary["test_indices"]
print(f"Loaded {len(test_idx)} test indices from {WANDB_RUN_PATH}")

#%% BUILD DATASET AND DATALOADER
assert os.path.exists(CSV_PATH), f"CSV not found at {CSV_PATH}"
df = pd.read_csv(CSV_PATH)
test_df = df.iloc[test_idx].reset_index(drop=True)

dataset = CTDataset(test_df, target_cols=TARGET_COLS)

# Use a custom collate to keep None samples filterable in evaluate
def collate_keep_none(batch):
    return batch  # raw list, evaluate filters Nones

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=collate_keep_none
)

#%% LOAD MODEL
device = torch.device("cuda:0" if torch.cuda.is_available() else DEVICE_FALLBACK)
model = ResNet3DRegressor(out_features=OUT_FEATURES).to(device)

assert os.path.exists(WEIGHTS_PATH), f"Weights not found at {WEIGHTS_PATH}"
state = torch.load(WEIGHTS_PATH, map_location=device)
model.load_state_dict(state)
model.eval()
print(f"Model loaded on {device}")

#%% EVALUATE
preds, gts, metas, metrics = evaluate(model, loader, device, out_features=OUT_FEATURES)



#%% plot boxplots of error metrics over all 5 values 
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].boxplot(metrics["MAE"])
axes[0].set_title("MAE")
axes[0].set_ylabel("MAE")
axes[1].boxplot(metrics["RelErr"])
axes[1].set_title("Relative Error")
axes[1].set_ylabel("Relative Error")
axes[2].boxplot(metrics["RMSE"])
axes[2].set_title("RMSE")
axes[2].set_ylabel("RMSE")
plt.tight_layout()
plt.show()
#%%
diffs = gts - preds   # just to trigger the plot
relDiffs = diffs / (np.abs(gts) + 1e-8)
print(f"Absolute differences mean: {np.mean(np.abs(diffs), axis=0)}")
print(f"Relative differences mean: {np.mean(np.abs(relDiffs), axis=0)}")

x = range(len(diffs))
plt.scatter(x, diffs[:, 0], label="L1", marker='_')
plt.scatter(x, diffs[:, 1], label="L2", marker='_')
plt.scatter(x, diffs[:, 2], label="L3", marker='_')
plt.scatter(x, diffs[:, 3], label="L4", marker='_')
plt.scatter(x, diffs[:, 4], label="L5", marker='_')
plt.legend()
plt.title("Differences between Predictions and Ground Truth Force")
plt.xlabel("Sample/patient Index")
plt.ylabel("Force Difference")
plt.xlim(0, 20)  # limit x-axis to number of samples
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
x = range(len(diffs))
plt.plot(x, gts[:, 0], label="L1 ground truth", color="tab:blue")
plt.plot(x, preds[:, 0], label="L1 prediction", color="tab:blue", linestyle='--')
plt.plot(x, gts[:, 1], label="L2 ground truth", color="tab:orange")
plt.plot(x, preds[:, 1], label="L2 prediction", color="tab:orange", linestyle='--')
plt.plot(x, gts[:, 2], label="L3 ground truth", color="tab:green")
plt.plot(x, preds[:, 2], label="L3 prediction", color="tab:green", linestyle='--')
plt.legend()
plt.title("Ground Truth vs Predictions for L1, L2, L3")
plt.xlabel("Sample/Patient Index")
plt.ylabel("Force")
plt.xlim(0, 30)  # limit x-axis to number of samples
plt.legend()
plt.show()

#%% plot it as histogram 
# Reshape data for violin plot
violin_data = pd.DataFrame()
for i in range(5):
    temp_df = pd.DataFrame({
        'Force Difference': relDiffs[:, i] * 100,
        'Variable': f'L{i+1}'
    })
    violin_data = pd.concat([violin_data, temp_df])

# Create the violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(x='Variable', y='Force Difference', data=violin_data)
plt.title("Violin Plot of Differences between Predictions and Ground Truth Force")
plt.xlabel("Variable")
plt.ylabel("Force Difference in %")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%% SAVE CSV AND PRINT METRICS
cols_y = [f"L{i+1}" for i in range(OUT_FEATURES)]
cols_p = [f"pred_{i}" for i in range(OUT_FEATURES)]

out_df = pd.DataFrame({"nifti_path": [m["nifti_path"] for m in metas]})
out_df[cols_y] = pd.DataFrame(gts, index=out_df.index)
out_df[cols_p] = pd.DataFrame(preds, index=out_df.index)

abs_err = np.abs(preds - gts)
rel_err = abs_err / (np.abs(gts) + 1e-8)
out_df[[f"abs_err_{i}" for i in range(OUT_FEATURES)]] = pd.DataFrame(abs_err, index=out_df.index)
out_df[[f"rel_err_{i}" for i in range(OUT_FEATURES)]] = pd.DataFrame(rel_err, index=out_df.index)

out_df.to_csv(OUT_CSV, index=False)
print(f"Saved per sample predictions to {OUT_CSV}")

for i in range(OUT_FEATURES):
    print(
        f"Target {i}, MAE {metrics['MAE'][i]:.6f}, RMSE {metrics['RMSE'][i]:.6f}, RelErr {metrics['RelErr'][i]:.6f}"
    )

# %%
