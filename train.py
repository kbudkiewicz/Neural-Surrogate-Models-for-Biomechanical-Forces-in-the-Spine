import argparse
import os
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18
import matplotlib.pyplot as plt
import pandas as pd
import wandb

from data import NiftiRegressionDataset, collate_fn
from utils import set_seed, save_yaml, load_yaml

def build_model(num_outputs: int = 5, in_channels: int = 1):
    model = r3d_18(weights=None)
    with torch.no_grad():
        w = model.stem[0].weight
        if w.shape[1] != in_channels:
            new_w = w.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1, 1)
            model.stem[0].weight = nn.Parameter(new_w)
    model.fc = nn.Linear(model.fc.in_features, num_outputs)
    return model

def make_splits(df, seed, train_frac=0.7, val_frac=0.15):
    rng = random.Random(seed)
    ids = list(df['id'].values)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train+n_val]
    test_ids = ids[n_train+n_val:]
    return {'train': train_ids, 'val': val_ids, 'test': test_ids}

def plot_middle_slice(volume: torch.Tensor, title: str = ""):
    v = volume.squeeze(0).cpu().numpy()
    D, H, W = v.shape
    mid = D // 2
    fig = plt.figure(figsize=(4,4))
    plt.imshow(v[mid, :, :], cmap='gray', origin='lower')
    plt.title(title)
    plt.axis('off')
    return fig

@torch.no_grad()
def log_sample_images(run, dataset, model, device, sample_positions, step: int = 0):
    model.eval()
    images = []
    for pos in sample_positions:
        sample = dataset[pos]
        x = sample['image'].unsqueeze(0).to(device)
        y_true = sample['target'].cpu().numpy().tolist()
        y_pred = model(x).squeeze(0).cpu().numpy().tolist()
        fig = plot_middle_slice(sample['image'], title=f"ID: {sample['id']}  step {step}")
        caption = " | ".join([
            "y_true: " + ", ".join(f"{t:.3f}" for t in y_true),
            "y_pred: " + ", ".join(f"{p:.3f}" for p in y_pred),
        ])
        images.append(wandb.Image(fig, caption=caption))
        plt.close(fig)
    if images:
        run.log({"samples": images}, step=step)

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        x = batch['image'].to(device)
        y = batch['target'].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(1, n)

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    n = 0
    all_pred = []
    all_true = []
    for batch in loader:
        x = batch['image'].to(device)
        y = batch['target'].to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        total += loss.item() * x.size(0)
        n += x.size(0)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.cpu().numpy())
    if n == 0:
        return math.nan, None, None
    return total / n, np.concatenate(all_pred), np.concatenate(all_true)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, required=True,
                        help="CSV with columns: id,nifti_path,y0,y1,y2,y3,y4")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--target_size", type=int, default=160)
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--project", type=str, default="mri-regression-3d")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {}
    if args.config and os.path.isfile(args.config):
        cfg = load_yaml(args.config)

    epochs = int(cfg.get("epochs", args.epochs))
    batch_size = int(cfg.get("batch_size", args.batch_size))
    lr = float(cfg.get("lr", args.lr))
    target_size = int(cfg.get("target_size", args.target_size))

    set_seed(args.seed)
    cudnn.benchmark = True

    df = pd.read_csv(args.data_csv)
    assert set(["id", "nifti_path", "y0", "y1", "y2", "y3", "y4"]).issubset(df.columns), \
        "CSV must have columns id,nifti_path,y0..y4"

    # Force plain Python types
    df["id"] = df["id"].astype(str)  # or .astype("int64").astype(int) if you prefer ints


    split_path = out_dir / "splits.json"
    if split_path.exists():
        with open(split_path, "r") as f:
            splits = json.load(f)
    else:
        splits = make_splits(df, seed=args.seed, train_frac=0.7, val_frac=0.15)
        with open(split_path, "w") as f:
            json.dump(splits, f, indent=2)

    id_to_idx = {row["id"]: i for i, row in df.reset_index().rename(columns={"index":"_idx"}).iterrows()}
    train_indices = [id_to_idx[_id] for _id in splits["train"]]
    val_indices = [id_to_idx[_id] for _id in splits["val"]]
    test_indices = [id_to_idx[_id] for _id in splits["test"]]

    train_ds = NiftiRegressionDataset(df, indices=train_indices, target_size=target_size, augment=True)
    val_ds = NiftiRegressionDataset(df, indices=val_indices, target_size=target_size, augment=False)
    test_ds = NiftiRegressionDataset(df, indices=test_indices, target_size=target_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_outputs=5, in_channels=1).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    run = None
    if args.log_wandb:
        run = wandb.init(project=args.project, config={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "target_size": target_size,
            "seed": args.seed
        })
        wandb.watch(model, log="gradients", log_freq=100)

    best_val = float("inf")
    save_yaml({"created": True}, out_dir / "meta.yaml")

    val_n = len(val_ds)
    sample_positions = list(np.linspace(0, max(0, val_n-1), num=min(3, val_n), dtype=int))

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, _, _ = evaluate(model, val_loader, loss_fn, device)

        if run is not None:
            run.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            if sample_positions:
                log_sample_images(run, val_ds, model, device, sample_positions, step=epoch)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_val_loss": best_val},
                       os.path.join(out_dir, "best.ckpt"))
        if epoch % 10 == 0:
            torch.save({"epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_loss": val_loss},
                       os.path.join(out_dir, f"epoch_{epoch:03d}.ckpt"))

    test_loss, test_pred, test_true = evaluate(model, test_loader, loss_fn, device)
    np.save(os.path.join(out_dir, "test_pred.npy"), test_pred if test_pred is not None else np.array([]))
    np.save(os.path.join(out_dir, "test_true.npy"), test_true if test_true is not None else np.array([]))

    if run is not None:
        run.summary["best_val_loss"] = best_val
        run.summary["test_loss"] = test_loss
        run.finish()

    print(f"Done. Best val loss {best_val:.6f}, test loss {test_loss:.6f}")
    print(f"Splits saved to {split_path}")
    print(f"Checkpoints in {out_dir}")

if __name__ == "__main__":
    main()
