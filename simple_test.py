import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from nn.networks import ResNet3DRegressor, MultilayerPerceptron, Chimera
from nn.losses import ScaledLoss
from data.datasets import *
from data.utils import prepare_dirs
from utils.eval import evaluate, relative_error, absolute_error, absolute_error_by_std
from utils.preprocessing import wrap_dataloader, get_splits

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader


def train_one_epoch(model, loader, optimizer, device, criterion, epoch, epochs, target_cols):
    model.train()
    total_loss = 0
    loader_wrap = wrap_dataloader(loader, desc=f'Training [{epoch+1}/{epochs}]')

    for batch in loader_wrap:
        imgs, targets, conditioning = batch
        imgs, targets = imgs.to(device), targets.to(device)
        if conditioning.isnan().any():
            preds = model(imgs)
        else:
            conditioning = conditioning.to(device)
            preds = model(imgs, conditioning)

        optimizer.zero_grad()
        loss = criterion(preds, targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)  # Scale loss by batch size
        loader_wrap.set_postfix(train_loss=total_loss)
        # print("preds0:", preds.detach().cpu().numpy()[0,0])

    return total_loss / len(loader.dataset)


def validate(model, loader, device, criterion, epoch, epochs, target_cols):
    model.eval()
    total_loss = 0
    loader_wrap = wrap_dataloader(loader, desc=f'Validation [{epoch+1}/{epochs}]')
    with torch.no_grad():
        for batch in loader_wrap:
            imgs, targets, conditioning = batch
            imgs, targets = imgs.to(device), targets.to(device)
            if conditioning.isnan().any():
                preds = model(imgs)
            else:
                conditioning = conditioning.to(device)
                preds = model(imgs, conditioning)
            loss = criterion(preds, targets) * imgs.size(0)  # Scale loss by batch size
            total_loss += loss.item() * imgs.size(0)

    #log all outputs for one patient
    return total_loss / len(loader.dataset)


def main():
    csv_path = "data/conditional_half.csv"  # TODO
    npz_path = csv_path.replace("csv", "npz")
    batch_size = 16
    lr = 5e-4
    num_epochs = 100
    num_workers = 16
    best_loss = float('inf')
    csv_path = "data/csv/nako_zeroed_.csv"
    npz_path = csv_path.replace("csv", "npz")
    df = pd.read_csv(csv_path)
    train_idx, val_idx, test_idx = get_splits(npz_path, df)
    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    criterion = nn.MSELoss()
    dataset = ShearComprDataset
    train_dataset, val_dataset = dataset(df.iloc[train_idx]), dataset(df.iloc[val_idx])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,)
    # pin_memory=True, sampler=DistributedSampler(train_dataset, shuffle=True))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,)
    # pin_memory=True, sampler=DistributedSampler(val_dataset, shuffle=False))

    model = ResNet3DRegressor(out_features=train_dataset.dim)
    # model = MultilayerPerceptron(103, 512, 512, 256, train_dataset.dim)
    # model = Chimera(512, 128, train_dataset.dim)
    # model = ResNet3D(train_dataset.dim)
    model.to(device)    # rank
    # model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    target_cols = train_dataset.target_cols
    eval_name, model_name = prepare_dirs(f'./data/eval_{model.__class__.__name__.lower()}', train_dataset.name)
    print(f'Training "{model_name}" on {device}. \nCriterion: {criterion.__class__.__name__}')

    try:
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion, epoch, num_epochs, target_cols)
            val_loss = validate(model, val_loader, device, criterion, epoch, num_epochs, target_cols)
            grad_norm = get_grad_norm(model)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} "
                  f"GradNorm: {grad_norm.item():.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), model_name)
        evaluate(absolute_error, relative_error, absolute_error_by_std,
                 model=model, loader=val_loader, device=device, csv_filename=eval_name, weights=model_name)
    except KeyboardInterrupt as e:
        print(f'Caught {e}. Stopping training and saving intermediate model params.')
        torch.save(model.state_dict(), model_name)


if __name__ == "__main__":
    main()
