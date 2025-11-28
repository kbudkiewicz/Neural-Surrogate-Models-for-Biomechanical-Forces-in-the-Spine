import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb

from nn.networks import ResNet3DRegressor, MultilayerPerceptron, Chimera
from data.datasets import CTDataset, ShearComprDataset, MuscleDataset, AllForcesDataset, ForceToForceDataset
from data.utils import prepare_dirs
from utils.eval import evaluate, relative_error, absolute_error
from utils.preprocessing import wrap_dataloader, tensor_to_dict, get_splits
from torch.utils.data import DataLoader


def train_one_epoch(model, loader, optimizer, device, criterion, epoch, epochs, target_cols):
    model.train()
    total_loss = 0
    relative_errors = []
    loader_wrap = wrap_dataloader(loader, desc=f'Training [{epoch+1}/{epochs}]')

    for batch in loader_wrap:
        imgs, targets, conditioning = batch
        imgs, targets = imgs.to(device), targets.to(device)
        if conditioning is not None:
            conditioning = conditioning.to(device)
            preds = model(imgs, conditioning)
        else:
            preds = model(imgs)
        optimizer.zero_grad()
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)  # Scale loss by batch size
        loader_wrap.set_postfix(train_loss=total_loss)
        relative_errors.append(torch.mean(torch.abs(preds - targets) / (targets + 1e-8)).item())
        # print("preds0:", preds.detach().cpu().numpy()[0,0])
        wandb.log(tensor_to_dict(preds, targets, target_cols, 'train-'))
    wandb.log({"train_relative_error": np.mean(relative_errors)})

    return total_loss / len(loader.dataset)


def validate(model, loader, device, criterion, epoch, epochs, target_cols):
    model.eval()
    total_loss = 0
    relative_error = []
    loader_wrap = wrap_dataloader(loader, desc=f'Validation [{epoch+1}/{epochs}]')
    with torch.no_grad():
        for batch in loader_wrap:
            imgs, targets, conditioning = batch
            imgs, targets = imgs.to(device), targets.to(device)
            if conditioning is not None:
                conditioning = conditioning.to(device)
                preds = model(imgs, conditioning)
            else:
                preds = model(imgs)
            loss = criterion(preds, targets) * imgs.size(0)  # Scale loss by batch size
            total_loss += loss.item() * imgs.size(0)
            relative_error.append(torch.mean(torch.abs(preds - targets) / (targets + 1e-8)).item())

            wandb.log({"val_all_outputs0": preds.detach().cpu().numpy()[0,0], "val_all_targets0": targets.detach().cpu().numpy()[0,1], "val_all_outputs1": preds.detach().cpu().numpy()[0,1], "val_all_targets1": targets.detach().cpu().numpy()[0,1], "val_all_outputs2": preds.detach().cpu().numpy()[0,2], "val_all_targets2": targets.detach().cpu().numpy()[0,2], "val_all_outputs3": preds.detach().cpu().numpy()[0,3], "val_all_targets3": targets.detach().cpu().numpy()[0,3], "val_all_outputs4": preds.detach().cpu().numpy()[0,4], "val_all_targets4": targets.detach().cpu().numpy()[0,4]})
    #log example preds and targets
    wandb.log({#"example_preds": wandb.Histogram(preds.cpu().numpy()),
        #"example_targets": wandb.Histogram(targets.cpu().numpy()),
        "val_relative_error": np.mean(relative_error)
    })

    #log all outputs for one patient
    return total_loss / len(loader.dataset)


def main():
    csv_path = "data/conditional_half.csv"  # TODO
    npz_path = csv_path.replace("csv", "npz")
    batch_size = 16
    lr = 5e-3
    num_epochs = 400
    patience: int = 10
    df = pd.read_csv(csv_path)
    train_idx, val_idx, test_idx = get_splits(npz_path, df)

    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    train_dataset, val_dataset = ShearComprDataset(df.iloc[train_idx]), ShearComprDataset(df.iloc[val_idx])   # TODO
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=16)
    target_cols = train_dataset.target_cols
    eval_name, model_name = prepare_dirs('./data/eval_chimera', train_dataset.name)
    best_loss = float('inf')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Training "{model_name}" on {device}.')
    # model = ResNet3DRegressor(out_features=train_dataset.dim)
    # model = MultilayerPerceptron(103, 512, 512, 256, train_dataset.dim)
    model = Chimera(512, 128, train_dataset.dim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    wandb.init(project='ct-3d-resnet-all', config={"batch_size": batch_size, "lr": lr, "epochs": num_epochs})
    wandb.log({
        "model_summary": str(model),
        "num_params": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_epochs": num_epochs,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist()
    })

    try:
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion, epoch, num_epochs, target_cols)
            val_loss = validate(model, val_loader, device, criterion, epoch, num_epochs, target_cols)
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), model_name)
        evaluate(absolute_error, relative_error,
                 model=model, loader=val_loader, device=device, csv_filename=eval_name, weights=model_name
        )
    except KeyboardInterrupt as e:
        print(f'Caught {e}. Stopping training and saving intermediate model params.')
        torch.save(model.state_dict(), model_name)


if __name__ == "__main__":
    main()
