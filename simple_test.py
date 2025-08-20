#%%
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models.video as video_models
import wandb
import os

# ----------------------------
# Dataset
# ----------------------------
class CTDataset(Dataset):
    def __init__(self, df, target_cols=None):
        if isinstance(df, str):
            self.df = pd.read_csv(df)
        else:
            self.df = df.reset_index(drop=True)
        self.target_cols = target_cols or ["y0", "y1", "y2", "y3", "y4"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = nib.load(row["nifti_path"], mmap=False).get_fdata().astype(np.float32)
            print(f"Loaded image shape: {img.shape} from {row['nifti_path']}")
        except EOFError:
            print(f"[WARNING] Skipping corrupted file: {row['nifti_path']}")
            return self.__getitem__((idx + 1) % len(self.df))  # pick next one

        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        img = torch.tensor(img).unsqueeze(0)
        img = F.interpolate(img.unsqueeze(0), size=(128, 128, 128), mode="trilinear", align_corners=False).squeeze(0)

        tissueSeg = nib.load(row["nifti_path"].replace("_ct.nii.gz", "_seg-tissue_msk.nii.gz"), mmap=False).get_fdata().astype(np.float32)
        tissueSeg = torch.tensor(tissueSeg).unsqueeze(0)
        tissueSeg = F.interpolate(tissueSeg.unsqueeze(0), size=(128, 128, 128), mode="trilinear", align_corners=False).squeeze(0)

        spineSeg = nib.load(row["nifti_path"].replace("_ct.nii.gz", "_seg-spine_msk.nii.gz"), mmap=False).get_fdata().astype(np.float32)
        spineSeg = torch.tensor(spineSeg).unsqueeze(0)
        spineSeg = F.interpolate(spineSeg.unsqueeze(0), size=(128, 128, 128), mode="trilinear", align_corners=False).squeeze(0)

        img = torch.cat([img, tissueSeg, spineSeg], dim=0)  # [3, D, H, W]
        #img = img.unsqueeze(0)  # [1, 3, D, H, W]



        #img = img.repeat(3, 1, 1, 1)
        target = torch.tensor(row[self.target_cols].values.astype(np.float32))
        return img, target


# ----------------------------
# Model
# ----------------------------
class ResNet3DRegressor(nn.Module):
    def __init__(self, out_features=5):
        super().__init__()
        self.backbone = video_models.r3d_18(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, out_features)

    def forward(self, x):
        return self.backbone(x)

# ----------------------------
# Training
# ----------------------------
def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    relative_errors = []

    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()* imgs.size(0)  # Scale loss by batch size
        relative_errors.append( torch.mean(torch.abs(preds - targets) / (targets + 1e-8)).item() )
        print("preds0:", preds.detach().cpu().numpy()[0,0])
        wandb.log({"all_outputs0": preds.detach().cpu().numpy()[0,0], "all_targets0": targets.detach().cpu().numpy()[0,1], "all_outputs1": preds.detach().cpu().numpy()[0,1], "all_targets1": targets.detach().cpu().numpy()[0,1], "all_outputs2": preds.detach().cpu().numpy()[0,2], "all_targets2": targets.detach().cpu().numpy()[0,2], "all_outputs3": preds.detach().cpu().numpy()[0,3], "all_targets3": targets.detach().cpu().numpy()[0,3], "all_outputs4": preds.detach().cpu().numpy()[0,4], "all_targets4": targets.detach().cpu().numpy()[0,4]})

    wandb.log({ "train_relative_error": np.mean(relative_errors) })

    return total_loss / len(loader.dataset)

def validate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    relative_error = []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)* imgs.size(0)  # Scale loss by batch size
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

# ----------------------------
# Main
# ----------------------------
def main():
    csv_path = "ct_nifti_scalars_full.csv"  # your CSV
    batch_size = 20
    lr = 5e-2
    num_epochs = 1000

    wandb.init(project="ct-3d-resnet", config={"batch_size": batch_size, "lr": lr, "epochs": num_epochs})

    df = pd.read_csv(csv_path)

    # Create 80/10/10 split (train/val/test)
    split_file = 'data_split_indices.npz'

    # Only create split if it doesn't exist already
    if True: #TODO not os.path.exists(split_file):
        indices = np.arange(len(df))
        np.random.shuffle(indices)
        
        # Calculate split indices
        train_split = int(0.8 * len(indices))
        val_split = int(0.9 * len(indices))
        
        # Split indices
        train_idx = indices[:train_split]
        val_idx = indices[train_split:val_split]
        test_idx = indices[val_split:]
        
        # Save indices to file
        np.savez(split_file, train=train_idx, val=val_idx, test=test_idx)
        print(f"Created and saved new data split to {split_file}")
    else:
        # Load existing split
        loaded_split = np.load(split_file)
        train_idx = loaded_split['train']
        val_idx = loaded_split['val']
        test_idx = loaded_split['test']
        print(f"Loaded existing data split from {split_file}")

    print(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    train_dataset = CTDataset(df.iloc[train_idx])
    val_dataset = CTDataset(df.iloc[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=10)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = ResNet3DRegressor(out_features=5).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    wandb.log({"model_summary": str(model), "num_params": sum(p.numel() for p in model.parameters()), "device": str(device), "batch_size": batch_size, "learning_rate": lr, "num_epochs": num_epochs, "train_size": len(train_dataset), "val_size": len(val_dataset), "train_indices": train_idx.tolist(), "val_indices": val_idx.tolist(), "test_indices": test_idx.tolist()})

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_loss = validate(model, val_loader, device, criterion)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

        torch.save(model.state_dict(), "ct_resnet3d.pth")

if __name__ == "__main__":
    main()

# %%
