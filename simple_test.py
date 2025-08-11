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
        img = nib.load(row["nifti_path"]).get_fdata().astype(np.float32)

        # Normalize
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)

        # Resize to (64,64,64)
        img = torch.tensor(img).unsqueeze(0)
        img = F.interpolate(img.unsqueeze(0), size=(64, 64, 64), mode="trilinear", align_corners=False).squeeze(0)

        img = img.repeat(3, 1, 1, 1)

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
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def validate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)
            total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

# ----------------------------
# Main
# ----------------------------
def main():
    csv_path = "data.csv"  # your CSV
    batch_size = 2
    lr = 1e-4
    num_epochs = 10

    wandb.init(project="ct-3d-resnet", config={"batch_size": batch_size, "lr": lr, "epochs": num_epochs})

    df = pd.read_csv(csv_path)

    # Manual 80/20 split
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    train_dataset = CTDataset(df.iloc[train_idx])
    val_dataset = CTDataset(df.iloc[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet3DRegressor(out_features=5).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_loss = validate(model, val_loader, device, criterion)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "ct_resnet3d.pth")

if __name__ == "__main__":
    main()

# %%
