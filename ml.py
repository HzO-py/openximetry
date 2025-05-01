import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from model import AttentionBiLSTM, PPGRegressionDataset,WINDOW_SEC  # Ensure imports
from torch.utils.data import DataLoader, WeightedRandomSampler

def make_balanced_train_loader(train_ds, batch_size=64, num_bins=10):
    # 1) Extract targets
    y = train_ds.y.numpy()
    
    # 2) Bin the targets into 'num_bins' equal-width bins
    bins = np.linspace(0, 1, num_bins + 1)
    bin_ids = np.digitize(y, bins) - 1  # bin indices from 0 to num_bins-1
    
    # 3) Compute bin counts and assign inverse-frequency weights
    bin_counts = np.bincount(bin_ids, minlength=num_bins)
    print(y.shape[0],bin_counts)
    weights = 1.0 / (bin_counts[bin_ids] + 1e-6)  # avoid division by zero
    
    # 4) Create a WeightedRandomSampler
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    # 5) Return a DataLoader using this sampler
    return DataLoader(
        train_ds, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=4, 
        pin_memory=True, 
        prefetch_factor=2
    )


def train_5fold(
    dataset_dir: str = 'dataset',
    split_csv: str = 'encounter_5folds.csv', 
    batch_size: int      = 256,
    epochs: int          = 30,
    lr: float            = 1e-3,
    hidden_dim: int      = 128,
    num_layers: int      = 2,
    dropout: float       = 0.3,
    output_dir: str      = 'output'
):
    os.makedirs(output_dir, exist_ok=True)
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    splits    = pd.read_csv(split_csv)
    splits['fold'] = splits['fold'].fillna(-1).astype(int)
    criterion = nn.MSELoss()

    # Prepare test loader once
    test_ds    = PPGRegressionDataset(dataset_dir, splits, fold=None, mode='test')
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=4, pin_memory=True, prefetch_factor=2)

    for fold in range(5):
        print(f"\n--- Fold {fold} ---")

        train_ds = PPGRegressionDataset(dataset_dir, splits, fold, mode='trainval', fold_mode='train')
        val_ds   = PPGRegressionDataset(dataset_dir, splits, fold, mode='trainval', fold_mode='val')
        train_loader = make_balanced_train_loader(train_ds, batch_size=batch_size, num_bins=10)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, num_workers=4, pin_memory=True, prefetch_factor=2)

        model     = AttentionBiLSTM(input_dim=2, hidden_dim=hidden_dim,
                                    num_layers=num_layers, dropout=dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_losses = []
        val_losses   = []
        best_val_loss = float('inf')

        # Epoch loop with tqdm
        for ep in range(1, epochs+1):
            # -- train --
            model.train()
            tloss = 0.0
            # for Xb, yb in tqdm(train_loader, desc=f"Fold {fold} Ep{ep} Train", leave=False):
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(Xb).squeeze(1)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                tloss += loss.item() * Xb.size(0)
            tloss /= len(train_ds)
            train_losses.append(tloss)

            # -- val --
            model.eval()
            vloss = 0.0
            with torch.no_grad():
                # for Xb, yb in tqdm(val_loader, desc=f"Fold {fold} Ep{ep} Val", leave=False):
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    vloss += criterion(model(Xb).squeeze(1), yb).item() * Xb.size(0)
            vloss /= len(val_ds)
            val_losses.append(vloss)
            print(f"Epoch {ep:02d}: train={tloss:.6f}  val={vloss:.6f}")

            # -- save best model on val --
            if vloss < best_val_loss:
                best_val_loss = vloss
                torch.save(model.state_dict(), os.path.join(output_dir, f"best_model_fold{fold}_win{WINDOW_SEC}.pth"))

            # -- update and save loss plot --
            plt.figure(figsize=(6,4))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses,   label='Val Loss')
            plt.title(f'Fold {fold} Loss Curves')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'loss_curve_fold{fold}_win{WINDOW_SEC}.png'))
            plt.close()

        # -- final test evaluation --
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            # for Xb, yb in tqdm(test_loader, desc=f"Test", leave=False):
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                test_loss += criterion(model(Xb).squeeze(1), yb).item() * Xb.size(0)
        test_loss /= len(test_ds)
        print(f"Fold {fold} → Test MSE: {test_loss:.4f}")

# Run training
if __name__ == '__main__':
    train_5fold()

