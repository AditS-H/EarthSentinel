# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 14:08:45 2025
@author: advit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, confusion_matrix
from sklearn.utils import resample
import random
import matplotlib.pyplot as plt

# ---------------- Chunk Loader ----------------
class ChunkedPatchLoader:
    def __init__(self, chunk_dir, verbose=False):
        self.chunk_dir = chunk_dir
        self.verbose = verbose
        metadata_path = os.path.join(chunk_dir, "metadata.npy")
        self.metadata = np.load(metadata_path, allow_pickle=True).item()
        self._chunk_cache = {}
        self._cache_size_limit = 3

    def get_patch_data(self, patch_indices, week_idx=None):
        patches_per_chunk = self.metadata['patches_per_chunk']
        patch_data = []
    
        chunk_groups = {}
        for patch_idx in patch_indices:
            chunk_idx = patch_idx // patches_per_chunk
            if chunk_idx not in chunk_groups:
                chunk_groups[chunk_idx] = []
            chunk_groups[chunk_idx].append(patch_idx)
    
        for chunk_idx, patch_list in chunk_groups.items():
            chunk_file = os.path.join(self.chunk_dir, f'chunk_{chunk_idx:03d}.npy')
            # Load only this chunk (still big, but fewer chunks in memory)
            chunk_data = np.load(chunk_file, mmap_mode='r')
            
            for patch_idx in patch_list:
                patch_in_chunk = patch_idx % patches_per_chunk
                patch = chunk_data[patch_in_chunk].copy()  # make writable
                # normalize per band
                for b in range(patch.shape[0]):
                    band = patch[b]
                    band = np.clip(band, -30000, 30000)
                    bmin, bmax = band.min(), band.max()
                    patch[b] = (band - bmin) / (bmax - bmin) if bmax - bmin > 0 else np.zeros_like(band)
                if week_idx is not None:
                    patch = patch[week_idx]
                patch_data.append(patch)

        return np.array(patch_data, dtype=np.float32)


    def _load_chunk(self, chunk_idx):
        if chunk_idx in self._chunk_cache:
            return self._chunk_cache[chunk_idx]

        chunk_file = os.path.join(self.chunk_dir, f'chunk_{chunk_idx:03d}.npy')
        chunk_data = np.load(chunk_file).astype(np.float32)

        # normalize per band
        for b in range(chunk_data.shape[2]):
            band = chunk_data[:, :, b]
            band = np.clip(band, -30000, 30000)
            band_min, band_max = band.min(), band.max()
            if band_max - band_min > 0:
                band = (band - band_min) / (band_max - band_min)
            else:
                band = np.zeros_like(band)
            chunk_data[:, :, b] = band

        if len(self._chunk_cache) >= self._cache_size_limit:
            oldest_chunk = next(iter(self._chunk_cache))
            del self._chunk_cache[oldest_chunk]
        self._chunk_cache[chunk_idx] = chunk_data
        return chunk_data

    def get_single_patch(self, patch_idx, week_idx=None):
        return self.get_patch_data([patch_idx], week_idx)[0]

    @property
    def shape(self):
        return (self.metadata['total_patches'],
                self.metadata['num_weeks'],
                self.metadata['bands'],
                self.metadata['patch_size'],
                self.metadata['patch_size'])


# ---------------- Models ----------------
class CNN_LSTM_ENCODER(nn.Module):
    def __init__(self, input_channels=4, cnn_feature_dim=512, lstm_hidden=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, cnn_feature_dim)
        self.lstm = nn.LSTM(input_size=cnn_feature_dim, hidden_size=lstm_hidden, batch_first=True)

    def forward(self, x):
        # x shape: (batch, weeks, bands, H, W)
        batch_size, weeks, bands, H, W = x.shape
        cnn_out = []
        for t in range(weeks):
            xi = x[:, t]                   # shape (batch, bands, H, W)
            fi = self.cnn(xi).view(batch_size, -1)
            fi = self.fc(fi)
            cnn_out.append(fi)
        cnn_out = torch.stack(cnn_out, dim=1)  # (batch, weeks, cnn_feature_dim)
        _, (h_n, _) = self.lstm(cnn_out)
        embedding = h_n[-1]  # (batch, lstm_hidden)
        return embedding


class Siamese_Network(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    def forward(self, x1, x2):
        emb1 = self.encoder(x1)
        emb2 = self.encoder(x2)
        return emb1, emb2


class Loss_fxn(nn.Module):
    def __init__(self, init_margin=0.5, min_margin=0.0, max_margin=5.0):
        super().__init__()
        self.margin = nn.Parameter(torch.tensor(init_margin))  # learnable
        self.min_margin = min_margin
        self.max_margin = max_margin

    def forward(self, emb1, emb2, label):
        # clip margin to avoid blow-up
        margin = torch.clamp(self.margin, self.min_margin, self.max_margin)

        dis = F.pairwise_distance(emb1, emb2)
        loss = label * dis**2 + (1 - label) * F.relu(margin - dis)**2
        return loss.mean()


# ---------------- Dataset ----------------
class PatchPairsDataset(Dataset):
    def __init__(self, patches, pairs, labels):
        self.patches = patches
        self.pairs = pairs
        self.labels = labels
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        a_idx, b_idx = self.pairs[idx]
        x1 = torch.tensor(self.patches.get_patch_data([a_idx], week_idx=None), dtype=torch.float32)[0]
        x2 = torch.tensor(self.patches.get_patch_data([b_idx], week_idx=None), dtype=torch.float32)[0]
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x1, x2, y


# ---------------- Utils ----------------
def compute_embeddings(model, loader, device):
    """
    Returns:
      embeddings: numpy array shape (N, emb_dim*2) where for each batch we stack emb1 then emb2
      labels: numpy array shape (N,) corresponding to each pair's label
    """
    model.eval()
    embeddings_list, labels_list = [], []
    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2 = x1.to(device), x2.to(device)
            emb1, emb2 = model(x1, x2)   # both (batch, emb_dim)
            # stack emb1 and emb2 horizontally for saving (optional)
            combined = torch.cat([emb1, emb2], dim=1).cpu().numpy()  # (batch, 2*emb_dim)
            embeddings_list.append(combined)
            labels_list.append(y.cpu().numpy())
    embeddings = np.vstack(embeddings_list) if len(embeddings_list) > 0 else np.zeros((0,))
    labels = np.concatenate(labels_list) if len(labels_list) > 0 else np.zeros((0,))
    return embeddings, labels


def select_optimal_threshold(model, loader, device):
    """
    Computes the ROC on the loader using scores = -distance (so higher score => more similar)
    Finds the threshold in score-space that maximizes Youden's J (tpr - fpr)
    Returns: threshold in distance-space (i.e., distances <= returned_threshold => predict positive)
    """
    model.eval()
    distances, labels = [], []
    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2 = x1.to(device), x2.to(device)
            emb1, emb2 = model(x1, x2)
            d = F.pairwise_distance(emb1, emb2)
            distances.extend(d.cpu().numpy())
            labels.extend(y.cpu().numpy())
    distances = np.array(distances)
    labels = np.array(labels)

    # Use scores = -distances so larger score => more likely "same" (label=1)
    scores = -distances
    fpr, tpr, thresholds = roc_curve(labels, scores)
    J = tpr - fpr
    ix = np.argmax(J)
    best_score_threshold = thresholds[ix]
    # convert back to distance threshold
    best_distance_threshold = -best_score_threshold
    return best_distance_threshold


# ---------------- Main ----------------
if __name__ == "__main__":
    CHUNK_DIR = "patch_chunks"
    print("Loading chunked patches...")
    patch_loader = ChunkedPatchLoader(CHUNK_DIR, verbose=False)
    print(f"Chunked patch loader initialized! Virtual shape: {patch_loader.shape}")

    pairs = np.load("siamese_week_pairs.npy")
    labels = np.load("siamese_week_pair_labels.npy")

    # balance pos/neg
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("One of the classes (pos or neg) has zero samples.")
    neg_downsampled = resample(neg_idx, replace=False, n_samples=len(pos_idx), random_state=42)
    balanced_idx = np.concatenate([pos_idx, neg_downsampled])
    np.random.shuffle(balanced_idx)
    pairs, labels = pairs[balanced_idx], labels[balanced_idx]

    train_pairs, test_pairs, train_labels, test_labels = train_test_split(
        pairs, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = PatchPairsDataset(patch_loader, train_pairs, train_labels)
    test_dataset  = PatchPairsDataset(patch_loader, test_pairs, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = CNN_LSTM_ENCODER(input_channels=patch_loader.metadata['bands'])
    model = Siamese_Network(encoder).to(device)
    loss_fxn = Loss_fxn()
    optimizer = torch.optim.Adam(list(model.parameters()) + [loss_fxn.margin], lr=1e-4)

    # -------- Normal training with monitoring --------
    for epoch in range(5):
        running_loss = 0.0
        for idx, (x1, x2, y) in enumerate(train_loader):
            # skip batches that are all-negative or all-positive (no gradient signal for contrasting)
            if y.sum().item() == 0 or y.sum().item() == y.numel():
                continue 
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            optimizer.zero_grad()
            emb1, emb2 = model(x1, x2)
            loss = loss_fxn(emb1, emb2, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Monitor every 50 batches
            if idx % 50 == 0:
                with torch.no_grad():
                    distances = F.pairwise_distance(emb1, emb2)
                    pos_mask = (y == 1)
                    neg_mask = (y == 0)
                    if pos_mask.any():
                        pos_dist = distances[pos_mask].cpu().numpy()
                    else:
                        pos_dist = np.array([np.nan])
                    if neg_mask.any():
                        neg_dist = distances[neg_mask].cpu().numpy()
                    else:
                        neg_dist = np.array([np.nan])

                    print(f"Epoch {epoch+1}, Batch {idx}, Loss {loss.item():.6f}")
                    print(f"   Margin: {loss_fxn.margin.item():.6f}")
                    print(f"   Pos Distances: min {np.nanmin(pos_dist):.6f}, max {np.nanmax(pos_dist):.6f}, mean {np.nanmean(pos_dist):.6f}")
                    print(f"   Neg Distances: min {np.nanmin(neg_dist):.6f}, max {np.nanmax(neg_dist):.6f}, mean {np.nanmean(neg_dist):.6f}")

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"✅ Epoch {epoch+1} completed, Avg Loss {avg_loss:.6f}")

        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss_state": loss_fxn.state_dict()
        }
        torch.save(checkpoint, f"checkpoint_epoch{epoch+1}.pth")

    torch.save(model.state_dict(), "siamese_model.pth")

    # -------- Evaluation --------
    train_embeddings, train_labels = compute_embeddings(model, train_loader, device)
    test_embeddings, test_labels = compute_embeddings(model, test_loader, device)
    np.save("train_embeddings.npy", train_embeddings)
    np.save("train_labels.npy", train_labels)
    np.save("test_embeddings.npy", test_embeddings)
    np.save("test_labels.npy", test_labels)

    # compute pairwise distances on test set and labels (for classification)
    model.eval()
    distances, labels_eval = [], []
    with torch.no_grad():
        for x1, x2, y in test_loader:
            if y.sum().item() == 0 or y.sum().item() == y.numel():
                continue
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            emb1, emb2 = model(x1, x2)
            d = F.pairwise_distance(emb1, emb2)
            distances.extend(d.cpu().numpy())
            labels_eval.extend(y.cpu().numpy())
    distances = np.array(distances)
    labels_eval = np.array(labels_eval)

    # select threshold (distance-space). returns d_threshold where distances <= d_threshold -> predict positive (label=1)
    threshold = select_optimal_threshold(model, train_loader, device)
    preds = (distances <= threshold).astype(int)

    print("\n📊 Test classification report:")
    print(classification_report(labels_eval, preds, digits=4, zero_division=0))

    print("\n🔍 Confusion Matrix:")
    print(confusion_matrix(labels_eval, preds))

    # Distance distribution visualization
    try:
        plt.figure(figsize=(8,5))
        plt.hist(distances[labels_eval==1], bins=50, alpha=0.5, label="Positives (label=1)")
        plt.hist(distances[labels_eval==0], bins=50, alpha=0.5, label="Negatives (label=0)")
        plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold={threshold:.6f}")
        plt.legend()
        plt.xlabel("Pairwise Distance")
        plt.ylabel("Count")
        plt.title("Distance Distributions (Test Set)")
        plt.show()
    except Exception as e:
        print("Could not plot distance histogram:", e)
