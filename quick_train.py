# -*- coding: utf-8 -*-
"""
Fast Training & Development Script - Optimized for Quick Iteration
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
import time
from tqdm import tqdm

# ============== FAST DEVELOPMENT SETTINGS ==============
QUICK_SAMPLE_SIZE = 1000  # Use only 1000 samples for quick testing
FAST_MODE = False  # Set to False for full training
QUICK_EPOCHS = 2
QUICK_BATCH_SIZE = 8  # Reasonable batch size for memory
CACHE_SAMPLES = False  # Disable caching to avoid memory issues
MAX_MEMORY_GB = 4  # Maximum memory to use for caching
# ---------------- Memory-Efficient Chunk Loader ----------------
class FastChunkedPatchLoader:

    def __init__(self, chunk_dir, verbose=False, cache_limit=None):
        self.chunk_dir = chunk_dir
        self.verbose = verbose
        metadata_path = os.path.join(chunk_dir, "metadata.npy")
        self.metadata = np.load(metadata_path, allow_pickle=True).item()
        self._sample_cache = {}  # Cache individual samples only
        self._cache_size_limit = self._calculate_cache_limit()
        
        if verbose:
            print(f"Sample cache limit: {self._cache_size_limit} samples")
        
    def _calculate_cache_limit(self):
        """Calculate how many samples we can cache based on memory limit"""
        # Each sample: weeks * bands * height * width * 4 bytes (float32)
        sample_size_mb = (self.metadata['num_weeks'] * self.metadata['bands'] * 
                         self.metadata['patch_size'] * self.metadata['patch_size'] * 4) / (1024**2)
        max_samples = int((MAX_MEMORY_GB * 1024) / sample_size_mb)
        return min(max_samples, 500)  # Cap at 500 samples
        
    def preload_samples(self, patch_indices):
        """Pre-load samples efficiently with memory management"""
        if not CACHE_SAMPLES:
            return
            
        # Only cache the most frequent samples
        cache_count = min(len(patch_indices), self._cache_size_limit)
        indices_to_cache = patch_indices[:cache_count]
        
        print(f"Pre-loading {cache_count}/{len(patch_indices)} samples into cache...")
        for idx in tqdm(indices_to_cache):
            if idx not in self._sample_cache:
                self._sample_cache[idx] = self._load_single_patch_direct(idx)
    
    def _load_single_patch_direct(self, patch_idx):
        """Load a single patch directly from disk without chunk caching"""
        patches_per_chunk = self.metadata['patches_per_chunk']
        chunk_idx = patch_idx // patches_per_chunk
        patch_in_chunk = patch_idx % patches_per_chunk
        
        chunk_file = os.path.join(self.chunk_dir, f'chunk_{chunk_idx:03d}.npy')
        
        # Use memory mapping to avoid loading entire chunk
        chunk_data = np.load(chunk_file, mmap_mode='r')
        patch = chunk_data[patch_in_chunk].astype(np.float32)
        
        # Normalize this single patch
        patch = np.clip(patch, -30000, 30000)
        for b in range(patch.shape[0]):
            band = patch[b]
            band_min, band_max = band.min(), band.max()
            if band_max > band_min:
                patch[b] = (band - band_min) / (band_max - band_min)
            else:
                patch[b] = 0
        
        return patch

    def get_patch_data(self, patch_indices, week_idx=None):
        patch_data = []
        for patch_idx in patch_indices:
            # Use cached sample if available
            if patch_idx in self._sample_cache:
                patch = self._sample_cache[patch_idx].copy()
            else:
                patch = self._load_single_patch_direct(patch_idx)
            
            if week_idx is not None:
                patch = patch[week_idx]
            patch_data.append(patch)
        
        return np.array(patch_data, dtype=np.float32)

    def get_single_patch(self, patch_idx, week_idx=None):
        return self.get_patch_data([patch_idx], week_idx)[0]

    @property
    def shape(self):
        return (self.metadata['total_patches'],
                self.metadata['num_weeks'], 
                self.metadata['bands'],
                self.metadata['patch_size'],
                self.metadata['patch_size'])


# ---------------- Simplified Models for Fast Testing ----------------
class SimpleCNN_LSTM(nn.Module):
    """Lighter version for fast experimentation"""
    def __init__(self, input_channels=4, cnn_feature_dim=256, lstm_hidden=128):
        super().__init__()
        # Simpler CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4),  # More aggressive pooling
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(32, cnn_feature_dim)
        self.lstm = nn.LSTM(cnn_feature_dim, lstm_hidden, batch_first=True)

    def forward(self, x):
        batch_size, weeks, bands, H, W = x.shape
        # Process all weeks at once for efficiency
        x_flat = x.view(batch_size * weeks, bands, H, W)
        cnn_out = self.cnn(x_flat).view(batch_size * weeks, -1)
        cnn_out = self.fc(cnn_out).view(batch_size, weeks, -1)
        _, (h_n, _) = self.lstm(cnn_out)
        return h_n[-1]


class FullCNN_LSTM(nn.Module):
    """Original full model"""
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
        batch_size, weeks, bands, H, W = x.shape
        cnn_out = []
        for t in range(weeks):
            xi = x[:, t]
            fi = self.cnn(xi).view(batch_size, -1)
            fi = self.fc(fi)
            cnn_out.append(fi)
        cnn_out = torch.stack(cnn_out, dim=1)
        _, (h_n, _) = self.lstm(cnn_out)
        return h_n[-1]


class Siamese_Network(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        
    def forward(self, x1, x2):
        emb1 = self.encoder(x1)
        emb2 = self.encoder(x2)
        return emb1, emb2


class AdaptiveLoss(nn.Module):
    """Improved loss with better initialization"""
    def __init__(self, init_margin=1.0, min_margin=0.1, max_margin=10.0):
        super().__init__()
        self.margin = nn.Parameter(torch.tensor(init_margin))
        self.min_margin = min_margin
        self.max_margin = max_margin

    def forward(self, emb1, emb2, label):
        margin = torch.clamp(self.margin, self.min_margin, self.max_margin)
        dist = F.pairwise_distance(emb1, emb2)
        
        # Contrastive loss with better scaling
        pos_loss = label * dist.pow(2)
        neg_loss = (1 - label) * F.relu(margin - dist).pow(2)
        
        return (pos_loss + neg_loss).mean()


# ---------------- Fast Dataset with Caching ----------------
class FastPatchPairsDataset(Dataset):
    def __init__(self, patches, pairs, labels, preload=True):
        self.patches = patches
        self.pairs = pairs
        self.labels = labels
        
        if preload and CACHE_SAMPLES:
            # Get unique patch indices
            unique_indices = np.unique(pairs.flatten())
            self.patches.preload_samples(unique_indices)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        a_idx, b_idx = self.pairs[idx]
        x1 = torch.tensor(self.patches.get_patch_data([a_idx]), dtype=torch.float32)[0]
        x2 = torch.tensor(self.patches.get_patch_data([b_idx]), dtype=torch.float32)[0]
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x1, x2, y


# ---------------- Training Functions ----------------
def fast_evaluation(model, loader, device, max_batches=10):
    """Quick evaluation on subset of data"""
    model.eval()
    distances, labels_eval = [], []
    
    with torch.no_grad():
        for i, (x1, x2, y) in enumerate(loader):
            if i >= max_batches:
                break
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            emb1, emb2 = model(x1, x2)
            d = F.pairwise_distance(emb1, emb2)
            distances.extend(d.cpu().numpy())
            labels_eval.extend(y.cpu().numpy())
    
    distances = np.array(distances)
    labels_eval = np.array(labels_eval)
    
    # Simple threshold = median distance
    threshold = np.median(distances)
    preds = (distances <= threshold).astype(int)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    acc = accuracy_score(labels_eval, preds)
    prec = precision_score(labels_eval, preds, zero_division=0)
    rec = recall_score(labels_eval, preds, zero_division=0)
    
    return acc, prec, rec, threshold


def train_epoch(model, loader, optimizer, loss_fn, device, verbose=True):
    """Train one epoch with progress tracking"""
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(loader, desc="Training") if verbose else loader
    
    for x1, x2, y in pbar:
        # Skip problematic batches
        if y.sum().item() == 0 or y.sum().item() == y.numel():
            continue
            
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        
        optimizer.zero_grad()
        emb1, emb2 = model(x1, x2)
        loss = loss_fn(emb1, emb2, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        num_batches += 1
        
        if verbose and hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return running_loss / max(num_batches, 1)


# ---------------- Main Training Script ----------------
if __name__ == "__main__":
    print("🚀 Starting Fast Training Pipeline")
    
    # Load data
    CHUNK_DIR = "patch_chunks"
    print("Loading chunked patches...")
    patch_loader = FastChunkedPatchLoader(CHUNK_DIR, verbose=True, cache_limit=15)
    print(f"Patch loader initialized! Shape: {patch_loader.shape}")

    pairs = np.load("siamese_week_pairs.npy")
    labels = np.load("siamese_week_pair_labels.npy")
    print(f"Loaded {len(pairs)} pairs")

    # FAST MODE: Use subset for quick testing
    if FAST_MODE:
        print(f"🏃‍♂️ FAST MODE: Using only {QUICK_SAMPLE_SIZE} samples")
        indices = np.random.choice(len(pairs), min(QUICK_SAMPLE_SIZE, len(pairs)), replace=False)
        pairs = pairs[indices]
        labels = labels[indices]

    # Balance classes
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Missing positive or negative samples!")
    
    min_samples = min(len(pos_idx), len(neg_idx))
    pos_balanced = np.random.choice(pos_idx, min_samples, replace=False)
    neg_balanced = np.random.choice(neg_idx, min_samples, replace=False)
    balanced_idx = np.concatenate([pos_balanced, neg_balanced])
    np.random.shuffle(balanced_idx)
    
    pairs = pairs[balanced_idx]
    labels = labels[balanced_idx]
    print(f"Balanced dataset: {len(pairs)} pairs ({labels.sum()} positive, {len(labels)-labels.sum()} negative)")

    # Train/test split
    train_pairs, test_pairs, train_labels, test_labels = train_test_split(
        pairs, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Create datasets
    train_dataset = FastPatchPairsDataset(patch_loader, train_pairs, train_labels, preload=CACHE_SAMPLES)
    test_dataset = FastPatchPairsDataset(patch_loader, test_pairs, test_labels, preload=False)

    batch_size = QUICK_BATCH_SIZE if FAST_MODE else 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Choose model based on mode
    if FAST_MODE:
        encoder = SimpleCNN_LSTM(input_channels=patch_loader.metadata['bands'])
        print("Using SimpleCNN_LSTM for fast training")
    else:
        encoder = FullCNN_LSTM(input_channels=patch_loader.metadata['bands'])
        print("Using FullCNN_LSTM for full training")
    
    model = Siamese_Network(encoder).to(device)
    loss_fn = AdaptiveLoss()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [loss_fn.margin], 
        lr=2e-4 if FAST_MODE else 1e-4,
        weight_decay=1e-5
    )

    # Training loop
    epochs = QUICK_EPOCHS if FAST_MODE else 10
    print(f"Training for {epochs} epochs...")
    
    best_acc = 0
    for epoch in range(epochs):
        print(f"\n📈 Epoch {epoch+1}/{epochs}")
        
        # Train
        start_time = time.time()
        avg_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        train_time = time.time() - start_time
        
        # Quick evaluation
        acc, prec, rec, thresh = fast_evaluation(model, test_loader, device, max_batches=20)
        
        print(f"⏱️  Time: {train_time:.1f}s | Loss: {avg_loss:.4f} | Margin: {loss_fn.margin.item():.3f}")
        print(f"📊 Quick Eval - Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | Thresh: {thresh:.3f}")
        
        # Save if improved
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_state_dict': loss_fn.state_dict(),
                'epoch': epoch,
                'accuracy': acc
            }, f"best_model_{'fast' if FAST_MODE else 'full'}.pth")
            print(f"💾 Saved best model (acc: {acc:.3f})")
        
        # Early stopping for fast mode
        if FAST_MODE and acc > 0.8:
            print("🎯 Good accuracy reached in fast mode!")
            break

    print(f"\n✅ Training completed! Best accuracy: {best_acc:.3f}")
    
    if FAST_MODE:
        print("\n" + "="*50)
        print("🎯 FAST MODE Results Look Good!")
        print("Set FAST_MODE = False to train on full dataset")
        print("="*50)
    else:
        print("\n🎉 Full training completed!")