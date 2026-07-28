import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# 1. Load configuration
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print("--- Generating Latent Space Visualization ---")

mem_path = "checkpoints/hidden_memorization.pt"
gen_path = "checkpoints/hidden_generalization.pt"

if not os.path.exists(mem_path) or not os.path.exists(gen_path):
    print("Error: State files not found. Please run train.py first.")
    exit()

# Load saved activation tensors
hidden_mem = torch.load(mem_path).cpu().numpy()
hidden_gen = torch.load(gen_path).cpu().numpy()

# 2. Dimensionality reduction to 2D using PCA (Principal Component Analysis)
pca = PCA(n_components=2)

# Project latent spaces of both stages
mem_2d = pca.fit_transform(hidden_mem)
gen_2d = pca.fit_transform(hidden_gen)

# 3. Plotting for side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Memorization Stage (Chaos / Overfitting)
axes[0].scatter(mem_2d[:, 0], mem_2d[:, 1], alpha=0.6, c='crimson', s=15)
axes[0].set_title("Stage 1: Memorization\n(Train Acc: 100%, Test Acc ~2%)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")
axes[0].grid(True, linestyle='--', alpha=0.5)

# Plot 2: Generalization / Grokking Stage (Structured Order)
axes[1].scatter(gen_2d[:, 0], gen_2d[:, 1], alpha=0.6, c='dodgerblue', s=15)
axes[1].set_title("Stage 2: Generalization (Grokking)\n(Test Acc > 95%, Structural Order)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.suptitle("Phase Transition of Latent Geometry (Stabilization Spectrum)", fontsize=16, y=0.98)
plt.tight_layout()

print("Visualization generated! Close the plot window to terminate the script.")
plt.show()
