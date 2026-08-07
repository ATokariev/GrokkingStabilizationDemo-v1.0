import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


# ============================================================
# 1. Configuration Loading
# ============================================================

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


print(
    "--- Generating Latent Representation Visualization ---"
)


# ============================================================
# 2. Load Representation Checkpoints
# ============================================================

mem_path = (
    "checkpoints/"
    "hidden_memorization.pt"
)

gen_path = (
    "checkpoints/"
    "hidden_generalization.pt"
)


if not os.path.exists(mem_path) or not os.path.exists(gen_path):

    print(
        "Error: Representation checkpoints not found."
    )

    print(
        "Run train.py first."
    )

    exit()



mem_checkpoint = torch.load(
    mem_path
)

gen_checkpoint = torch.load(
    gen_path
)



hidden_mem = (
    mem_checkpoint["hidden_representation"]
    .cpu()
    .numpy()
)


hidden_gen = (
    gen_checkpoint["hidden_representation"]
    .cpu()
    .numpy()
)



# ============================================================
# 3. Metadata Extraction
# ============================================================

mem_epoch = mem_checkpoint["epoch"]
gen_epoch = gen_checkpoint["epoch"]


mem_train_acc = mem_checkpoint["train_accuracy"]
mem_test_acc = mem_checkpoint["test_accuracy"]

gen_train_acc = gen_checkpoint["train_accuracy"]
gen_test_acc = gen_checkpoint["test_accuracy"]



# ============================================================
# 4. Shared PCA Projection
# ============================================================

"""
Important:
Both representations are projected into the same PCA basis.

Independent PCA fits would create unrelated coordinate systems
and would not allow direct qualitative comparison.
"""


combined_hidden = np.concatenate(
    [
        hidden_mem,
        hidden_gen
    ],
    axis=0
)


pca = PCA(
    n_components=2
)


combined_2d = pca.fit_transform(
    combined_hidden
)



mem_2d = combined_2d[
    :len(hidden_mem)
]


gen_2d = combined_2d[
    len(hidden_mem):
]



# ============================================================
# 5. Visualization
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)



# Memorization regime

axes[0].scatter(
    mem_2d[:, 0],
    mem_2d[:, 1],
    alpha=0.6,
    s=15
)


axes[0].set_title(
    "Memorization Regime\n"
    f"Epoch {mem_epoch} | "
    f"Train {mem_train_acc:.3f} | "
    f"Test {mem_test_acc:.3f}"
)


axes[0].set_xlabel(
    "Principal Component 1"
)

axes[0].set_ylabel(
    "Principal Component 2"
)


axes[0].grid(
    True,
    linestyle="--",
    alpha=0.4
)



# Generalization regime

axes[1].scatter(
    gen_2d[:, 0],
    gen_2d[:, 1],
    alpha=0.6,
    s=15
)


axes[1].set_title(
    "Generalization Regime\n"
    f"Epoch {gen_epoch} | "
    f"Train {gen_train_acc:.3f} | "
    f"Test {gen_test_acc:.3f}"
)


axes[1].set_xlabel(
    "Principal Component 1"
)

axes[1].set_ylabel(
    "Principal Component 2"
)


axes[1].grid(
    True,
    linestyle="--",
    alpha=0.4
)



plt.suptitle(
    "Latent Representation Reorganization Across Behavioral Regimes",
    fontsize=14
)


plt.tight_layout()



# ============================================================
# 6. Save Visualization
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


output_path = (
    "results/"
    "latent_representation_reorganization.png"
)


plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)


print(
    f"Visualization saved to: {output_path}"
)


print(
    "PCA visualization is intended for qualitative inspection "
    "and is not used as the primary structural criterion."
)


plt.show()
