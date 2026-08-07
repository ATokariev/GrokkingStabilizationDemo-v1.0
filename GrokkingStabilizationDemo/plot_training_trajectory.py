import torch
import os
import matplotlib.pyplot as plt


# ============================================================
# 1. Load Training History
# ============================================================

print("--- Generating Training Trajectory Visualization ---")


history_path = (
    "checkpoints/"
    "training_history.pt"
)


if not os.path.exists(history_path):

    print(
        "Error: training_history.pt not found."
    )

    print(
        "Run train.py first."
    )

    exit()



history = torch.load(
    history_path
)



# ============================================================
# 2. Extract Metrics
# ============================================================

epochs = [
    item["epoch"]
    for item in history
]


train_accuracy = [
    item["train_accuracy"]
    for item in history
]


test_accuracy = [
    item["test_accuracy"]
    for item in history
]


train_loss = [
    item["train_loss"]
    for item in history
]


test_loss = [
    item["test_loss"]
    for item in history
]



# ============================================================
# 3. Identify Behavioral Regimes
# ============================================================

memorization_epoch = None
generalization_epoch = None


for item in history:

    if (
        item["train_accuracy"] > 0.99
        and item["test_accuracy"] < 0.20
    ):

        memorization_epoch = item["epoch"]

        break



for item in history:

    if (
        item["train_accuracy"] > 0.99
        and item["test_accuracy"] > 0.95
    ):

        generalization_epoch = item["epoch"]

        break



# ============================================================
# 4. Create Visualization
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(10, 8),
    sharex=True
)



# ------------------------------------------------------------
# Accuracy trajectory
# ------------------------------------------------------------

axes[0].plot(
    epochs,
    train_accuracy,
    label="Train Accuracy"
)


axes[0].plot(
    epochs,
    test_accuracy,
    label="Test Accuracy"
)


axes[0].set_ylabel(
    "Accuracy"
)


axes[0].set_title(
    "Behavioral Trajectory During Training"
)


axes[0].set_ylim(
    0,
    1.05
)


axes[0].grid(
    True,
    linestyle="--",
    alpha=0.4
)


axes[0].legend()



# ------------------------------------------------------------
# Loss trajectory
# ------------------------------------------------------------

axes[1].plot(
    epochs,
    train_loss,
    label="Train Loss"
)


axes[1].plot(
    epochs,
    test_loss,
    label="Test Loss"
)


axes[1].set_xlabel(
    "Epoch"
)


axes[1].set_ylabel(
    "Loss"
)


axes[1].set_title(
    "Optimization Trajectory"
)


axes[1].grid(
    True,
    linestyle="--",
    alpha=0.4
)


axes[1].legend()



# ============================================================
# 5. Mark Regime Locations
# ============================================================

for ax in axes:

    if memorization_epoch is not None:

        ax.axvline(
            memorization_epoch,
            linestyle="--",
            alpha=0.7
        )


        ax.text(
            memorization_epoch,
            0.05,
            "Memorization\ncheckpoint",
            rotation=90,
            verticalalignment="bottom"
        )


    if generalization_epoch is not None:

        ax.axvline(
            generalization_epoch,
            linestyle="--",
            alpha=0.7
        )


        ax.text(
            generalization_epoch,
            0.05,
            "Generalization\ncheckpoint",
            rotation=90,
            verticalalignment="bottom"
        )



plt.suptitle(
    "Behavioral Transition Across Training",
    fontsize=14
)


plt.tight_layout()



# ============================================================
# 6. Save Result
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)


output_path = (
    "results/"
    "training_trajectory.png"
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
    "Trajectory plot is intended for behavioral regime inspection."
)


plt.show()

