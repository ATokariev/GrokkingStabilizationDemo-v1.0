import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os


# ============================================================
# 1. Configuration Loading
# ============================================================

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

P = config["experiment"]["prime_modulus"]
TRAIN_RATIO = config["experiment"]["train_split_ratio"]
SEED = config["experiment"]["random_seed"]

HIDDEN_DIM = config["model"]["hidden_dim"]
LR = config["model"]["learning_rate"]
WD = config["model"]["weight_decay"]
EPOCHS = config["model"]["max_epochs"]


# ============================================================
# 2. Reproducibility Control
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 3. Dataset Generation
# ============================================================

def generate_data(p):
    """
    Exhaustive modular addition dataset.

    Task:
        y = (a + b) mod p
    """

    X, Y = [], []

    for a in range(p):
        for b in range(p):

            x = torch.zeros(2 * p)

            x[a] = 1.0
            x[p + b] = 1.0

            y = (a + b) % p

            X.append(x)
            Y.append(y)

    return torch.stack(X), torch.tensor(Y, dtype=torch.long)


print(f"Generating data for prime modulus p={P}...")

X_all, Y_all = generate_data(P)


# ============================================================
# 4. Train / Test Split
# ============================================================

indices = list(range(len(X_all)))

random.shuffle(indices)

split_idx = int(len(indices) * TRAIN_RATIO)

train_idx = indices[:split_idx]
test_idx = indices[split_idx:]


X_train = X_all[train_idx]
Y_train = Y_all[train_idx]

X_test = X_all[test_idx]
Y_test = Y_all[test_idx]


# ============================================================
# 5. Model Architecture
# ============================================================

class ModularMLP(nn.Module):

    def __init__(self, p, hidden_dim):
        super().__init__()

        self.layer1 = nn.Linear(2 * p, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, p)


    def forward(self, x):

        hidden = self.relu(self.layer1(x))

        output = self.layer2(hidden)

        return output, hidden



model = ModularMLP(P, HIDDEN_DIM)

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WD
)


# ============================================================
# 6. Checkpoint Preparation
# ============================================================

os.makedirs("checkpoints", exist_ok=True)

memorization_captured = False
generalization_captured = False

training_history = []


# ============================================================
# Helper: Save Representation Checkpoint
# ============================================================

def save_regime_checkpoint(
        path,
        hidden,
        regime,
        epoch,
        train_accuracy,
        test_accuracy,
        train_loss,
        test_loss
):

    checkpoint = {

        # Representation
        "hidden_representation": hidden,

        # Behavioral regime metadata
        "regime": regime,

        # Training state metadata
        "epoch": epoch,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_loss": train_loss,
        "test_loss": test_loss,

        # Experimental metadata
        "seed": SEED,
        "prime_modulus": P,
        "hidden_dimension": HIDDEN_DIM
    }

    torch.save(checkpoint, path)



# ============================================================
# 7. Training Loop
# ============================================================

print(
    "Starting training. Searching for behaviorally identified "
    "memorization and generalization regimes..."
)


for epoch in range(1, EPOCHS + 1):

    model.train()

    optimizer.zero_grad()


    out_train, _ = model(X_train)

    loss_train = criterion(
        out_train,
        Y_train
    )

    loss_train.backward()

    optimizer.step()



    if epoch % 100 == 0 or epoch == EPOCHS:

        model.eval()

        with torch.no_grad():

            out_test, hidden_test = model(X_test)

            loss_test = criterion(
                out_test,
                Y_test
            )


            acc_train = (
                out_train.argmax(dim=1) == Y_train
            ).float().mean().item()


            acc_test = (
                out_test.argmax(dim=1) == Y_test
            ).float().mean().item()



        training_history.append(
            {
                "epoch": epoch,
                "train_accuracy": acc_train,
                "test_accuracy": acc_test,
                "train_loss": loss_train.item(),
                "test_loss": loss_test.item()
            }
        )


        print(
            f"Epoch {epoch:04d} | "
            f"Train Acc: {acc_train:.4f} | "
            f"Test Acc: {acc_test:.4f}"
        )


        # ----------------------------------------------------
        # Memorization regime
        # ----------------------------------------------------

        if (
            acc_train > 0.99
            and acc_test < 0.20
            and not memorization_captured
        ):

            save_regime_checkpoint(
                "checkpoints/hidden_memorization.pt",
                hidden_test,
                "memorization",
                epoch,
                acc_train,
                acc_test,
                loss_train.item(),
                loss_test.item()
            )

            print(
                "--- Memorization regime captured ---"
            )

            memorization_captured = True



        # ----------------------------------------------------
        # Generalization regime
        # ----------------------------------------------------

        if (
            acc_train > 0.99
            and acc_test > 0.95
            and not generalization_captured
        ):

            save_regime_checkpoint(
                "checkpoints/hidden_generalization.pt",
                hidden_test,
                "generalization",
                epoch,
                acc_train,
                acc_test,
                loss_train.item(),
                loss_test.item()
            )


            print(
                "--- Generalization regime captured ---"
            )

            generalization_captured = True

            break



# ============================================================
# 8. Save Training History
# ============================================================

torch.save(
    training_history,
    "checkpoints/training_history.pt"
)


print(
    "Training complete. "
    "Proceeding to representation consistency analysis."
)
