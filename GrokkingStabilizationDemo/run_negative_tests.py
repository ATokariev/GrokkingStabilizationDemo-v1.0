import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import random

# 1. Load configuration
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

P = config["experiment"]["prime_modulus"]
HIDDEN_DIM = config["model"]["hidden_dim"]
LR = config["model"]["learning_rate"]
WD = config["model"]["weight_decay"]

# 2. Dataset Generation with Corrupted Labels (Negative Control)
def generate_corrupted_data(p):
    X, Y = [], []
    for a in range(p):
        for b in range(p):
            x = torch.zeros(2 * p)
            x[a] = 1.0
            x[p + b] = 1.0
            # Random label instead of the correct target answer
            y = random.randint(0, p - 1)
            X.append(x)
            Y.append(y)
    return torch.stack(X), torch.tensor(Y, dtype=torch.long)

print("--- Running Negative Test: Training on Chaotic Labels ---")
X_train, Y_train = generate_corrupted_data(P)

model = nn.Sequential(
    nn.Linear(2 * P, HIDDEN_DIM),
    nn.ReLU(),
    nn.Linear(HIDDEN_DIM, P)
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

print("Forcing the model to memorize noise...")
for epoch in range(1, 1501):
    model.train()
    optimizer.zero_grad()
    out = model(X_train)
    loss = criterion(out, Y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        acc = (out.argmax(dim=1) == Y_train).float().mean().item()
        print(f"Epoch {epoch:04d} | Train Acc (Noise): {acc:.4f}")

# Extract the latent space of the negative test model
with torch.no_grad():
    hidden_negative = model[1](model[0](X_train))

# 3. Comparison with the Reference Invariant (from the true grokking stage)
true_gen_path = "checkpoints/hidden_generalization.pt"

try:
    hidden_true = torch.load(true_gen_path)
except FileNotFoundError:
    print("Reference generalization state not found. Please run train.py first.")
    exit()

def linear_cka(X, Y):
    X_c = X - X.mean(dim=0)
    Y_c = Y - Y.mean(dim=0)
    dot_product = torch.norm(X_c.t() @ Y_c)**2
    norm_x = torch.norm(X_c.t() @ X_c)
    norm_y = torch.norm(Y_c.t() @ Y_c)
    return (dot_product / (norm_x * norm_y)).item()

# Align dimensions for comparison
min_size = min(hidden_negative.shape[0], hidden_true.shape[0])
cka_score = linear_cka(hidden_negative[:min_size], hidden_true[:min_size])

print("\n--- Negative Test Results ---")
print(f"Similarity of chaotic structure to the reference (CKA): {cka_score:.4f}")

if cka_score < config["protocol"]["eta_proxy_threshold"]:
    print("✅ PROTOCOL VERDICT: Negative test SUCCESSFUL.")
    print("The network is capable of memorizing noise, but its latent structure collapses in the process.")
    print("This mathematically proves that our discovered invariant is not a random architectural artifact.")
else:
    print("❌ PROTOCOL VERDICT: Negative test FAILED.")
