import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

# 1. Завантаження конфігурації (аналог appsettings.json)
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

P = config["experiment"]["prime_modulus"]
TRAIN_RATIO = config["experiment"]["train_split_ratio"]
SEED = config["experiment"]["random_seed"]
HIDDEN_DIM = config["model"]["hidden_dim"]
LR = config["model"]["learning_rate"]
WD = config["model"]["weight_decay"]
EPOCHS = config["model"]["max_epochs"]

# Фіксуємо seed для відтворюваності (Крок протоколу щодо контролю пайплайна)
random.seed(SEED)
torch.manual_seed(SEED)

# 2. Генерація датасету (Усі пари a, b для модульного додавання)
def generate_data(p):
    X, Y = [], []
    for a in range(p):
        for b in range(p):
            # One-hot кодування для a та b
            x = torch.zeros(2 * p)
            x[a] = 1.0
            x[p + b] = 1.0
            y = (a + b) % p
            X.append(x)
            Y.append(y)
    return torch.stack(X), torch.tensor(Y, dtype=torch.long)

print(f"Генеруємо дані для p={P}...")
X_all, Y_all = generate_data(P)

# Розбиття на Train / Test
indices = list(range(len(X_all)))
random.shuffle(indices)
split_idx = int(len(indices) * TRAIN_RATIO)
train_idx, test_idx = indices[:split_idx], indices[split_idx:]

X_train, Y_train = X_all[train_idx], Y_all[train_idx]
X_test, Y_test = X_all[test_idx], Y_all[test_idx]

# 3. Архітектура моделі (Мінімальний MLP)
class ModularMLP(nn.Module):
    def __init__(self, p, hidden_dim):
        super().__init__()
        self.layer1 = nn.Linear(2 * p, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, p)

    def forward(self, x):
        hidden = self.relu(self.layer1(x))
        out = self.layer2(hidden)
        return out, hidden # Повертаємо hidden для майбутнього аналізу інваріантів

model = ModularMLP(P, HIDDEN_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

# 4. Основний цикл навчання (Пошук Grokking)
print("Починаємо навчання. Шукаємо розрив між запам'ятовуванням та узагальненням...")

# Папки для збереження станів (щоб потім перевірити інваріанти)
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    
    out_train, _ = model(X_train)
    loss_train = criterion(out_train, Y_train)
    loss_train.backward()
    optimizer.step()
    
    if epoch % 100 == 0 or epoch == EPOCHS:
        model.eval()
        with torch.no_grad():
            out_test, hidden_test = model(X_test)
            loss_test = criterion(out_test, Y_test)
            
            acc_train = (out_train.argmax(dim=1) == Y_train).float().mean().item()
            acc_test = (out_test.argmax(dim=1) == Y_test).float().mean().item()
            
        print(f"Епоха {epoch:04d} | Train Acc: {acc_train:.4f} | Test Acc: {acc_test:.4f}")
        
        # Зберігаємо "зрізи" на ключових етапах для compute_invariants.py
        if acc_train > 0.99 and acc_test < 0.2:
            torch.save(hidden_test, "checkpoints/hidden_memorization.pt")
        if acc_train > 0.99 and acc_test > 0.95:
            torch.save(hidden_test, "checkpoints/hidden_generalization.pt")
            print("🔥 Grokking зафіксовано! Тестова точність стрибнула.")
            break

print("Навчання завершено. Переходимо до перевірки інваріантів.")
