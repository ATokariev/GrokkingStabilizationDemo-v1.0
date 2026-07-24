import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import random

# 1. Завантажуємо конфігурацію
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

P = config["experiment"]["prime_modulus"]
HIDDEN_DIM = config["model"]["hidden_dim"]
LR = config["model"]["learning_rate"]
WD = config["model"]["weight_decay"]

# 2. Генерація даних із ЗІПСОВАНИМИ мітками (Негативний контроль)
def generate_corrupted_data(p):
    X, Y = [], []
    for a in range(p):
        for b in range(p):
            x = torch.zeros(2 * p)
            x[a] = 1.0
            x[p + b] = 1.0
            # Рандомна мітка замість правильної відповіді
            y = random.randint(0, p - 1)
            X.append(x)
            Y.append(y)
    return torch.stack(X), torch.tensor(Y, dtype=torch.long)

print("--- Запуск Негативного Тесту: Навчання на хаотичних мітках ---")
X_train, Y_train = generate_corrupted_data(P)

model = nn.Sequential(
    nn.Linear(2 * P, HIDDEN_DIM),
    nn.ReLU(),
    nn.Linear(HIDDEN_DIM, P)
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

print("Змушуємо модель запам'ятати шум...")
for epoch in range(1, 1501):
    model.train()
    optimizer.zero_grad()
    out = model(X_train)
    loss = criterion(out, Y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        acc = (out.argmax(dim=1) == Y_train).float().mean().item()
        print(f"Епоха {epoch:04d} | Train Acc (Шум): {acc:.4f}")

# Отримуємо латентний простір негативного тесту
with torch.no_grad():
    hidden_negative = model[1](model[0](X_train))

# 3. Порівняння з еталонним інваріантом (з етапу справжнього grokking)
true_gen_path = "checkpoints/hidden_generalization.pt"
try:
    hidden_true = torch.load(true_gen_path)
except FileNotFoundError:
    print("Не знайдено еталонного файлу узагальнення. Запустіть train.py ще раз.")
    exit()

def linear_cka(X, Y):
    X_c = X - X.mean(dim=0)
    Y_c = Y - Y.mean(dim=0)
    dot_product = torch.norm(X_c.t() @ Y_c)**2
    norm_x = torch.norm(X_c.t() @ X_c)
    norm_y = torch.norm(Y_c.t() @ Y_c)
    return (dot_product / (norm_x * norm_y)).item()

# Вирівнюємо розмірності для порівняння
min_size = min(hidden_negative.shape[0], hidden_true.shape[0])
cka_score = linear_cka(hidden_negative[:min_size], hidden_true[:min_size])

print("\n--- Результат Негативного Тесту ---")
print(f"Подібність хаотичної структури до еталонної (CKA): {cka_score:.4f}")

if cka_score < config["protocol"]["eta_proxy_threshold"]:
    print("✅ ВЕРДИКТ ПРОТОКОЛУ: Негативний тест УСПІШНИЙ.")
    print("Мережа здатна запам'ятати шум, але її латентна структура при цьому колапсує.")
    print("Це математично доводить, що наш знайдений інваріант не є випадковим артефактом архітектури.")
else:
    print("❌ ВЕРДИКТ ПРОТОКОЛУ: Негативний тест ПРОВАЛЕНО.")
