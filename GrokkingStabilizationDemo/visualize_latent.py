import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# 1. Завантажуємо конфігурацію
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print("--- Генерація візуалізації латентного простору ---")

mem_path = "checkpoints/hidden_memorization.pt"
gen_path = "checkpoints/hidden_generalization.pt"

if not os.path.exists(mem_path) or not os.path.exists(gen_path):
    print("Помилка: Файли станів не знайдені. Спочатку запустіть train.py.")
    exit()

# Завантажуємо збережені тензори активтацій
hidden_mem = torch.load(mem_path).cpu().numpy()
hidden_gen = torch.load(gen_path).cpu().numpy()

# 2. Зниження розмірності за допомогою PCA (метод головних компонент) до 2D
pca = PCA(n_components=2)

# Проектуємо латентні простори обох етапів
mem_2d = pca.fit_transform(hidden_mem)
gen_2d = pca.fit_transform(hidden_gen)

# 3. Побудова графіків для порівняння
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Графік 1: Етап запам'ятовування (хаос / перенавчення)
axes[0].scatter(mem_2d[:, 0], mem_2d[:, 1], alpha=0.6, c='crimson', s=15)
axes[0].set_title("Етап 1: Запам'ятовування\n(Train Acc: 100%, Test Acc ~2%)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Головна компонента 1")
axes[0].set_ylabel("Головна компонента 2")
axes[0].grid(True, linestyle='--', alpha=0.5)

# Графік 2: Етап узагальнення / grokking (структурований порядок)
axes[1].scatter(gen_2d[:, 0], gen_2d[:, 1], alpha=0.6, c='dodgerblue', s=15)
axes[1].set_title("Етап 2: Узагальнення (Grokking)\n(Test Acc > 95%, CKA = 0.47)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Головна компонента 1")
axes[1].set_ylabel("Головна компонента 2")
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.suptitle("Фазовий перехід латентної геометрії (Спектр стабілізації)", fontsize=16, y=0.98)
plt.tight_layout()

print("Візуалізацію побудовано! Закрийте вікно графіку, щоб завершити роботу скрипта.")
plt.show()
