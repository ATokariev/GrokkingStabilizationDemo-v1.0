import yaml
import torch
import os

# Завантажуємо конфігурацію
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

ETA_THRESHOLD = config["protocol"]["eta_proxy_threshold"]

def linear_cka(X, Y):
    """
    Обчислює лінійний Centered Kernel Alignment (CKA) між двома матрицями активацій.
    CKA = 1.0 означає ідентичну геометрію, CKA = 0.0 - абсолютно ортогональну.
    """
    # Центрування матриць по стовпцях
    X_centered = X - X.mean(dim=0)
    Y_centered = Y - Y.mean(dim=0)
    
    # Обчислення скалярного добутку (подібності) та норм
    dot_product = torch.norm(X_centered.t() @ Y_centered)**2
    norm_x = torch.norm(X_centered.t() @ X_centered)
    norm_y = torch.norm(Y_centered.t() @ Y_centered)
    
    return (dot_product / (norm_x * norm_y)).item()

def run_invariant_check():
    print("--- Крок протоколу: Аналіз міжканальної узгодженості (η-проксі) ---")
    
    mem_path = "checkpoints/hidden_memorization.pt"
    gen_path = "checkpoints/hidden_generalization.pt"
    
    if not os.path.exists(mem_path) or not os.path.exists(gen_path):
        print("Помилка: Файли станів не знайдені. Переконайтеся, що train.py досяг порогу 0.95.")
        return

    # Завантажуємо латентні стани моделі
    hidden_mem = torch.load(mem_path)
    hidden_gen = torch.load(gen_path)
    
    # Обчислюємо дефект узгодженості
    cka_score = linear_cka(hidden_mem, hidden_gen)
    eta_proxy = 1.0 - cka_score # Дефект узгодженості обернено пропорційний подібності
    
    print(f"Подібність латентної геометрії (CKA): {cka_score:.4f}")
    print(f"Дефект узгодженості (η-proxy): {eta_proxy:.4f}")
    print(f"Цільовий поріг узгодженості: {ETA_THRESHOLD}")
    
    if cka_score >= ETA_THRESHOLD:
        print("\n✅ ВЕРДИКТ: Структура інваріантна. Кандидатна інтерпретація ЗБЕРІГАЄТЬСЯ.")
        print("Геометрія латентного простору стабілізувалася і відповідає цільовому алгоритму.")
    else:
        print("\n❌ ВЕРДИКТ: Інваріант зруйновано. Перехід до резервного пояснення.")
        print("Хоча точність висока, внутрішня структура хаотична.")

if __name__ == "__main__":
    run_invariant_check()
