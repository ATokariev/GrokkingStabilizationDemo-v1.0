import yaml
import torch
import os

# Load configuration
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

ETA_THRESHOLD = config["protocol"]["eta_proxy_threshold"]

def linear_cka(X, Y):
    """
    Computes the linear Centered Kernel Alignment (CKA) between two activation matrices.
    CKA = 1.0 indicates identical geometry, CKA = 0.0 indicates absolute orthogonality.
    """
    # Center the matrices by columns
    X_centered = X - X.mean(dim=0)
    Y_centered = Y - Y.mean(dim=0)
    
    # Compute the dot product (similarity) and norms
    dot_product = torch.norm(X_centered.t() @ Y_centered)**2
    norm_x = torch.norm(X_centered.t() @ X_centered)
    norm_y = torch.norm(Y_centered.t() @ Y_centered)
    
    return (dot_product / (norm_x * norm_y)).item()

def run_invariant_check():
    print("--- Protocol Step: Cross-Channel Consistency Analysis (η-proxy) ---")
    
    mem_path = "checkpoints/hidden_memorization.pt"
    gen_path = "checkpoints/hidden_generalization.pt"
    
    if not os.path.exists(mem_path) or not os.path.exists(gen_path):
        print("Error: State files not found. Ensure that train.py reached the 0.95 threshold.")
        return

    # Load the latent states of the model
    hidden_mem = torch.load(mem_path)
    hidden_gen = torch.load(gen_path)
    
    # Compute the consistency defect
    cka_score = linear_cka(hidden_mem, hidden_gen)
    eta_proxy = 1.0 - cka_score # Consistency defect is inversely proportional to similarity
    
    print(f"Latent geometry similarity (CKA): {cka_score:.4f}")
    print(f"Consistency defect (η-proxy): {eta_proxy:.4f}")
    print(f"Target consistency threshold: {ETA_THRESHOLD}")
    
    if cka_score >= ETA_THRESHOLD:
        print("\n✅ VERDICT: Structure is invariant. Candidate interpretation RETAINED.")
        print("Latent space geometry has stabilized and corresponds to the target algorithm.")
    else:
        print("\n❌ VERDICT: Invariant destroyed. Reverting to fallback explanation.")
        print("Although accuracy is high, the internal structure is chaotic.")

if __name__ == "__main__":
    run_invariant_check()
