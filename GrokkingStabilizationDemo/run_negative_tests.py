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
HIDDEN_DIM = config["model"]["hidden_dim"]
LR = config["model"]["learning_rate"]
WD = config["model"]["weight_decay"]

CONSISTENCY_THRESHOLD = (
    config["protocol"]["representation_consistency_threshold"]
)


# ============================================================
# 2. Corrupted Task Generation
# ============================================================

def generate_corrupted_data(p):

    """
    Negative control:
    preserve input distribution while destroying
    input-output algorithmic structure.
    """

    X, Y = [], []

    for a in range(p):
        for b in range(p):

            x = torch.zeros(2 * p)

            x[a] = 1.0
            x[p + b] = 1.0

            # Destroy modular relation
            y = random.randint(0, p - 1)

            X.append(x)
            Y.append(y)

    return (
        torch.stack(X),
        torch.tensor(Y, dtype=torch.long)
    )


# ============================================================
# 3. Representation Metric
# ============================================================

def linear_cka(X, Y):

    X_c = X - X.mean(dim=0)
    Y_c = Y - Y.mean(dim=0)

    dot_product = torch.norm(
        X_c.t() @ Y_c
    ) ** 2

    norm_x = torch.norm(
        X_c.t() @ X_c
    )

    norm_y = torch.norm(
        Y_c.t() @ Y_c
    )

    if norm_x == 0 or norm_y == 0:
        return 0.0

    return (
        dot_product / (norm_x * norm_y)
    ).item()



# ============================================================
# 4. Negative Control Experiment
# ============================================================

print(
    "--- Negative Control: "
    "Task Structure Disruption ---"
)


X_noise, Y_noise = generate_corrupted_data(P)


model = nn.Sequential(

    nn.Linear(2 * P, HIDDEN_DIM),
    nn.ReLU(),
    nn.Linear(HIDDEN_DIM, P)

)


criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WD
)


print(
    "Training model under disrupted task structure..."
)


for epoch in range(1, 1501):

    model.train()

    optimizer.zero_grad()

    logits = model(X_noise)

    loss = criterion(
        logits,
        Y_noise
    )

    loss.backward()

    optimizer.step()


    if epoch % 500 == 0:

        accuracy = (
            logits.argmax(dim=1) == Y_noise
        ).float().mean().item()


        print(
            f"Epoch {epoch:04d} | "
            f"Noise Train Accuracy: {accuracy:.4f}"
        )



# ============================================================
# 5. Extract Negative Representation
# ============================================================

with torch.no_grad():

    hidden_negative = model[1](
        model[0](X_noise)
    )



# ============================================================
# 6. Load Reference Representation
# ============================================================

reference_path = (
    "checkpoints/"
    "hidden_generalization.pt"
)


if not os.path.exists(reference_path):

    print(
        "Reference generalization checkpoint not found."
    )

    print(
        "Run train.py before executing negative tests."
    )

    exit()



reference_checkpoint = torch.load(
    reference_path
)


hidden_reference = (
    reference_checkpoint["hidden_representation"]
)



# ============================================================
# 7. Evaluate Candidate Marker
# ============================================================

min_size = min(
    hidden_negative.shape[0],
    hidden_reference.shape[0]
)


compatibility_score = linear_cka(
    hidden_negative[:min_size],
    hidden_reference[:min_size]
)



marker_detected = (
    compatibility_score >= CONSISTENCY_THRESHOLD
)



# ============================================================
# 8. Report
# ============================================================

print(
    "\n--- Negative Control Results ---"
)


print(
    f"Representation compatibility (CKA): "
    f"{compatibility_score:.4f}"
)


print(
    f"Decision threshold: "
    f"{CONSISTENCY_THRESHOLD}"
)


if marker_detected:

    print(
        "\n❌ NEGATIVE TEST FAILED"
    )

    print(
        "The candidate representation marker "
        "persists despite disrupted task structure."
    )


else:

    print(
        "\n✅ NEGATIVE TEST PASSED"
    )

    print(
        "The candidate representation marker "
        "is not detected under disrupted task structure."
    )



# ============================================================
# 9. Save Result
# ============================================================

negative_result = {

    "test_type":
        "corrupted_labels",

    "metric":
        "Linear CKA",

    "representation_compatibility":
        compatibility_score,

    "threshold":
        CONSISTENCY_THRESHOLD,

    "marker_detected":
        marker_detected
}


torch.save(
    negative_result,
    "checkpoints/negative_test_result.pt"
)
