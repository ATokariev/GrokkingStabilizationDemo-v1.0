import yaml
import torch
import os


# ============================================================
# 1. Configuration Loading
# ============================================================

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


CONSISTENCY_THRESHOLD = config["protocol"]["representation_consistency_threshold"]


# ============================================================
# 2. Representation Compatibility Metric
# ============================================================

def linear_cka(X, Y):
    """
    Computes linear Centered Kernel Alignment (CKA)
    between two latent representations.

    CKA is treated as a representation compatibility
    observable, not as an invariant itself.
    """

    X_centered = X - X.mean(dim=0)
    Y_centered = Y - Y.mean(dim=0)

    dot_product = torch.norm(
        X_centered.t() @ Y_centered
    ) ** 2

    norm_x = torch.norm(
        X_centered.t() @ X_centered
    )

    norm_y = torch.norm(
        Y_centered.t() @ Y_centered
    )

    if norm_x == 0 or norm_y == 0:
        return 0.0

    return (
        dot_product / (norm_x * norm_y)
    ).item()



# ============================================================
# 3. Candidate Marker Evaluation
# ============================================================

def evaluate_representation_marker(
        compatibility_score,
        threshold
):

    return {
        "marker_detected":
            compatibility_score >= threshold,

        "criterion":
            threshold
    }



# ============================================================
# 4. Main Analysis Procedure
# ============================================================

def run_representation_consistency_analysis():

    print(
        "--- Protocol Step: "
        "Representation Consistency Analysis ---"
    )


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
            "Run train.py and capture both behavioral regimes first."
        )

        return



    # --------------------------------------------------------
    # Load checkpoint objects
    # --------------------------------------------------------

    mem_checkpoint = torch.load(
        mem_path
    )

    gen_checkpoint = torch.load(
        gen_path
    )


    hidden_mem = (
        mem_checkpoint["hidden_representation"]
    )

    hidden_gen = (
        gen_checkpoint["hidden_representation"]
    )


    # --------------------------------------------------------
    # Compute representation compatibility
    # --------------------------------------------------------

    compatibility_score = linear_cka(
        hidden_mem,
        hidden_gen
    )


    consistency_defect = (
        1.0 - compatibility_score
    )


    marker_result = evaluate_representation_marker(
        compatibility_score,
        CONSISTENCY_THRESHOLD
    )


    analysis_result = {

        "metric":
            "Linear CKA",

        "representation_compatibility":
            compatibility_score,

        "consistency_defect":
            consistency_defect,

        "threshold":
            CONSISTENCY_THRESHOLD,

        "memorization_regime":
            mem_checkpoint["regime"],

        "generalization_regime":
            gen_checkpoint["regime"],

        "marker_detected":
            marker_result["marker_detected"],

        "memorization_epoch":
            mem_checkpoint["epoch"],

        "generalization_epoch":
            gen_checkpoint["epoch"]
    }



    # --------------------------------------------------------
    # Save analysis result
    # --------------------------------------------------------

    torch.save(
        analysis_result,
        "checkpoints/representation_consistency_analysis.pt"
    )


    # --------------------------------------------------------
    # Reporting
    # --------------------------------------------------------

    print(
        f"Representation compatibility (CKA): "
        f"{compatibility_score:.4f}"
    )

    print(
        f"Consistency defect: "
        f"{consistency_defect:.4f}"
    )

    print(
        f"Decision threshold: "
        f"{CONSISTENCY_THRESHOLD}"
    )


    if marker_result["marker_detected"]:

        print(
            "\n✅ VERDICT: "
            "Candidate representation marker retained."
        )

        print(
            "The observed representation consistency "
            "is compatible with the candidate structural interpretation."
        )

    else:

        print(
            "\n❌ VERDICT: "
            "Candidate representation marker not detected."
        )

        print(
            "Alternative explanations remain possible."
        )



if __name__ == "__main__":

    run_representation_consistency_analysis()
