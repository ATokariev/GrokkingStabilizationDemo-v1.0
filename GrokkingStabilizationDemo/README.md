# Grokking Stabilization Demo

This repository contains a minimal reproducibility package (Proof-of-Concept) to demonstrate the operability of the **Stabilization Spectrum** framework. Using the modular addition task ($p = 97$) as an example, it demonstrates the epistemic filtration of a phase transition (grokking) and the verification of invariants using cross-channel consistency (CKA) and negative tests.

## Full Step-by-Step Reproducibility Guide

### Step 1: Environment Setup and Dependency Installation
Open the project in Visual Studio or a terminal, create a local Python virtual environment, and install the packages from the dependencies file:

```bash
pip install -r requirements.txt
```
### Step 2: Model Training and Phase Transition Capture (train.py)Run the training script, which will generate data for modular addition ($p = 97$), train an MLP neural network, and save latent space checkpoints for the memorization and generalization stages:
```bash
python train.py
```
### Step 3: Verification of Invariants and the $\eta$-proxy (compute_invariants.py)Run the cross-channel consistency analysis to compute the CKA metric and verify the consistency defect against the registered threshold:
```bash
python compute_invariants.py
```
### Step 4: Execution of Negative Controls (run_negative_tests.py)
Perform a check using corrupted (noisy) labels to confirm that the latent geometry collapses in the absence of a genuine data structure:
```bash
python run_negative_tests.py
```
### (Optional)
To generate a visual comparison of the latent space using PCA, run:
```bash
python visualize_latent.py
```
