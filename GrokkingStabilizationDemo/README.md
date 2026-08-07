# Grokking Stabilization Demo

This repository contains a minimal reproducibility package (Proof-of-Concept)
for studying latent representation changes during a controlled behavioral
transition in neural networks.

Using modular addition ($p = 97$) as a synthetic task, the experiment
demonstrates an operational procedure for identifying behaviorally distinct
checkpoints, comparing hidden-layer representations using linear CKA, and
evaluating candidate representation markers through negative controls.

The goal of this repository is not to establish a universal mechanism of
grokking, but to provide a reproducible experimental framework for studying
representation reorganization during behavioral transitions.

---

# Full Step-by-Step Reproducibility Guide

## Step 1: Environment Setup and Dependency Installation

Open the project in Visual Studio or a terminal.

Create and activate a Python virtual environment:

```bash
python -m venv env
```
Activate the environment:
Windows:
```bash
.\env\Scripts\activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```

## Step 2: Model Training and Checkpoint Extraction (train.py)
Run:
```bash
python train.py
```
The training script:
generates the modular addition dataset:
\[
(a+b)\mod p
\]

with:
```
p = 97
```

trains a minimal MLP model;
tracks training and test performance;
identifies behaviorally distinct checkpoints;
saves hidden-layer representations and training history.
Generated checkpoints:
```
checkpoints/
|
├── hidden_memorization.pt
├── hidden_generalization.pt
└── training_history.pt
```
The extracted checkpoints correspond to:
memorization-dominated performance;
generalization-capable performance.

## Step 3: Representation Compatibility Analysis (compute_invariants.py)
Run:
```bash
python compute_invariants.py
```
The analysis compares hidden-layer representations using:
\[
CKA(H_1,H_2)
\]where CKA (Centered Kernel Alignment) is used as a representation
compatibility observable.
The script evaluates:
\[
H_{mem}
\quad\text{vs}\quad
H_{gen}
\]and reports whether the candidate representation marker is detected.
The result should be interpreted as:
high CKA → stronger representation compatibility;
lower CKA → increased representation reorganization.
CKA is used as a measurement observable and is not treated as a universal
invariant.
## Step 4: Negative Control Evaluation (run_negative_tests.py)
Run:
```bash
python run_negative_tests.py
```
The negative control preserves:
model architecture;
optimization pipeline;
experimental procedure;
while disrupting the underlying task structure by replacing correct labels with
random labels.
The purpose is to test whether the observed representation marker depends on
the structured task rather than generic optimization dynamics.
Expected outcome:
```
NEGATIVE TEST PASSED
```

indicating that the candidate representation marker is absent under disrupted
task structure.
## Step 5: Visualization
Latent Representation Visualization
Run:
```bash
python visualize_latent.py
```
Generates:
```
results/
└── latent_representation_reorganization.png
```
The visualization uses PCA only for qualitative inspection of latent
representations.
It is not used as the primary structural criterion.
Training Trajectory Visualization
Run:
```bash
python plot_training_trajectory.py
```
Generates:
```
results/
└── training_trajectory.png
```
The plot shows:
training accuracy trajectory;
test accuracy trajectory;
training loss;
test loss;
extracted behavioral checkpoints.
Experimental Results
A detailed summary of the experiment is available in:
```
results_summary.md
```
The main observations are:
The model exhibits behaviorally distinct checkpoints:
\[
\text{memorization-dominated}
\rightarrow
\text{generalization-capable}
\]Hidden-layer representations at these checkpoints show measurable
compatibility differences:
\[
CKA(H_{mem},H_{gen})=0.405
\]The representation marker disappears under task disruption:
\[
CKA(H_{noise},H_{gen})=0.0168
\]
Repository Structure
```
GrokkingStabilizationDemo/

├── train.py
├── compute_invariants.py
├── run_negative_tests.py
├── visualize_latent.py
├── plot_training_trajectory.py
│
├── config.yaml
├── requirements.txt
│
├── checkpoints/
│   ├── hidden_memorization.pt
│   ├── hidden_generalization.pt
│   └── training_history.pt
│
├── results/
│   ├── latent_representation_reorganization.png
│   └── training_trajectory.png
│
├── results_summary.md
└── README.md
```
Scope and Limitations
This repository provides a minimal operational demonstration.
The current experiment is limited to:
one synthetic task;
one model architecture;
one training configuration;
one random seed.
The results do not establish:
a universal explanation of grokking;
a mathematical invariant;
causal necessity of representation changes.
Further validation requires:
multiple seeds;
alternative architectures;
additional algorithmic tasks;
trajectory-level representation analysis.
Status
Current status:
```
Demonstration completed
```
This repository provides a reproducible experimental package for studying
latent representation changes during controlled behavioral transitions.
```
