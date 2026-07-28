# Experiment Pre-registration: Phase Transition and Latent Space Geometry during Grokking

## 1. Hypothesis
The phenomenon of grokking in neural networks trained on modular arithmetic tasks is accompanied by a structural phase transition in the latent space. The transition from a state of "blind memorization" to a state of "generalization" is not a simple smooth optimization process, but constitutes a change in geometric singularity, manifested through a sharp drop in cross-channel similarity (CKA).

## 2. Channels and Diagnostic Slices
- **Measurement Channel:** Hidden layer activations (hidden representations) of the `ModularMLP` model.
- **Metric (η-proxy):** Linear Centered Kernel Alignment (CKA) between the memorization state and the generalization state.
- **Decision Threshold:** A CKA similarity below 0.85 is considered a consistency defect (the invariant is not preserved).

## 3. Negative Tests and Null Model
- **Negative Control:** Training the model with target labels replaced by white noise (uniform random integers).
- **Negative Test Failure Criterion:** The latent structure of the noise model must demonstrate a similarity collapse relative to the reference invariant (CKA < 0.15).

## 4. Fallback Explanations
If the CKA similarity between states remains high despite low test accuracy, or if the negative test does not result in a structural collapse, the candidate interpretation is rejected as an architectural artifact rather than a genuine algorithmic invariant.

*Registration Date:* 2026-07-24  
*Version Status:* Immutable
