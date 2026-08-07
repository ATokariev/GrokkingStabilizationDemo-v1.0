# Experiment Pre-registration:
# Representation Reorganization during Grokking

## 1. Hypothesis

Grokking in modular arithmetic neural networks is accompanied by a measurable
reorganization of latent representations.

This experiment tests whether a CKA-based representation compatibility
observable changes between behaviorally identified memorization and
generalization checkpoints under controlled conditions.

The objective is to evaluate whether behavioral transition is associated with
measurable changes in hidden-layer representation geometry.

---

## 2. Channels and Diagnostic Slices

Measurement Channel:

Hidden layer activations.

Representation Metric:

Linear CKA used as a representation compatibility observable.

Candidate Marker:

A registered representation-compatibility criterion evaluated across predefined
experimental variations.

Decision Threshold:

Thresholds are fixed before analysis.

---

## 3. Negative Tests

Negative controls disrupt task structure while preserving the experimental
pipeline.

Failure criterion:

The candidate representation marker should disappear under disrupted task
structure.

---

## 4. Fallback Explanations

Failure to satisfy the criterion indicates that the observed marker may be
explained by optimization dynamics, architectural bias, or measurement artifacts
rather than a task-dependent representation effect.

---

*Registration Date:* 2026-07-24

*Version Status:* Immutable
