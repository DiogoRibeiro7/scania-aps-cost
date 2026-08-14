# Experimental programme

This repository uses one real operational prediction problem to study several machine-learning mechanisms under a common decision objective.

## Central question

Given incomplete, highly imbalanced truck telemetry, which combination of representation, loss, regularization, optimizer, calibration and decision threshold minimizes the cost of APS failure detection?

The experimental chain is

\[
X\rightarrow\text{preprocessing}\rightarrow\text{representation}\rightarrow
\text{training loss}\rightarrow\text{regularization}\rightarrow
\text{optimizer}\rightarrow\text{calibration}\rightarrow\text{decision}.
\]

## Model families

- Logistic regression: near-unregularized, L1, L2 and Elastic Net.
- Linear SVM: hinge loss, margin regularization through `C`, class weighting.
- Random Forest: bagging, depth, leaf-size and feature-subsampling regularization.
- Extra Trees: stronger split randomization and bias-variance comparison.
- XGBoost: shrinkage, depth, child constraints, L1/L2 and row/column subsampling.
- LightGBM: leaf-wise growth, leaf complexity, L1/L2, sampling and class weighting.
- MLP: SGD/Adam/AdamW, schedules, weight decay, dropout, batch norm, batch size, clipping and early stopping.
- Autoencoder + logistic classifier: unsupervised latent representation followed by regularized classification.

## Imbalance study

The repository compares no correction, positive-class weighting, random undersampling, SMOTE and focal loss. Sampling is performed only inside the training pipeline. Validation, calibration, threshold and test rows are never resampled.

## Calibration study

The research split reserves a dedicated calibration subset. Platt/sigmoid and isotonic calibration are therefore compared without reusing threshold-selection observations. Brier score and log loss are reported alongside the maintenance cost.

## Feature-selection study

The implemented comparisons include L1 embedded selection, mutual information, recursive feature elimination and Extra-Trees-based selection. Separate utilities expose coefficient sparsity, permutation importance, tree importance and optional SHAP values.

## Ablations

The XGBoost ablation removes, one at a time:

1. L1/L2 leaf penalties;
2. row/column subsampling;
3. positive-class weighting;
4. cost-optimal threshold selection.

This separates gains due to the predictive model from gains due only to the deployment decision rule.

## Reproducible result tables

Each study writes CSV/JSON artifacts under `artifacts/`. No result is hard-coded in notebooks or documentation. The official UCI test set is used for final evaluation only; model and threshold choices are made on the official training data.
