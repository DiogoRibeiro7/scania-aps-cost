# Scania APS Cost-Sensitive Machine Learning

[![CI](https://github.com/DiogoRibeiro7/scania-aps-cost/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/scania-aps-cost/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://docs.astral.sh/ruff/)
[![Checked with mypy](https://img.shields.io/badge/types-mypy%20--strict-2a6db2)](https://mypy-lang.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21940301.svg)](https://doi.org/10.5281/zenodo.21940301)

A real-data machine-learning study of failure detection in Scania trucks. The repository uses the UCI **APS Failure at Scania Trucks** dataset to investigate how regularization, optimization, imbalance handling, calibration, representation learning, feature selection and decision thresholds affect a concrete maintenance decision.

The project is not organized around finding the model with the highest accuracy. The industrial objective is

\[
\boxed{J(\theta,\tau)=10\,FP(\theta,\tau)+500\,FN(\theta,\tau)}
\]

where \(\theta\) denotes learned model parameters and \(\tau\) is the operating threshold.

## Why this problem

The official data has the properties that make the technical details of machine learning matter:

- 60,000 training trucks and a separate 16,000-truck test set;
- 171 anonymized operational features;
- severe class imbalance;
- substantial missingness;
- a false negative is 50 times more expensive than a false positive;
- nonlinear interactions may matter;
- probabilities and margins must be converted into an operational decision;
- the feature semantics are anonymized, forcing the modelling choices to stand on statistical evidence.

The central question is:

> Given incomplete and strongly imbalanced truck telemetry, which combination of representation, loss, regularization, optimizer, calibration and decision rule minimizes the real cost of detecting APS failures?

## Implemented model families

| Model | Main technical questions |
|---|---|
| Logistic Regression | L1, L2, Elastic Net, sparsity, coefficient stability, class weighting |
| Linear SVM | hinge loss, margin regularization through \(C\), class weighting, margin threshold |
| Random Forest | bagging, depth, leaf-size and feature-subsampling regularization |
| Extra Trees | stronger randomization and bias-variance effects |
| XGBoost | learning rate, depth, child constraints, L1/L2, row/column subsampling, imbalance |
| LightGBM | leaf-wise growth, leaf complexity, L1/L2, sampling and imbalance |
| PyTorch MLP | SGD/Adam/AdamW, schedules, weight decay, dropout, batch norm, clipping, early stopping |
| Autoencoder + classifier | nonlinear representation learning followed by regularized classification |

## Technical studies

### Regularization

The project studies regularization in several different forms rather than treating it as one hyperparameter.

For a linear model,

\[
L_{reg}=L+\lambda_1\lVert\beta\rVert_1+\lambda_2\lVert\beta\rVert_2^2.
\]

For neural networks, regularization includes weight decay, dropout, architecture size and early stopping. For boosted trees it includes L1/L2 leaf penalties, structural constraints, shrinkage and stochastic subsampling.

### Optimization

The MLP exposes **SGD, Adam and AdamW** together with learning-rate schedules, batch size, gradient clipping and early stopping. Epoch-level training and validation loss is retained in `history_`, so convergence can be studied explicitly.

Boosting experiments separately expose the interaction between learning rate and number of boosting rounds.

### Imbalanced learning

The repository implements and compares:

- no imbalance correction;
- class weighting;
- random undersampling;
- SMOTE;
- XGBoost/LightGBM positive-class weighting;
- focal loss for the MLP.

SMOTE and undersampling live **inside fit-only imbalanced-learn pipelines**. Validation, calibration, threshold and test observations are never resampled.

### Probability calibration

The research protocol reserves a dedicated calibration subset and compares:

- uncalibrated scores/probabilities;
- Platt/sigmoid calibration;
- isotonic regression.

Probability models additionally report Brier score and log loss.

### Threshold optimization

Model fitting and action selection are kept separate. Every model receives a cost-optimal threshold learned from a dedicated threshold subset:

\[
\tau^*=\arg\min_\tau\left[10FP(\tau)+500FN(\tau)\right].
\]

For calibrated probabilities, the theoretical Bayes threshold implied by the stated costs is

\[
\tau_B=\frac{10}{10+500}\approx0.0196,
\]

which illustrates why `0.5` is not a defensible default for this decision.

For models such as Linear SVM, the same exact optimizer works directly on margins rather than pretending they are probabilities.

### Feature selection and interpretation

Implemented comparisons include:

- L1 embedded sparsity;
- mutual information;
- recursive feature elimination;
- Extra-Trees-based selection;
- permutation importance;
- tree feature importance;
- optional SHAP values.

### Ablation

The XGBoost ablation removes one component at a time:

- L1/L2 regularization;
- row/column subsampling;
- positive-class weighting;
- cost-optimal thresholding.

This distinguishes improvements in representation and training from improvements caused purely by the deployment decision rule.

## Leakage-safe experimental protocol

The official training set is split into four stratified roles:

1. **fit** — estimate model parameters;
2. **tune** — choose model family hyperparameters;
3. **calibration** — fit Platt/isotonic calibration when requested;
4. **threshold** — choose the maintenance operating threshold.

The official UCI test set is kept outside all four roles.

```text
fit -> tune -> refit -> calibration -> threshold -> official test
```

The older three-way split is retained for the original logistic/XGBoost baseline commands.

## Experiment notebooks

```text
experiments/
├── 01_data_quality_and_missingness.ipynb
├── 02_cost_sensitive_baselines.ipynb
├── 03_logistic_regularization.ipynb
├── 04_svm_margin_regularization.ipynb
├── 05_tree_ensembles.ipynb
├── 06_gradient_boosting.ipynb
├── 07_neural_network_optimization.ipynb
├── 08_imbalance_methods.ipynb
├── 09_probability_calibration.ipynb
├── 10_threshold_optimization.ipynb
├── 11_feature_selection.ipynb
├── 12_ablation_study.ipynb
└── 13_final_model_comparison.ipynb
```

The notebooks call package code and do not contain hard-coded performance numbers.

## Repository structure

```text
scania-aps-cost/
├── src/scania_aps/
│   ├── calibration.py
│   ├── costs.py
│   ├── data.py
│   ├── feature_selection.py
│   ├── metrics.py
│   ├── model_search.py
│   ├── optimization.py
│   ├── resampling.py
│   ├── scoring.py
│   ├── split.py
│   ├── studies.py
│   └── models/
│       ├── autoencoder.py
│       ├── boosted.py
│       ├── factory.py
│       ├── lightgbm.py
│       ├── logistic.py
│       ├── mlp.py
│       ├── svm.py
│       └── trees.py
├── experiments/
├── tests/
├── docs/
├── data/
├── artifacts/
└── pyproject.toml
```

## Installation

Base package:

```bash
poetry install
```

Full research environment:

```bash
poetry install --with boost,neural,imbalance,explain,notebooks
```

The optional groups are deliberately separated so a simple logistic/SVM experiment does not require PyTorch, LightGBM or SHAP.

## Download the real data

```bash
poetry run scania-aps download
```

The original UCI data is downloaded at runtime and is not committed to the repository.

## Run the studies

Original baselines:

```bash
poetry run scania-aps train-logistic --trials 36
poetry run scania-aps train-boosted
```

Compare all model families:

```bash
poetry run scania-aps run-study --profile quick
poetry run scania-aps run-study --profile full
```

Run selected model families:

```bash
poetry run scania-aps run-study --models logistic linear_svm xgboost mlp --profile full
```

Calibration:

```bash
poetry run scania-aps study-calibration --model xgboost
```

Imbalance methods:

```bash
poetry run scania-aps study-imbalance
```

Feature selection:

```bash
poetry run scania-aps study-features
```

Ablation:

```bash
poetry run scania-aps study-ablation
```

Each study writes machine-readable CSV/JSON results under `artifacts/`.

## Metrics

The primary metric is **maintenance cost**. The repository also reports:

- false positives;
- false negatives;
- cost per truck;
- recall;
- precision;
- PR-AUC;
- ROC-AUC;
- Brier score for probability models;
- log loss for probability models;
- savings relative to always-negative and always-positive policies.

Accuracy is intentionally not the optimization target.

## Reproducibility

The package uses fixed random seeds for candidate generation, splits and stochastic estimators where supported. Tests cover the cost function, parsing, split isolation, score handling, model builders and the neural estimators.

```bash
poetry run pytest
```

## Documentation

- `docs/methodology.md` — statistical protocol and modelling rationale.
- `docs/experiments.md` — experiment matrix and ablations.
- `docs/neural_optimization.md` — optimizer and regularization study for the MLP.

## Dataset

UCI Machine Learning Repository: **APS Failure at Scania Trucks**, dataset ID 421.

The code is MIT licensed. The dataset remains subject to its original UCI/Scania terms.

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) first — it lists
the research rules that reviews enforce, in particular that the official UCI test
set is never used for any modelling decision and that resampling stays fit-only.

- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)
- [Release process and Zenodo archival](docs/releasing.md)

## Citation

If you use this software or its results, please cite it. Machine-readable
metadata lives in [CITATION.cff](CITATION.cff), which GitHub renders as a
"Cite this repository" button, and in [.zenodo.json](.zenodo.json), which
supplies the deposit metadata when a release is archived.

The concept DOI [10.5281/zenodo.21940301](https://doi.org/10.5281/zenodo.21940301) always resolves to
the newest version. Cite a specific version's own DOI when you need the exact
code a result came from.

```bibtex
@software{ribeiro_scania_aps_cost,
  author  = {Ribeiro, Diogo},
  title   = {scania-aps-cost: Cost-Sensitive Machine Learning on Scania APS Failure Data},
  version = {0.2.2},
  doi     = {10.5281/zenodo.21940301},
  year    = {2026},
  url     = {https://github.com/DiogoRibeiro7/scania-aps-cost},
  orcid   = {0009-0001-2022-7072}
}
```

Please cite the dataset separately as the UCI Machine Learning Repository
*APS Failure at Scania Trucks* (dataset 421); this repository does not
redistribute it.
