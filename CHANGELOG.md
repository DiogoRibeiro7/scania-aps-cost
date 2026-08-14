# Changelog

## 0.2.0

Expanded the project from the original logistic/XGBoost baseline into a full real-data machine-learning study.

- Added Linear SVM, Random Forest, Extra Trees, LightGBM, PyTorch MLP and autoencoder classifier.
- Added SGD/Adam/AdamW, schedules, weight decay, dropout, batch normalization, gradient clipping and early stopping experiments.
- Added class weighting, random undersampling, SMOTE and focal-loss imbalance strategies.
- Added dedicated probability calibration with sigmoid/Platt and isotonic regression.
- Added arbitrary-score threshold optimization so SVM margins can be cost-optimized without pretending to be probabilities.
- Added L1, mutual-information, RFE and tree-based feature-selection studies.
- Added permutation/tree importance and optional SHAP utilities.
- Added XGBoost component ablations.
- Added a four-way fit/tune/calibration/threshold development split.
- Added reproducible multi-model study runners and CLI commands.
- Added 13 experiment notebooks and expanded methodology documentation.
