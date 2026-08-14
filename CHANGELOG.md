# Changelog

## 0.2.2 - 2026-08-15

First release archived on Zenodo. The Zenodo-GitHub integration only picks up
releases published after it is switched on, so this release exists partly to
trigger the initial deposit and mint the DOI.

### Changed

- CI no longer triggers on a `develop` branch, which never existed. Feature
  branches are still covered through the `pull_request` trigger.
- `main` is now protected: pull request required, all six CI jobs must pass and
  the branch must be current before merging, with force pushes and deletion
  blocked.

## 0.2.1 - 2026-08-14

Repository infrastructure and code quality. **No reported result changes**: the
candidate-sampling RNG draws are bit-for-bit identical to 0.2.0, and no
modelling logic was altered.

### Added

- `CITATION.cff` and `.zenodo.json` so releases are citable and archivable, both
  carrying the author ORCID.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` and `CODEOWNERS`.
- Issue forms, a pull-request template and a Dependabot configuration that
  groups the scientific stack and holds back major bumps of the heavy optional
  libraries, because those move numerical results.
- `docs/releasing.md` covering the version bump, tag and Zenodo deposit, for
  both the public GitHub integration and the manual private route.
- `.gitattributes` and `.editorconfig`.
- `src/scania_aps/_types.py` with documented aliases for the scikit-learn and
  PyTorch objects that have no static type.
- A PEP 561 `py.typed` marker, so downstream users get this package's types.
- Packaging metadata: classifiers, keywords and project URLs.

### Changed

- CI now runs four jobs: lint and type checking, a core-dependency test matrix
  on Python 3.11–3.13, a full run with the optional groups installed, and a
  build job that verifies the wheel installs and the CLI starts on its own.
- Migrated packaging metadata from `[tool.poetry]` to PEP 621 `[project]`.
- Dropped `--maxfail=1` from the pytest defaults so a CI run reports every
  failure rather than only the first.
- `joblib.dump` failures during a study now use `contextlib.suppress`.

### Fixed

- The test suite no longer requires PyTorch, XGBoost and LightGBM to be
  installed; tests needing an optional backend skip instead of failing, which is
  what broke CI on the initial commit.
- 54 ruff violations and 67 mypy `--strict` errors across the package; both now
  pass clean. Missing third-party stubs are declared as mypy overrides rather
  than left as noise.

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
