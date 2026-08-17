# Contributing

Thanks for your interest in this project. It is a research codebase, so
contributions are judged by whether they keep results **correct and
reproducible**, not only by whether the code runs.

## Getting set up

```bash
poetry install                                                    # core + dev tools
poetry install --with boost,neural,imbalance,explain,notebooks    # full research env
poetry run pre-commit install
```

The optional dependency groups are deliberately separated so a logistic or SVM
experiment does not require PyTorch, LightGBM or SHAP. Code must keep working
when those groups are absent: import optional libraries **inside** the function
that needs them and raise a clear `ImportError` naming the group to install.

Download the source data once:

```bash
poetry run scania-aps download
```

The UCI data is fetched at runtime and is never committed.

## Checks before you open a pull request

```bash
make lint    # pre-commit over every file: ruff, ruff-format, mypy --strict, hygiene
make test    # pytest
```

Both must pass. `make lint` delegates to pre-commit, and so does CI, so the
checks you run locally are exactly the checks that gate the pull request —
including the notebooks, which the bare `ruff check src tests` did not cover.
Tests run on Python 3.11, 3.12 and 3.13, plus one job with every optional group
installed.

## Research rules that reviews enforce

These are the project's non-negotiables. A change that violates one will be
rejected even if the tests pass.

1. **The official UCI test set is never used for any modelling decision.** Not
   for hyperparameters, not for calibration, not for thresholds.
2. **The four development roles stay separate.** `fit` estimates parameters,
   `tune` chooses hyperparameters, `calibration` fits Platt/isotonic, and
   `threshold` chooses the operating point. Never reuse one role for another.
3. **Resampling is fit-only.** SMOTE and undersampling belong inside an
   imbalanced-learn pipeline. Validation, calibration, threshold and test data
   are never resampled.
4. **Cost is the primary metric.** New models and studies must report
   `J = 10 * FP + 500 * FN`. Accuracy is not the optimization target.
5. **Thresholds are learned, not assumed.** `0.5` is not a defensible default
   here; the Bayes threshold implied by the stated costs is ~0.0196.
6. **Seeds are fixed** for candidate generation, splits and stochastic
   estimators wherever the library supports it.

## Style

- Ruff and mypy `--strict` are authoritative; the line limit is 100.
- Public functions need type annotations and a docstring that says *why*, not
  only *what*.
- Prefer adding to an existing module over creating a near-duplicate one.

## Adding a model family

1. Add a config dataclass and builder under `src/scania_aps/models/`.
2. Register it in `models/factory.py` and add candidates to `model_search.py`
   for both the `quick` and `full` profiles.
3. Add a test asserting the builder produces the expected estimator, and mark
   it `pytest.importorskip(...)` if it needs an optional group.
4. If it emits margins rather than probabilities, make sure `scoring.py`
   classifies it correctly so thresholds are optimized on the right scale.

## Notebooks

Notebooks call package code and must not hard-code performance numbers: every
figure in the text is produced by a cell, never typed in by hand.

**Commit them executed, with their outputs.** The notebooks are the published
form of the study — a reader on GitHub should see the tables and figures without
installing anything or obtaining the data. An unexecuted notebook is an empty
claim.

That means:

- Run the whole notebook top to bottom against the real UCI data before
  committing. A notebook with stale outputs from an older code path is worse
  than one with none.
- Charts go through `scania_aps.plotting`, which carries the house style and a
  colour-vision-safe palette. Do not hand-roll matplotlib styling per notebook.
- Keep the narrative. Each notebook states the question it answers, what the
  reader should look at, and what the result means for the maintenance decision.
  Code with no prose is not an experiment write-up.
- Serialized model artifacts still belong in `artifacts/`, which stays
  gitignored. Only the notebook's own outputs are committed.

## Commits and pull requests

Write commit subjects in the imperative mood ("Add isotonic calibration study").
In the pull request, state what changed, why, and whether any reported cost
figures move as a result. If a number changes, say which experiment produced it.

## Where the open work is

[docs/roadmap.md](docs/roadmap.md) lists what is worth doing next, ordered by how much each
item would change what the study can claim, with a rough cost for each. It also has a
"not planned" section — worth reading before proposing a leaderboard-chasing change.

The highest-value item is confidence intervals: every result currently rests on a single
train/test split, and that caveat attaches to most of `docs/findings.md`.

## Reporting problems

Open an issue using one of the templates. For anything security-related, follow
[SECURITY.md](SECURITY.md) instead of filing a public issue.
