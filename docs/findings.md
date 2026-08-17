# Findings

What the study established, with the notebook that produced each number.

Every figure here comes from a committed notebook run against the real UCI data. Nothing
is hard-coded: re-running the notebook regenerates the number. Where a figure is a
development-set result rather than an official test-set one, it says so — the two are not
comparable and mixing them is the easiest way to overstate a result.

Read [methodology.md](methodology.md) for the protocol these results rest on, and
[the caveats](#what-this-study-does-not-establish) before quoting any of them.

## The problem in four numbers

| | training set | official test set |
|---|---|---|
| trucks | 60,000 | 16,000 |
| failures | 1,000 (1.67%) | 375 (2.34%) |
| features | 170 | 170 |
| cost of inspecting nothing | — | 187,500 |
| cost of inspecting everything | — | 156,250 |

Inspecting every truck is already cheaper than inspecting none, before any model exists.
That asymmetry — a missed failure costs 50 unnecessary inspections — drives every result
below. ([02](../experiments/02_cost_sensitive_baselines.ipynb))

---

## 1. The decision rule is worth more than the model

**On an unchanged model, moving from a `0.5` threshold to a cost-optimal one cuts the
bill 64%.**

| decision rule | threshold | missed failures | cost |
|---|---|---|---|
| conventional `0.5` | 0.5 | 57 | 28,970 |
| Bayes, from the cost ratio | 0.0196 | 11 | **9,740** |
| learned on held-out data | 0.0140 | 11 | 10,350 |

*Official test set, XGBoost.* ([10](../experiments/10_threshold_optimization.ipynb))

The spread across all eight model families is about **9,300**
([13](../experiments/13_final_model_comparison.ipynb)); the spread between these decision
rules on a *single* model is about **18,600**. Choosing the threshold well is worth
roughly twice what choosing the algorithm is worth, and costs nothing to apply to a model
you have already trained.

**The theoretical threshold beat the learned one.** `0.0196` comes from the cost ratio and
looks at no data, so it has no variance; the learned threshold is fitted to 6,000 trucks
and inherits their sampling noise. When the probabilities are close enough to calibrated,
the zero-variance estimate can win. The gap is small and well inside what one split can
resolve — neither should be reported as definitively better.

## 2. Missingness is a feature, not a nuisance

**Some sensors are absent up to 74 percentage points more often on trucks that later
fail.** ([01](../experiments/01_data_quality_and_missingness.ipynb))

The gap runs in both directions: six sensors are far *more* complete on failed trucks,
nine far less. If missingness were random every one of those gaps would sit at zero.

This is why every pipeline in the package imputes with
`SimpleImputer(strategy="median", add_indicator=True)`. The median fills the hole so the
estimator can run; the indicator preserves the fact that there was one. Several
`missing::` indicators survive L1 selection with non-zero coefficients
([03](../experiments/03_logistic_regularization.ipynb)), so the model is genuinely using
them.

## 3. Rebalancing the training data mostly does not pay

**SMOTE, the most elaborate strategy here, is the most expensive - and it loses to doing
nothing.**

| strategy | cost | vs no correction |
|---|---|---|
| random undersampling | **16,830** | −5% |
| class weighting | 17,170 | −3% |
| *no correction* | *17,710* | — |
| focal loss (MLP) | 18,120 | +2% |
| SMOTE | 18,880 | +7% |

*Official test set, each at its own cost-optimal threshold.*
([08](../experiments/08_imbalance_methods.ipynb))

Rebalancing and thresholding attack the same problem — a model reluctant to predict the
rare class — and the threshold solves it directly, at the decision, without touching what
the model learned. Applying both does not compound; it mostly repeats.

The whole spread across five strategies is about 2,000, against the 18,600 that finding 1
attributes to the threshold alone. That comparison is the point: rebalancing is a small
lever pulled at fit time, thresholding a large one pulled at decision time.

SMOTE deserves the sharpest reading. Interpolating between real failures in 170 anonymised
dimensions gives no guarantee the synthetic point corresponds to a realisable truck, and
the cost column suggests it does not help find real ones.

The ordering within the top three is not robust: undersampling and class weighting swapped
places when the logistic solver was corrected, on differences of a few percent. Read the
grouping — simple corrections marginally ahead, elaborate ones behind — rather than the
rank order.

## 4. Calibration did not improve the probabilities

**Uncalibrated XGBoost has the best Brier score and log loss; both corrections made them
worse.**

| method | Brier | log loss | PR-AUC |
|---|---|---|---|
| uncalibrated | **0.005251** | **0.020241** | 0.9228 |
| Platt / sigmoid | 0.005290 | 0.026402 | 0.9228 |
| isotonic | 0.005302 | 0.026653 | 0.8858 |

*Official test set.* ([09](../experiments/09_probability_calibration.ipynb))

Gradient boosting on log loss is already close to calibrated here, and fitting a
correction to a model that does not need one adds variance from the calibration subset.

Note the PR-AUC column. Platt scaling is strictly monotone, so it cannot reorder any pair
of trucks and the ranking metric is unchanged to four decimals. Isotonic regression is
monotone but only *non-decreasing* — a step function that maps ranges of scores to one
value. Those ties collapse the ordering inside each step, which is exactly what the drop
to 0.886 measures.

## 5. The regularization was costing money

**Removing the L1/L2 leaf penalties from the boosting pipeline *improved* test cost by
13%.**

| component removed | change in cost |
|---|---|
| cost-optimal threshold → `0.5` | **+14,960 (+135%)** |
| positive-class weighting | +310 (+3%) |
| row/column subsampling | −430 (−4%) |
| L1/L2 leaf penalties | −1,470 (−13%) |

*Official test set, against the full pipeline at 11,050.*
([12](../experiments/12_ablation_study.ipynb))

The threshold row dominates everything else combined — the same conclusion as finding 1,
reached from the opposite direction.

The negative rows are the uncomfortable ones. The supported claim is **"these penalties
are mistuned"**, not "regularization is useless": one setting of `reg_alpha`/`reg_lambda`
on one split is not a verdict on the technique. The actionable response is to fold the
penalty strengths into the search rather than fixing them by default. At a few percent,
these two rows are also not clearly outside single-split noise; the threshold row is.

## 6. The training objective is not the business objective

**Three optimisers, three different winners.**

| optimiser | best validation loss | PR-AUC | cost |
|---|---|---|---|
| SGD | 0.1677 | 0.7458 | **5,410** |
| Adam | 0.1781 | **0.7663** | 6,910 |
| AdamW | **0.1606** | 0.7478 | 6,070 |

*Development (tuning) subset. Same architecture, same seed, same schedule.*
([07](../experiments/07_neural_network_optimization.ipynb))

Each optimiser wins on a different measure. The network is trained on binary
cross-entropy, ranked by average precision, and invoiced at `10·FP + 500·FN`; those three
functions put their minima in different places, so "the best optimiser" is not a
well-formed question until the metric is named.

Practically: selecting on validation loss would have picked the wrong optimiser here, and
early stopping on loss — which the estimator does internally — is a proxy, not the target.

## 7. Nonlinearity helps, substantially

| family | cost | cost/truck | missed failures |
|---|---|---|---|
| extra trees | **9,940** | 0.62 | 11 |
| XGBoost | 10,610 | 0.66 | 13 |
| LightGBM | 12,750 | 0.80 | 18 |
| MLP | 13,250 | 0.83 | 18 |
| random forest | 13,750 | 0.86 | 23 |
| logistic regression | 15,540 | 0.97 | 23 |
| autoencoder + logistic | 16,200 | 1.01 | 18 |
| linear SVM | 19,250 | 1.20 | 30 |

*Official test set, each at its own cost-optimal threshold.*
([13](../experiments/13_final_model_comparison.ipynb))

The cheapest tree ensemble costs **56% less than logistic regression** and **48% less
than the linear SVM**. That is a real gap, large enough to justify deploying a tree
ensemble where the operational cost of doing so is moderate.

Two things about the middle of this table. Logistic regression **beats the autoencoder**,
so an unsupervised nonlinear representation feeding a linear classifier bought nothing
over the linear classifier alone — unsurprising with only 1,000 positive examples to learn
a representation from. And logistic regression improved 19% (from 19,170) once its solver
was corrected to one that converges; the earlier figure came from a fit that had hit its
iteration cap, and it flattered the nonlinear families.

**Extra trees edging XGBoost is not meaningful.** This comparison runs three candidates
per family; treat the top few as tied and consult the per-family notebooks for tuned
results.

The best model misses 11 of 375 failures at 0.62 per truck — a **93.6% saving against
inspecting every truck**, and 94.7% against inspecting none.

## 8. Feature selection can help, but not the obvious methods

| selector | features kept | cost |
|---|---|---|
| extra-trees selection | 170 of 339 | **13,840** |
| RFE | 50 | 15,660 |
| all features | 339 | 17,170 |
| L1 embedded | 339 (selected all) | 17,170 |
| mutual information | 50 | 19,760 |

*Official test set. 339 = 170 sensors + 169 missingness indicators.*
([11](../experiments/11_feature_selection.ipynb))

Extra-trees selection halves the feature set and cuts cost by **19%**. Mutual
information, the only purely univariate method here, is the one selector that does clearly
worse than using everything — **15% worse** — because it scores each feature in isolation
and so cannot see the interactions the model relies on.

The L1 row is not a null result but a non-result: `SelectFromModel` with a median
threshold retained all 339 columns, so it is the all-features row under another name.

## 9. Development-set results, for reference

These rank configurations within a family and are **not** comparable to the test-set
figures above.

- **Logistic** — best is L1, `C=0.0014`, class weight 10, at 6,440. The top twelve
  configurations span 6,440–7,270, so the penalty family matters less than the fact that
  some regularization is present. The winning model keeps **33 non-zero coefficients of
  339**, nine of them missingness indicators, so the indicators from finding 2 carry real
  weight and the sensor set is heavily redundant.
  ([03](../experiments/03_logistic_regularization.ipynb))
- **Linear SVM** — the cost-optimal cut sits at a margin of **−0.84**, deep in what a
  default `decision_function > 0` rule would call healthy. That default costs roughly six
  times the optimum. ([04](../experiments/04_svm_margin_regularization.ipynb))
- **Tree ensembles** — random forest 4,620 best / 5,050 median; extra trees 5,040 / 6,465.
  Random forest is both better and less sensitive to its structural settings here.
  ([05](../experiments/05_tree_ensembles.ipynb))
- **Boosting** — best is LightGBM at 3,860 with 3 missed failures; the six cheapest
  configurations are all LightGBM. ([06](../experiments/06_gradient_boosting.ipynb))

---

## What this study does not establish

- **No confidence intervals.** Every number is one train/test split. Differences between
  adjacent families, and the few-percent ablation rows, are not statistically separated
  and should not be reported as if they were. Only the threshold effect is large enough
  to be safe from this.
- **The costs are taken as given.** `10` and `500` come from the challenge. A different
  ratio moves every threshold in this repository and could reorder the families.
- **The final comparison uses a reduced search.** Three candidates per family, not twelve
  — at full profile it exceeds four hours. The per-family notebooks tune properly.
- **The features are anonymised.** Nothing here explains *why* a truck fails. This is a
  detection study, not a diagnostic one, and the feature rankings carry no physical
  meaning.
- **Large-`C` SVM fits do not converge.** `LinearSVC` exceeds its 10,000-iteration budget
  at weak margin regularization on the full fit subset. The selected configuration uses
  heavy regularization and converges in a few dozen iterations, so the headline SVM result
  stands, but the upper end of the sweep in
  [04](../experiments/04_svm_margin_regularization.ipynb) is approximate.
- **Elastic-net fits do not converge.** L1 and L2 use solvers that reach an optimum
  (`liblinear` and `lbfgs`), but elastic net has only `saga`, which hits its iteration cap
  on this design matrix. Roughly a third of the logistic sweep is elastic net, so those
  coefficients are approximate. `methodology.md` carries the measurements.
- **Some figures here superseded earlier ones.** Every logistic number was recomputed
  after the solver was corrected; the previous values came from capped fits. Logistic
  regression's test cost improved 19% and the imbalance ranking reordered. Where a result
  moved on a change of solver it was not robust to begin with, and this document says so
  at the point it matters.

Several of these limitations are addressable, and
[roadmap.md](roadmap.md) sets out how — confidence intervals first, since nearly every
claim above inherits that one.

## Reproducing any of this

```bash
poetry install --with boost,neural,imbalance,explain,notebooks
poetry run scania-aps download
poetry run jupyter lab experiments/
```

Runtimes vary sharply: notebook 02 takes seconds, notebook 03 about 110 minutes, and the
full-profile version of notebook 13 exceeds four hours. The committed outputs mean you do
not have to run any of it to read the results.
