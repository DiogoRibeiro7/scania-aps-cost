# Roadmap

What is worth doing next, and why. Ordered by how much each item would change what the
study can claim — not by how easy it is.

Each entry states the gap, what closing it would buy, and roughly what it costs.
[findings.md](findings.md) lists what the study currently establishes;
[methodology.md](methodology.md) describes the protocol. This document is the complement:
the things those two cannot yet support.

Status as of 0.3.1.

---

## 1. Confidence intervals — the one that limits everything else

**Gap.** Every number in the study comes from a single train/test split. There is no
estimate of variance on any of them, so differences between adjacent model families, the
few-percent ablation rows, and the reordering seen in the imbalance study cannot be
distinguished from noise. [findings.md](findings.md) has to say so repeatedly, and that
caveat currently attaches to most of the document.

**What it would buy.** The ability to state which findings are real. Two results have
already moved under changes that should have been inconsequential — the imbalance ranking
reordered on a solver change, and the learned threshold lost to the theoretical one — and
in both cases the honest reading was "this was never robust". A resampling estimate would
say that up front instead of after the fact.

**Approach.** Repeated stratified splits of the official training data, evaluating each
model family end to end per repeat, reporting cost as a distribution rather than a point.
The official test set stays untouched; the variance being estimated is over the
development split. Bootstrap over the test set is a cheaper partial alternative that
captures evaluation noise but not selection noise, and the second is likely larger here.

**Cost.** The expensive part is that each repeat multiplies the full study runtime, which
is already hours. Realistically this needs either a reduced family set per repeat or a
machine that is not a laptop. Worth scoping before committing.

**Priority: highest.** Nearly every other claim inherits this limitation.

---

## 2. Cost-ratio sensitivity

**Gap.** The costs `10` and `500` are taken from the challenge definition and never
varied. Every threshold in the repository, and possibly the family ranking, depends on
them.

**What it would buy.** The study's central claim is that the decision rule matters more
than the model. That claim should be *more* true as the cost ratio becomes more extreme
and less true as it approaches 1:1 — showing that curve would turn one observation into a
mechanism. It would also tell a fleet operator whose real ratio is 30:1 or 80:1 whether
these conclusions transfer.

**Approach.** Sweep the ratio across a range, recompute the cost-optimal threshold per
model, and plot cost against ratio per family. Cheap, because it needs no refitting: the
scores are already computed and only the decision rule changes.

**Cost.** Low. This is the best value-per-hour item on the list — a new notebook reusing
existing fitted scores.

**Priority: high.**

---

## 3. Threshold stability

**Gap.** The learned threshold was beaten by the theoretical Bayes threshold on the test
set ([findings.md](findings.md) finding 1). The explanation offered is that the learned
value is fitted to 6,000 trucks and carries their sampling noise while the Bayes value has
none. That explanation is plausible and untested.

**What it would buy.** A direct measurement of how much the learned threshold varies
across resamples of the threshold subset. If its spread straddles the Bayes value, the
recommendation becomes "use the theoretical threshold when the model is roughly
calibrated", which is simpler and cheaper to operate than fitting one.

**Approach.** Bootstrap the threshold subset, re-optimise the threshold on each resample,
and compare the resulting distribution against `10/(10+500)`. Requires no refitting.

**Cost.** Low, and it directly tests a claim the study currently only asserts.

**Priority: high**, and it pairs naturally with item 2.

---

## 4. Elastic net does not converge

**Gap.** `saga` is the only solver supporting elastic net, and it exhausts its iteration
budget on this design matrix. Roughly a third of the logistic sweep therefore has
approximate coefficients. Documented in [methodology.md](methodology.md), not fixed.

**Options, none free.**

- Raise the iteration budget substantially. Honest, and makes an already 176-minute
  notebook considerably slower.
- Fit elastic net with `SGDClassifier(loss="log_loss", penalty="elasticnet")`, which
  optimises the same objective by a different route and may converge here. This is a
  different estimator with its own hyperparameters, so it is not a drop-in substitute and
  would need its own validation.
- Drop elastic net. Cheapest, and loses a comparison the study exists to make — L1 versus
  L2 versus their combination.

**Recommendation.** Try `SGDClassifier` as a spike first and compare its coefficients
against a very-long-budget `saga` run on a subsample. If they agree, switch; if not, raise
the budget and accept the runtime.

**Priority: medium.** It affects one family's within-sweep ranking, not the headline
results.

---

## 5. Runtime, and the reproducibility it costs

**Gap.** Notebook 03 takes 176 minutes and notebook 13 exceeded four hours at full
profile, which is why it ships at `quick`. A notebook nobody can re-run is weak evidence
however carefully it was produced.

**What it would buy.** The final comparison at full search depth, and notebooks a reader
can actually execute.

**Where the time goes.** Measured, not assumed: `liblinear` at the very small `C` values
the sweep explores is expensive on 42,000 rows, and elastic-net `saga` runs to its cap
every time. Item 4 addresses part of it.

**Approach.** Profile before optimising. Candidate levers: cache the imputer and scaler
across candidates within a family rather than refitting per candidate; reduce the sweep to
a coarse-then-fine search; parallelise across candidates rather than within each fit.

**Priority: medium.** It gates item 1 in practice, since repeated splits multiply whatever
the base runtime is.

---

## 6. Calibration on a model that needs it

**Gap.** Calibration was tested on XGBoost, which turned out to be nearly calibrated
already, so both corrections made things slightly worse. That is a real finding but a
narrow test of the technique.

**What it would buy.** The MLP and the SVM are the natural candidates — a network trained
with class weighting produces distorted probabilities, and the SVM produces no
probabilities at all. Calibrating those would show whether the machinery earns its place
on a model that is genuinely miscalibrated, rather than only that it is unnecessary on one
that is not.

**Cost.** Low; the calibration study already accepts a family parameter.

**Priority: medium.**

---

## 7. SHAP is implemented but unused

**Gap.** `feature_selection.shap_values` exists, is covered by the optional `explain`
group, and no notebook calls it. The interpretation study uses mutual information and
permutation importance only.

**What it would buy.** Permutation importance answers "what does this model lose without
the feature". SHAP answers "what pushed this particular truck over the threshold", which
is the question a maintenance planner actually asks about a flagged vehicle. With
anonymised features it cannot support physical explanation, and the notebook should say so
plainly.

**Cost.** Low, though SHAP on a large tree ensemble is not fast and will need a bounded
sample.

**Priority: low-medium.** Useful, and honest about what anonymised features permit.

---

## 8. Smaller items

- **Notebook runtimes are undocumented.** A reader starting notebook 03 has no warning it
  will take three hours. A one-line note at the top of the slow notebooks costs nothing.
- **`docs/experiments.md` overlaps the notebook index.** Now that each notebook states its
  own question and `findings.md` collects the results, that document's role is unclear and
  it may be better merged into one of them.
- **Coverage sits at 94%**, and the remainder is mostly error branches in `costs.py` and
  `metrics.py`. Worth closing only if it catches something; coverage as a target rather
  than a diagnostic is not a goal here.
- **Notebooks 04, 09, 10 and 12 have not been re-executed** since the deprecation cleanup.
  Their results are unaffected — verified, not assumed — but a fresh run would drop the
  last of the stale warning output. Worth folding into the next change that touches them
  rather than doing for its own sake.

---

## Explicitly not planned

- **Beating a leaderboard.** The study exists to measure which mechanisms move a cost
  function, not to win the challenge. A tuned ensemble that answers no question is not
  progress.
- **Explaining why trucks fail.** The features are anonymised. No amount of
  interpretability tooling turns this into a diagnostic study, and claiming otherwise
  would be the most tempting available overreach.
- **Deep architectures.** With 1,000 positive training examples the MLP is already at the
  edge of what the labels support. The autoencoder result — beaten by plain logistic
  regression — is the evidence for that.
