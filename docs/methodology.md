# Methodology

## 1. Operational objective

The binary target indicates whether a truck failure is attributable to the Air Pressure System (APS). The challenge defines the asymmetric cost

\[
L(\hat y,y)=
\begin{cases}
10,&\hat y=1,\ y=0,\\
500,&\hat y=0,\ y=1,\\
0,&\text{otherwise}.
\end{cases}
\]

The primary empirical objective is therefore

\[
J(\theta,\tau)=10FP(\theta,\tau)+500FN(\theta,\tau).
\]

This is intentionally different from optimizing accuracy, ROC-AUC or even PR-AUC.

## 2. Data roles and leakage control

The official training set is divided into four stratified subsets:

- fit: parameter estimation;
- tune: hyperparameter/model selection;
- calibration: Platt/isotonic calibration only;
- threshold: final operating-threshold selection.

The official test set is not used by imputation, scaling, resampling, model fitting, hyperparameter selection, calibration or threshold selection.

This distinction is particularly important for this project because threshold tuning can itself overfit an imbalanced validation set.

## 3. Missing values

All linear, margin, neural and bagged-tree pipelines use median imputation fit only on training observations. Missing-value indicators are appended because missingness can itself contain operational information.

Boosting libraries can often handle missing values natively, but the comparative study uses the same explicit imputation/indicator convention so that preprocessing is controlled across families.

## 4. Scaling

Scaling is used for logistic regression, SVM, MLP and the autoencoder. It is required for fair coefficient regularization and materially affects gradient-based optimization.

For L1/L2 penalties, lack of scaling would make the effective regularization strength depend on arbitrary feature units.

## 5. Linear regularization

For logistic regression,

\[
\min_{\beta,b}
\frac1n\sum_i
\log(1+\exp[-y_i(\beta^Tx_i+b)])
+\lambda\left[
\alpha\lVert\beta\rVert_1+
\frac{1-\alpha}{2}\lVert\beta\rVert_2^2
\right].
\]

The experiments include near-unregularized, L1, L2 and Elastic Net models.

The solver is chosen per penalty rather than fixed. SAGA is the only solver covering all three penalties, which makes it attractive as a uniform choice, but on this design matrix (42,000 rows by 340 columns after missingness indicators) it does not converge within a practical iteration budget, and a capped SAGA fit is not the estimator that was requested. Measured on an 8,000-row sample at `C=1`:

| penalty | solver | iterations | converged | seconds | zeroed coefficients |
|---|---|---|---|---|---|
| L2 | SAGA | 4000 (capped) | no | 67.1 | 2 |
| L2 | lbfgs | 104 | yes | 0.4 | 2 |
| L1 | SAGA | 4000 (capped) | no | 171.1 | 121 |
| L1 | liblinear | 30 | yes | 5.4 | 185 |

The last column is the substantive point: under an L1 penalty the non-converged SAGA fit understates sparsity by a third, so coefficient counts read off it are wrong, not merely imprecise.

L2 therefore uses lbfgs and L1 uses liblinear. **Elastic net still uses SAGA and still does not converge** — no alternative solver supports it — so elastic-net coefficients in this repository are approximate, and the elastic-net rows of the logistic sweep should be read as indicative rather than exact.

One consequence of liblinear worth recording: it penalizes the intercept, which lbfgs and SAGA do not. For this problem the effect is largely absorbed by the separately-learned decision threshold, but the L1 and L2 fits are not identically parameterized.

The linear SVM studies

\[
\min_{w,b}\frac12\lVert w\rVert_2^2+C\sum_i\max(0,1-y_i(w^Tx_i+b)),
\]

where \(C\) controls the trade-off between margin width and hinge-loss violations.

## 6. Tree regularization

Random Forest and Extra Trees expose:

- number of trees;
- maximum depth;
- minimum samples per leaf;
- feature subsampling;
- class weighting.

XGBoost and LightGBM add:

- shrinkage through learning rate;
- L1 and L2 leaf penalties;
- row subsampling;
- column subsampling;
- depth/leaf complexity;
- child constraints;
- positive-class weighting.

These mechanisms let the repository show underfitting, useful regularization and overfitting in the same real problem.

## 7. Neural optimization

The PyTorch MLP supports SGD, Adam and AdamW. Training controls include learning rate, cosine/plateau schedules, batch size, gradient clipping, dropout, batch normalization, weight decay and early stopping.

Adam maintains moment estimates

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\]

whereas AdamW decouples the weight-decay update from the adaptive gradient normalization.

The estimator stores epoch-level training and validation losses so convergence is observable rather than inferred from the final test metric.

## 8. Representation learning

The autoencoder learns a latent vector \(z=f_\phi(x)\) by minimizing reconstruction error

\[
\min_{\phi,\psi}\sum_i\lVert x_i-g_\psi(f_\phi(x_i))\rVert_2^2.
\]

A regularized logistic classifier is then trained on the latent codes. This experiment asks whether an unsupervised nonlinear representation is useful when positive failure labels are scarce.

## 9. Class imbalance

The repository compares five approaches:

1. no correction;
2. class-weighted loss;
3. random undersampling;
4. SMOTE;
5. focal loss.

Sampling methods run only during `fit` inside an imbalanced-learn pipeline. This prevents synthetic samples or discarded observations from altering validation distributions.

Focal loss uses the form

\[
L_{focal}=-(1-p_t)^\gamma\log p_t,
\]

with optional positive-class weighting.

The experiment tests an important distinction: imbalance can be handled in the training loss, in the sampling distribution, in the deployment threshold, or through combinations of those mechanisms.

## 10. Calibration

The repository compares raw probabilities with sigmoid/Platt and isotonic calibration. Calibration receives its own holdout subset.

For probability-producing models, Brier score and log loss are reported in addition to ranking metrics and maintenance cost.

Calibration is not assumed to improve the operational objective. It is tested empirically.

## 11. Decision threshold

For calibrated posterior probability \(p=P(Y=1\mid X)\), a positive maintenance action is preferred when

\[
10(1-p)<500p,
\]

which gives

\[
p>\frac{10}{510}\approx0.0196.
\]

The implementation nevertheless estimates the exact empirical cost-minimizing threshold on the dedicated threshold subset because finite samples, misspecification and miscalibration can move the optimum.

For SVM margins, the same threshold optimizer works on arbitrary finite scores; no probability interpretation is imposed.

## 12. Feature selection

The feature-selection study compares:

- L1 embedded selection;
- mutual information;
- recursive feature elimination;
- Extra-Trees-based selection.

Interpretation utilities also provide coefficient sparsity, permutation importance, tree importance and optional SHAP explanations.

Because the Scania variables are anonymized, these tools are used primarily for statistical stability and predictive structure rather than unsupported physical stories about individual variable names.

## 13. Hyperparameter selection

Each model family receives a reproducible candidate set. Candidates are fit on the fit subset and ranked on the tune subset by minimum maintenance cost, with PR-AUC as a tie-breaker.

The selected configuration is refit on fit+tune before optional calibration. The threshold subset remains untouched until the decision rule is chosen.

The `quick` search profile provides a small reproducible study. The `full` profile broadens the search over regularization and optimizer settings.

## 14. Ablation design

The XGBoost ablation removes L1/L2 penalties, stochastic subsampling, positive-class weighting and threshold optimization one at a time. This identifies which components actually reduce maintenance cost.

Neural optimizer variants similarly isolate SGD, Adam and AdamW under controlled regularization settings.

## 15. Evaluation

The primary outputs are:

- total maintenance cost;
- cost per truck;
- false negatives;
- false positives;
- recall;
- precision;
- PR-AUC;
- ROC-AUC;
- Brier score and log loss for probabilities;
- savings relative to trivial always-negative and always-positive policies.

No result is hard-coded. Experiment CSV/JSON files are generated from the real data at runtime.
