# Rigor Extensions: Cross-Validation, SMOTE, Explainability, Fairness

Everything in this document was actually executed against the trained
models and the real dataset — not estimated. See `analysis/` for the
runnable scripts.

## 1. 5-Fold Stratified Cross-Validation (Classification)

Ran the selected model (HistGradientBoosting, tuned, `High_Risk`
sample-weighted) across 5 folds individually (chunked due to the dev
sandbox's single-CPU time limits per run — each fold trains independently,
see `analysis/cv_fold.py`).

| Fold | Accuracy | F1-macro |
|---|---|---|
| 1 | 0.9751 | 0.9015 |
| 2 | 0.9751 | 0.9014 |
| 3 | 0.9758 | 0.9048 |
| 4 | 0.9702 | 0.8832 |
| 5 | 0.9750 | 0.9019 |
| **Mean ± std** | **0.9743 ± 0.0020** | **0.8985 ± 0.0078** |

**Interpretation**: extremely tight spread across folds (accuracy range
0.9702–0.9758, under 0.6 percentage points). This is strong evidence the
97%+ accuracy is not an artifact of one lucky train/val split — the model
generalizes consistently regardless of which 20% of data is held out.

## 2. SMOTE vs. Sample-Weighting (High_Risk class imbalance)

The training pipeline currently handles `High_Risk` class imbalance via a
tuned `sample_weight=2.0` (see PROJECT_REPORT.md for how that value was
selected). SMOTE was implemented manually (`analysis/smote_manual.py` —
nearest-neighbor interpolation via `sklearn.neighbors.NearestNeighbors`,
since `imbalanced-learn` isn't installable in the offline dev sandbox) and
tested at 3 oversampling ratios for direct comparison:

| Approach | Macro F1 (val) | High_Risk F1 | High_Risk Recall | High_Risk Precision |
|---|---|---|---|---|
| No correction (baseline) | 0.863 | 0.63 | 0.52 | 0.82 |
| **Sample-weight = 2.0 (current)** | **0.888** | **0.71** | **0.76** | 0.68 |
| SMOTE, ratio=0.3 | 0.878 | 0.68 | 0.60 | 0.79 |
| SMOTE, ratio=0.5 | 0.879 | 0.68 | — | — |
| SMOTE, ratio=0.7 | 0.875 | 0.67 | — | — |

**Interpretation**: sample-weighting outperformed SMOTE at every tested
ratio on this dataset. A plausible reason: SMOTE's synthetic points are
linear interpolations in the *scaled/encoded* feature space, which can
blur the boundary for one-hot-encoded categorical features (interpolating
between two one-hot vectors produces a value like 0.5, which isn't a
category any real applicant can have) — sample-weighting doesn't have
this issue since it reuses real data points, just more heavily. The
current pipeline keeps sample-weighting as the production approach.

## 3. Feature Importance (Permutation Importance, in place of SHAP)

True SHAP requires the `shap` package, not installable in this offline
sandbox. Permutation importance (native to scikit-learn) is used instead
— it answers the same core question ("how much does this feature matter
to the model's predictions") globally, though unlike SHAP it doesn't give
per-individual-prediction attribution. If you have internet access,
`pip install shap` and swap in `shap.TreeExplainer(model)` for richer,
per-applicant explanations in the Streamlit app.

**Classification (top drivers of eligibility decisions):**
1. `requested_amount` (0.418 drop in F1 when shuffled)
2. `disposable_income` (0.415)
3. `requested_tenure` (0.287)
4. `house_type_Rented` (0.139)
5. `monthly_rent` (0.104)
6. `current_emi_amount` (0.096)
7. `debt_to_income_ratio` (0.061)
8. `credit_score` (0.056)

**Regression (top drivers of max EMI capacity):**
1. `disposable_income` (1.401 drop in R² when shuffled — dominant by far)
2. `house_type_Rented` (0.286)
3. `monthly_rent` (0.190)
4. `credit_risk_score` (0.042)

**Interpretation**: both models lean heavily on financially-sound,
explainable signals (requested loan size relative to disposable income,
housing costs) rather than demographic proxies — a good sign for a
FinTech risk model, though see the fairness audit below for one caveat.

## 4. Fairness Audit (Classifier, validation set)

| Group | Accuracy | Approval Rate |
|---|---|---|
| **Gender** — Female | 0.972 | 17.8% |
| **Gender** — Male | 0.975 | 18.1% |
| **Age** — 18-30 | 0.972 | 18.9% |
| **Age** — 31-40 | 0.974 | 18.0% |
| **Age** — 41-50 | 0.974 | 17.5% |
| **Age** — 51+ | 0.974 | 17.3% |
| **Education** — High School | 0.981 | 10.3% |
| **Education** — Graduate | 0.975 | 15.4% |
| **Education** — Post Graduate | 0.970 | 22.2% |
| **Education** — Professional | 0.969 | 26.6% |

**Interpretation**:
- **Gender**: no meaningful disparity — accuracy and approval rates are
  nearly identical across both groups.
- **Age**: no meaningful disparity — accuracy is flat, approval rate
  drifts mildly downward with age (18.9% → 17.3%), a small effect.
- **Education**: accuracy is fine everywhere (96.9–98.1%), but the
  **approval rate varies 2.5x** between High School (10.3%) and
  Professional (26.6%) applicants. This most likely reflects genuine
  underlying differences in income/credit history correlated with
  education level, rather than the model using education as a raw proxy
  — but a 2.5x gap is exactly the kind of finding that would need formal
  disparate-impact justification (e.g. under fair lending regulations)
  before this model could be used in a real production lending decision.
  **This was not further investigated or corrected — flagging it
  honestly as an open item, not resolving it.**

## What this does and doesn't establish

This is a legitimate strengthening of the original submission — real
numbers from real runs, not claims. It is still not an exhaustive
audit: a production deployment would want true SHAP (per-applicant
explanations), a formal disparate-impact statistical test (e.g. the
four-fifths rule) rather than a visual approval-rate comparison, and
likely input from a fair-lending compliance specialist, not just a
data science pass.
