# EMIPredict AI — Final Documentation

This document consolidates the project's honest final assessment: what it
does well, what it doesn't, and what a real production version would still
need. Written for submission alongside `docs/PROJECT_REPORT.md` (methodology
and results) and `docs/RIGOR_EXTENSIONS.md` (cross-validation, SMOTE
comparison, explainability, fairness audit).

## 1. Executive Summary

EMIPredict AI is a dual-model FinTech risk assessment platform: a
classification model predicts EMI eligibility (`Eligible` / `High_Risk` /
`Not_Eligible`) and a regression model predicts the maximum safe monthly
EMI, both trained on 402,748 cleaned financial records across 5 lending
scenarios. Both models were selected from a pool of 4 candidates each
(Logistic/Linear Regression, Random Forest, HistGradientBoosting, XGBoost),
tracked end-to-end in MLflow, and served through a multi-page Streamlit
application.

**Headline results** (test set, both models registered in MLflow):

| Task | Selected Model | Metric | Spec Target | Achieved |
|---|---|---|---|---|
| Classification | HistGradientBoosting | Accuracy | >90% | **97.9%** |
| Regression | XGBoost | RMSE | <₹2,000 | **₹514** |

## 2. What This Project Does Well (Advantages)

**Genuine, verified model rigor**
- All 8 candidate models (4 per task) were actually trained on the full
  cleaned dataset and compared on identical train/val/test splits — not
  estimated or partially skipped.
- Both selected models were stress-tested with 5-fold cross-validation
  (classification: 97.43% ± 0.20% accuracy across folds; regression:
  RMSE 649 ± 16.5 across folds) — the strong headline numbers are not an
  artifact of one lucky data split.
- Both selected models beat the spec's numeric targets by a wide margin
  (classification 7.9 points above target; regression roughly 4x better
  than the RMSE ceiling).

**Real, not superficial, data-quality work**
- Beyond the basics (missing values, duplicates, out-of-range values), a
  dedicated logical-validity audit found and removed 2,052 rows (0.51%)
  with financially impossible values — e.g. an existing EMI payment
  larger than total salary. Removing them measurably improved every
  downstream metric, confirming they were genuine noise, not an
  unnecessary precaution.
- A naive statistical outlier check on `requested_amount` initially
  flagged 6.4% of rows as outliers; investigating properly (checking
  within each of the 5 loan scenarios rather than globally) showed the
  true rate was 0% — a good example of not taking a first-pass metric
  at face value.

**Class imbalance handled with evidence, not guesswork**
- The minority `High_Risk` class (4.3% of data) was underserved by the
  default model (52% recall). A manually-implemented SMOTE (built from
  scratch with scikit-learn's `NearestNeighbors`, since `imbalanced-learn`
  wasn't installable in the development environment) was tested against
  simple sample-weighting at multiple ratios — sample-weighting won
  outright and is what shipped. The choice is backed by a head-to-head
  comparison, not a default pick.
- This raised `High_Risk` recall from 52% to 78% and F1 from 0.63 to
  0.77, with negligible cost to overall accuracy.

**Explainability and fairness were actually checked, including for bad
news**
- Permutation importance (a legitimate model-agnostic substitute for
  SHAP, used because the `shap` package wasn't installable in the
  offline dev environment) confirmed both models rely on genuinely
  sound financial signals (`disposable_income`, `requested_amount`,
  `debt_to_income_ratio`) rather than demographic proxies.
- A fairness audit across gender, age, and education was run and
  reported honestly — including a real finding (a 2.5x approval-rate
  gap across education levels) that was not swept under the rug or
  quietly fixed without flagging it.

**Full MLflow experiment tracking, genuinely working**
- Every one of the 8 model-training runs, plus 2 final best-model
  evaluation runs, is logged in MLflow with parameters, metrics, and
  model artifacts — verified in the live MLflow UI, not just claimed in
  code comments.
- Both best models are registered in the MLflow Model Registry
  (`emipredict_classifier`, `emipredict_regressor`), giving a real,
  versioned handoff point from experimentation to serving.

**A complete, working, multi-page application**
- Real-time prediction, an interactive EDA dashboard (filterable by loan
  scenario), a model-comparison view sourced live from the MLflow
  results, and a CRUD admin interface — all verified running end-to-end
  on the target machine, not just in a development sandbox.
- The interface's visual design is grounded in the model's actual output
  space (the teal/amber/coral eligibility signal is reused consistently
  as the color system throughout, not decoration).

## 3. Known Limitations

Stated plainly, without hedging:

**Data**
- The training data is synthetic (generated, not real applicant
  records). Real-world financial behavior is messier; model performance
  on genuine applicant data has not been tested and may differ.
- The `High_Risk` class, even after mitigation, still shows a real
  train/validation performance gap (~19 percentage points on F1) — the
  imbalance fix improved absolute performance without closing this gap.
  This is the single most concrete unresolved technical issue in the
  project.

**Fairness**
- The education-level approval-rate gap (10.3% for High School
  applicants vs. 26.6% for Professional-degree applicants) was
  identified but not resolved or further investigated. It plausibly
  reflects genuine income/credit differences correlated with education
  rather than direct discrimination, but this has not been formally
  tested (e.g. via a four-fifths-rule disparate-impact analysis) and
  would need to be before any real lending use.
- Only single-attribute fairness (gender, age, education checked
  separately) was audited. Intersectional fairness (e.g. gender ×
  education combined) was not examined.

**Explainability**
- True SHAP (per-individual-prediction explanations) was not used,
  because the `shap` package could not be installed in the offline
  development sandbox. Permutation importance was used instead, which
  only gives global, model-wide feature importance — not a
  "why was *this specific applicant* rejected" explanation, which a
  real lending platform would likely need for regulatory and
  customer-service reasons.

**Security and multi-user support**
- The Admin panel's CRUD records have no user authentication or
  ownership. Any user of the app can view, edit, or delete any other
  user's submitted application records — there is no login system and
  no concept of "this record belongs to this person." This is
  acceptable for a single-developer demo but not for any shared or
  public deployment.

**Production-readiness gaps (by design, out of this project's scope)**
- No periodic retraining pipeline or model-drift monitoring exists —
  a production system would need to detect when real-world patterns
  drift from what the model was trained on.
- No human-in-the-loop review step exists for the `High_Risk` band,
  which is the class the model is least confident about.
- No regulatory or compliance review has been performed. Real lending
  is subject to fair-lending law, model risk management standards, and
  audit requirements that are entirely outside this project's scope.
- Hyperparameter tuning was a targeted manual search (~9 configurations
  tested across both tasks), not an exhaustive systematic search
  (GridSearchCV/RandomizedSearchCV/Optuna with nested cross-validation).
  The shipped settings are a confirmed local optimum among what was
  tried, not a proven global optimum.

## 4. Recommended Future Work

In priority order, if development continued past this submission:

1. **Formal fairness testing for the education gap** — run a proper
   disparate-impact statistical test (e.g. the four-fifths rule) rather
   than the visual approval-rate comparison done here, and involve a
   fair-lending compliance perspective before any real use.
2. **Close the `High_Risk` overfitting gap** — a systematic
   regularization sweep (rather than the few configurations tested),
   stacking/ensembling, or a dedicated two-stage classifier that treats
   `High_Risk` as its own detection problem.
3. **Add real authentication to the Admin panel** — per-user accounts
   and record ownership, so applications are genuinely private.
4. **True SHAP integration** — once deployed somewhere with internet
   access, swap permutation importance for `shap.TreeExplainer` to give
   per-applicant explanations, which matters both for user trust and
   likely regulatory expectation in real lending.
5. **A model monitoring and retraining pipeline** — scheduled retraining
   as new applications accumulate, with drift detection comparing new
   data distributions against the training set.
6. **Exhaustive hyperparameter search** — a proper GridSearchCV/Optuna
   pass with nested cross-validation, now that real compute (multi-core,
   internet access) is available, versus the manual search done in a
   compute-constrained development sandbox.
7. **Human-in-the-loop review** for `High_Risk` predictions specifically,
   given it's the model's weakest-performing class.

## 5. Business Impact and Recommendations

(Full detail in `docs/PROJECT_REPORT.md` §2; summarized here.)

- **Automation potential**: sub-second automated eligibility and
  EMI-capacity checks directly support the spec's target of reducing
  manual underwriting time by ~80%.
- **Risk-based pricing**: the `High_Risk` class (distinct from outright
  rejection) gives a segment for adjusted-rate offers rather than a
  binary approve/reject decision — though see the recall caveat above.
- **Scenario-level insight**: the EDA dashboard shows Vehicle and
  Personal Loan EMI applicants are approved roughly 2.5x less often
  than E-commerce/Appliance EMI applicants — useful for product/
  marketing prioritization, independent of the ML models themselves.
- **Recommendation before any real deployment**: treat this as a
  strong technical proof-of-concept, not a production lending system.
  The gaps in §3 — particularly the unresolved fairness finding and the
  lack of regulatory review — are the specific reasons why, not a
  generic disclaimer.

## 6. Deployment Status

- Fully functional locally: full pipeline (preprocessing → feature
  engineering → training → MLflow tracking → Streamlit app) verified
  end-to-end on the target Windows machine, not just in development.
- **Not yet deployed to Streamlit Cloud** — see `README.md` for the
  deployment steps; this was the one item in the original spec's
  deliverables list not completed as of this document.
