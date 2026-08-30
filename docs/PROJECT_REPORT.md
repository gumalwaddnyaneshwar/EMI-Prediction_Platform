# EMIPredict AI — Project Report

## 1. Methodology

### 1.1 Data Preprocessing
- Loaded 404,800 raw records (22 input features + 2 targets).
- Fixed malformed numeric encodings in `age`, `monthly_salary`, `bank_balance`
  (source data occasionally exported doubled decimal suffixes, e.g. "58.0.0").
- Standardized inconsistent `gender` casing (Male/male/M/MALE → Male, etc.).
- Corrected out-of-range `credit_score` values (spec domain: 300–850) by
  treating out-of-range values as missing, then imputing.
- Imputed missing values: `education` (mode), `monthly_rent` (0 — absence
  plausibly means owns home / lives with family), `credit_score`,
  `bank_balance`, `emergency_fund` (median).
- **Removed 2,052 rows (0.51%) with logically impossible values**, found via
  a dedicated validity audit (not just statistical outlier checks):
  - 849 rows where `current_emi_amount` exceeded `monthly_salary`
  - 1,066 rows where the regression target `max_monthly_emi` exceeded
    `monthly_salary` (an impossible target value)
  - 789 rows labeled `Eligible` despite negative disposable income
    (spending more than they earn) — a label/data contradiction
  - (some rows matched more than one condition; 2,052 is the deduplicated
    total removed)
- Also checked and ruled out: naive global IQR flagged `requested_amount`
  as 6.4% "outliers," but this was a false alarm from mixing 5 loan
  scenarios with legitimately different ranges — checked correctly
  (within each scenario), the real outlier rate is 0%.
- Validated: zero duplicates, zero missing values, zero out-of-range
  values, and zero logically-impossible values post-cleaning.
  **402,748 of 404,800 rows retained (99.49%).**

### 1.2 Exploratory Data Analysis
Key findings (full detail in `notebooks/eda_outputs/eda_summary_report.txt`):
- Eligibility is heavily imbalanced: 77.3% Not_Eligible, 18.4% Eligible,
  4.3% High_Risk — reflecting realistic conservative lending.
- Eligibility rate varies sharply by scenario: E-commerce/Home Appliances
  EMI (~26% eligible) vs. Vehicle/Personal Loan EMI (~11% eligible), since
  the latter involve larger amounts relative to typical income.
- Average credit score rises monotonically with eligibility outcome
  (Not_Eligible: 694 → High_Risk: 716 → Eligible: 725).
- `financial_health_score`, `credit_risk_score`, and `monthly_salary` are
  the strongest correlates of `max_monthly_emi`.

### 1.3 Feature Engineering
10 derived features were added on top of the 22 base variables:
- **Ratios**: debt-to-income, expense-to-income, affordability ratio,
  savings ratio, implied current-EMI burden ratio
- **Composite scores**: employment stability score (tenure weighted by
  employer type), credit risk score (normalized credit score adjusted for
  existing loans), financial health score (blended affordability/credit/
  savings)
- **Aggregates**: total monthly expenses, disposable income

All categorical variables were one-hot encoded (nominal) or ordinally
encoded (education, since it has a natural order); all numeric features
were median-imputed and standard-scaled. The full transform is a single
fitted `ColumnTransformer` (`models/feature_pipeline.joblib`), reused
identically for training and live app inference to avoid train/serve skew.

### 1.4 Model Development

**Classification (emi_eligibility, 3-class):**

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 81.1% | 0.677 | 0.776 | 0.661 | 0.940 |
| Random Forest | 91.0% | 0.716 | 0.785 | 0.739 | 0.965 |
| **HistGradientBoosting (selected, tuned, regularized)** | **97.9%** | **0.905** | **0.919** | **0.912** | **0.996** |

**Regression (max_monthly_emi):**

| Model | RMSE | MAE | R² | MAPE |
|---|---|---|---|---|
| Linear Regression | ₹3,942 | ₹2,833 | 0.737 | 186.1% |
| Random Forest | ₹876 | ₹286 | 0.987 | 5.7% |
| **HistGradientBoosting (selected, tuned)** | **₹541** | **₹212** | **0.995** | **6.3%** |

**Every number above improved after removing the 2,052 logically-invalid
rows** (classification accuracy 97.7%→97.9%, regression RMSE ₹624→₹541) —
confirming those rows were genuine label/data noise, not just harmless
edge cases.

Both selected models were evaluated on a held-out test set (15% of data,
never seen during training or model selection) to confirm generalization —
test performance matched validation performance closely, indicating no
overfitting to the validation set during model selection.

### 1.5 MLflow Integration
Every model run logs: model type, all hyperparameters, and the full metric
set, plus the fitted model as an artifact. The best model of each task is
additionally logged in a final run and registered to the MLflow Model
Registry (`emipredict_classifier`, `emipredict_regressor`) for versioned,
production-ready storage.

### 1.6 Application & Deployment
Multi-page Streamlit app: real-time prediction, interactive EDA dashboard,
model performance/MLflow comparison view, and an admin CRUD interface for
managing submitted applications. Designed for Streamlit Cloud deployment
via GitHub integration (see README for steps).

## 2. Business Impact

- **Automation potential**: replacing manual underwriting with a
  sub-second automated eligibility + EMI-capacity check directly
  supports the target of reducing manual processing time by ~80%.
- **Standardization**: a single model applied uniformly across all 5 EMI
  scenarios removes inconsistency between individual underwriters'
  judgment calls.
- **Risk-based pricing**: the `High_Risk` class (distinct from outright
  rejection) gives institutions a segment to offer at adjusted interest
  rates rather than a binary approve/reject decision.
- **Portfolio insight**: the EDA dashboard's scenario-level eligibility
  rates (e.g. Vehicle/Personal Loan EMI applicants are approved ~2.5x
  less often than E-commerce/Appliance EMI applicants) can directly
  inform which loan products to prioritize marketing for a given
  applicant's risk profile.

## 3. Recommendations for Production Use

- Retrain periodically as new applications and repayment outcomes
  accumulate, to capture drift in credit behavior.
- Consider a human-in-the-loop review specifically for the `High_Risk`
  band, where the model is least confident (lowest F1 of the 3 classes).
- If deployed at a financial institution, add fairness/bias auditing
  across demographic groups (age, gender, education) before production
  rollout — this build optimizes for accuracy/RMSE only.
