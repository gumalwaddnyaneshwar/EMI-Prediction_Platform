# EMIPredict AI — Intelligent Financial Risk Assessment Platform

A machine-learning platform that predicts **EMI eligibility** (classification)
and **maximum safe monthly EMI** (regression) from an applicant's financial
profile, with full MLflow experiment tracking and a multi-page Streamlit app.

Built on 404,800 real financial records across 5 EMI scenarios (E-commerce,
Home Appliances, Vehicle, Personal Loan, Education).

## Results

| Task | Best Model | Key Metric | Target | Achieved |
|---|---|---|---|---|
| Classification (eligibility) | HistGradientBoosting (tuned, regularized, High_Risk-weighted) | Accuracy | >90% | **97.9%** |
| Regression (max EMI) | HistGradientBoosting (tuned) | RMSE | <₹2,000 | **₹541** |

Trained on 402,748 records (404,800 raw records, minus 2,052 rows removed for
logical data-quality issues — see "Data Quality" section below).

Full comparison across all 4 models per task is in `models/classification_results.csv`
and `models/regression_results.csv`.

## Project Structure

```
emipredict-ai/
├── data/
│   ├── raw/                    # original dataset
│   └── processed/              # cleaned + feature-engineered data
├── src/
│   ├── data_preprocessing.py   # cleaning & validation
│   ├── feature_engineering.py  # derived ratios + encoding pipeline
│   ├── train_utils.py          # shared split/pipeline helpers
│   ├── train_classification.py # 4 classifiers + MLflow tracking
│   ├── train_regression.py     # 4 regressors + MLflow tracking
│   └── eda.py                  # exploratory analysis + visualizations
├── app/
│   ├── Home.py                 # Streamlit entry point
│   ├── app_utils.py            # shared inference logic
│   └── pages/
│       ├── 1_Predict.py        # real-time prediction form
│       ├── 2_Explore_Data.py   # interactive EDA dashboard
│       ├── 3_Model_Performance.py  # model comparison + MLflow info
│       └── 4_Admin.py          # CRUD for applicant records
├── models/                     # trained models + result CSVs (generated)
├── notebooks/eda_outputs/      # EDA charts + summary report (generated)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place your dataset at `data/raw/emi_prediction_dataset.csv` (already done
if you're continuing from this handoff).

## Running the pipeline (in order)

```bash
# 1. Clean and validate the raw data
python src/data_preprocessing.py

# 2. Engineer features + fit the preprocessing pipeline
python src/feature_engineering.py

# 3. Generate the small sample + summary stats the DEPLOYED app uses
#    (the full featured/clean CSVs are git-ignored - too large for GitHub
#    and unnecessary to ship to Streamlit Cloud; re-run this any time you
#    regenerate the full dataset above, so the app's sample stays in sync)
python src/generate_app_data.py

# 4. Generate EDA charts + summary report
python src/eda.py

# 5. Train classification models (Logistic Regression, Random Forest,
#    XGBoost, HistGradientBoosting) with MLflow tracking
python src/train_classification.py

# 6. Train regression models (Linear Regression, Random Forest,
#    XGBoost, HistGradientBoosting) with MLflow tracking
python src/train_regression.py
```

Steps 4 and 5 log every run to MLflow and save the best model of each
type to `models/best_classifier.joblib` and `models/best_regressor.joblib`,
which the Streamlit app loads for predictions.

> **Note on compute:** `train_regression.py` also accepts a single model
> name as an argument (e.g. `python src/train_regression.py RandomForestRegressor`)
> if you want to train models one at a time, followed by
> `python src/train_regression.py finalize` to pick the best and register it.
> Useful on slower machines or limited-resource environments.

## Viewing MLflow experiments

```bash
mlflow ui
```
Open **http://localhost:5000** — you'll see two experiments
(`EMIPredict_Classification`, `EMIPredict_Regression`), each with one run
per model plus a final run that registers the selected best model.

## Running the Streamlit app locally

```bash
streamlit run app/Home.py
```

## Deploying to Streamlit Cloud

1. **Before pushing**, run through the pre-push checklist below.
2. Push this repository to GitHub. `models/*.joblib` and `models/*_results.csv`
   ARE committed (small, needed for the app) — `data/raw/` and the full
   `data/processed/emi_dataset_*.csv` files are NOT (git-ignored, too large;
   the app uses the small `emi_dataset_sample.csv` + `dataset_summary.json`
   instead — see `src/generate_app_data.py`).
3. Go to [share.streamlit.io](https://share.streamlit.io), connect your
   GitHub account, and select this repo.
4. Set the **main file path** to `app/Home.py`.
5. Deploy. Streamlit Cloud installs `requirements.txt` automatically.

### Pre-push checklist

- [ ] `git status` shows no file over ~50MB staged (GitHub hard-blocks >100MB)
- [ ] `data/processed/emi_dataset_sample.csv` and `dataset_summary.json`
      exist and ARE staged (the app needs these — they're small and safe to commit)
- [ ] `models/best_classifier.joblib`, `models/best_regressor.joblib`,
      `models/feature_pipeline.joblib`, `models/label_encoder.joblib`,
      and both `*_results.csv` files ARE staged (the app needs these)
- [ ] No stray `models/regressor_<ModelName>.joblib` intermediate files
      staged (only the two `best_*.joblib` files should be there)
- [ ] `data/processed/admin_records.csv` is NOT staged (it's your local
      test data, git-ignored by default — don't force-add it)
- [ ] `mlruns/` and `venv/` are NOT staged (git-ignored by default)
- [ ] No API keys, passwords, or `.streamlit/secrets.toml` staged

> Note: `models/*.joblib` files are small enough here (a few MB each) that
> Git LFS isn't needed. If you retrain with a much larger config in the
> future and files grow past ~50MB, consider
> [Git LFS](https://git-lfs.github.com/) at that point.

## Documentation

- `docs/PROJECT_REPORT.md` — full methodology, architecture, and results
- `docs/RIGOR_EXTENSIONS.md` — 5-fold CV, SMOTE comparison, explainability, fairness audit
- `docs/FINAL_DOCUMENTATION.md` — **limitations, advantages, future work, and business recommendations** (read this for the honest final assessment)

## Notes on this build

- Classification and regression models were trained on the **same
  stratified 70/15/15 train/val/test split** (stratified on
  `emi_eligibility`) so both tasks share consistent data partitions.
- `RandomForestRegressor` uses `max_samples=0.2` for faster training on
  large data — increase or remove this on a machine with more CPU cores
  for a marginally stronger model.
- The Admin page manages a separate `admin_records.csv` of new
  applications submitted through the app — it does not modify the
  400K-row training dataset.
