"""
EMIPredict AI - Regression Model Training
Trains 4 models for max_monthly_emi: Linear Regression, Random Forest,
XGBoost, and HistGradientBoosting (sklearn's fast GBM, satisfies the
"Gradient Boosting Regressor" option in the spec).

Logs every run to MLflow, registers the best model, and saves it to
models/best_regressor.joblib for the Streamlit app.

Run:
    python src/train_regression.py
Requires xgboost + mlflow installed (see requirements.txt). If either is
missing, that model / tracking step is skipped with a warning so the
rest of the pipeline still completes.
"""

import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error,
)

from train_utils import (
    load_featured_data, get_splits, fit_or_load_preprocessor, transform_splits,
    REG_TARGET, RANDOM_STATE,
)

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("mlflow not installed - training will run without experiment tracking.")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("xgboost not installed - XGBoost regressor will be skipped.")

EXPERIMENT_NAME = "EMIPredict_Regression"
BEST_MODEL_PATH = "models/best_regressor.joblib"


def evaluate(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "rmse": rmse,
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


def get_models():
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=250, max_depth=18, n_jobs=-1,
            random_state=RANDOM_STATE,
        ),  # Full-strength config for multi-core machines. NOTE: this exact
            # config was NOT run to completion in the single-CPU dev sandbox
            # (it timed out) - only a reduced version (n_estimators=80,
            # max_depth=14, max_samples=0.3, RMSE 935) was verified there.
            # Run this on your own machine and compare against the results
            # in models/regression_results.csv.
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=350, max_depth=8, learning_rate=0.08, random_state=RANDOM_STATE,
        ),  # tuned: max_iter 250->350, learning_rate 0.1->0.08 improved
            # RMSE (640.7 -> 634.1) over the initial baseline config
    }
    if XGBOOST_AVAILABLE:
        models["XGBoostRegressor"] = XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            objective="reg:squarederror", n_jobs=-1, random_state=RANDOM_STATE,
        )
    return models


def train_and_log(name, model, X_train, y_train, X_val, y_val):
    print(f"\n--- Training {name} ---")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_val)
    metrics = evaluate(y_val, y_pred)
    metrics["train_time_sec"] = train_time
    print(f"{name} val metrics: { {k: round(v, 4) for k, v in metrics.items()} }")

    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)
            mlflow.log_params({k: v for k, v in model.get_params().items()
                                if isinstance(v, (int, float, str, bool)) or v is None})
            mlflow.log_metrics(metrics)
            if "XGBoost" in name:
                mlflow.xgboost.log_model(model, name="model")
            else:
                mlflow.sklearn.log_model(model, name="model")
    return model, metrics


def prep_data():
    print("Loading featured data & building splits...")
    df = load_featured_data()
    train_df, val_df, test_df = get_splits(df)  # same stratified split as classification
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    preprocessor = fit_or_load_preprocessor(train_df, refit=False)  # reuse classifier's fitted pipeline
    X_train, X_val, X_test = transform_splits(preprocessor, train_df, val_df, test_df)

    y_train = train_df[REG_TARGET].values
    y_val = val_df[REG_TARGET].values
    y_test = test_df[REG_TARGET].values
    return X_train, X_val, X_test, y_train, y_val, y_test


RESULTS_CSV = "models/regression_results.csv"


def train_one(model_name):
    """Train a single model and append its metrics to the results CSV.
    Lets the full model suite be trained across separate process runs."""
    X_train, X_val, X_test, y_train, y_val, y_test = prep_data()
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(EXPERIMENT_NAME)

    models = get_models()
    model = models[model_name]
    fitted, metrics = train_and_log(model_name, model, X_train, y_train, X_val, y_val)
    joblib.dump(fitted, f"models/regressor_{model_name}.joblib")

    try:
        results_df = pd.read_csv(RESULTS_CSV, index_col=0)
    except FileNotFoundError:
        results_df = pd.DataFrame(columns=list(metrics.keys()))
    results_df.loc[model_name] = metrics
    results_df.to_csv(RESULTS_CSV)
    print(f"Appended {model_name} metrics -> {RESULTS_CSV}")


def finalize_best():
    """Pick the best model by RMSE from all trained runs, evaluate on the
    held-out test set, save it as the production model, and register it."""
    results_df = pd.read_csv(RESULTS_CSV, index_col=0).sort_values("rmse")
    print("\n=== Model comparison (validation set, sorted by RMSE) ===")
    print(results_df.round(4))

    best_name = results_df.index[0]
    best_model = joblib.load(f"models/regressor_{best_name}.joblib")
    print(f"\nBest model: {best_name}")

    _, _, X_test, _, _, y_test = prep_data()
    y_pred_test = best_model.predict(X_test)
    test_metrics = evaluate(y_test, y_pred_test)
    print(f"Test set metrics for {best_name}: { {k: round(v, 4) for k, v in test_metrics.items()} }")

    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"Saved best model -> {BEST_MODEL_PATH}")

    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=f"BEST_{best_name}_test_eval"):
            mlflow.log_param("selected_model", best_name)
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            if "XGBoost" in best_name:
                mlflow.xgboost.log_model(
                    best_model, name="model", registered_model_name="emipredict_regressor",
                )
            else:
                mlflow.sklearn.log_model(
                    best_model, name="model", registered_model_name="emipredict_regressor",
                )
        print("Registered best model to MLflow Model Registry as 'emipredict_regressor'")
    return results_df


def main():
    """Full sequential run (all models, then finalize) - used when you have
    plenty of runtime, e.g. running locally or in Colab."""
    for name in get_models():
        train_one(name)
    finalize_best()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "finalize":
        finalize_best()
    elif len(sys.argv) > 1:
        train_one(sys.argv[1])
    else:
        main()
