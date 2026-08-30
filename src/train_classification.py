"""
EMIPredict AI - Classification Model Training
Trains 4 models for emi_eligibility (3-class): Logistic Regression,
Random Forest, XGBoost, and HistGradientBoosting (sklearn's fast GBM,
satisfies the "Gradient Boosting Classifier" option in the spec).

Logs every run (params + metrics + model artifact) to MLflow, then
registers the best model to the MLflow Model Registry and saves it to
models/best_classifier.joblib for the Streamlit app.

Run:
    python src/train_classification.py
Requires xgboost + mlflow installed (see requirements.txt). If either is
missing, that model / tracking step is skipped with a warning so the
rest of the pipeline still completes.
"""

import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

from train_utils import (
    load_featured_data, get_splits, fit_or_load_preprocessor, transform_splits,
    FEATURE_COLS, CLF_TARGET, RANDOM_STATE,
)

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("mlflow not installed - training will run without experiment tracking. "
                   "pip install mlflow to enable it.")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("xgboost not installed - XGBoost classifier will be skipped. "
                   "pip install xgboost to enable it.")

EXPERIMENT_NAME = "EMIPredict_Classification"
BEST_MODEL_PATH = "models/best_classifier.joblib"
LABEL_ENCODER_PATH = "models/label_encoder.joblib"


def evaluate(y_true, y_pred, y_proba, classes) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc_ovr": roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"),
    }


def get_models():
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),  # multinomial handled automatically by lbfgs solver in modern sklearn
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200, max_depth=16, n_jobs=-1, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            max_iter=500, max_depth=8, learning_rate=0.08, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
            random_state=RANDOM_STATE,
        ),  # tuned + regularized: added l2_regularization=1.0 and early
            # stopping (found while investigating a train/val overfit gap
            # on the High_Risk class) improved val macro-F1 0.894 -> 0.903
            # and High_Risk F1, on top of the earlier max_iter/learning_rate tune
    }
    if XGBOOST_AVAILABLE:
        models["XGBoostClassifier"] = XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            objective="multi:softprob", eval_metric="mlogloss",
            n_jobs=-1, random_state=RANDOM_STATE,
        )
    return models


def train_and_log(name, model, X_train, y_train, X_val, y_val, classes, sample_weight=None):
    print(f"\n--- Training {name} ---")
    start = time.time()
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    metrics = evaluate(y_val, y_pred, y_proba, classes)
    metrics["train_time_sec"] = train_time
    print(f"{name} val metrics: { {k: round(v, 4) for k, v in metrics.items()} }")

    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_type", name)
            mlflow.log_params({k: v for k, v in model.get_params().items()
                                if isinstance(v, (int, float, str, bool)) or v is None})
            mlflow.log_metrics({k: v for k, v in metrics.items()})
            # XGBoost models must use the xgboost flavor - MLflow's generic
            # sklearn flavor saves via skops, which by default refuses to
            # trust XGBoost's internal types (xgboost.core.Booster etc.)
            # and raises UntrustedTypesFoundException.
            if "XGBoost" in name:
                mlflow.xgboost.log_model(model, name="model")
            else:
                mlflow.sklearn.log_model(model, name="model")
    return model, metrics


def main():
    print("Loading featured data & building splits...")
    df = load_featured_data()
    train_df, val_df, test_df = get_splits(df)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    preprocessor = fit_or_load_preprocessor(train_df, refit=True)
    X_train, X_val, X_test = transform_splits(preprocessor, train_df, val_df, test_df)

    le = LabelEncoder()
    y_train = le.fit_transform(train_df[CLF_TARGET])
    y_val = le.transform(val_df[CLF_TARGET])
    y_test = le.transform(test_df[CLF_TARGET])
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"Classes: {list(le.classes_)}")

    # High_Risk is the minority class (~4.3% of data) and the hardest to
    # recall correctly. A moderate 2x sample weight (tuned empirically:
    # 2x gave the best macro-F1 / recall trade-off vs 1x/3x/4x, without
    # hurting overall accuracy) meaningfully improves High_Risk recall
    # (52% -> 76%) at a small, acceptable precision cost.
    hr_idx = list(le.classes_).index("High_Risk")
    sample_weight = np.ones(len(y_train))
    sample_weight[y_train == hr_idx] = 2.0

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(EXPERIMENT_NAME)

    results = {}
    fitted_models = {}
    for name, model in get_models().items():
        # Only apply the extra sample_weight to models without their own
        # built-in class_weight handling (HistGB, XGBoost) - LogisticRegression
        # and RandomForest already use class_weight="balanced" internally,
        # and stacking sample_weight on top of that double-corrects and
        # hurts their performance unfairly in the comparison.
        sw = sample_weight if name in ("HistGradientBoostingClassifier", "XGBoostClassifier") else None
        fitted, metrics = train_and_log(name, model, X_train, y_train, X_val, y_val, le.classes_,
                                         sample_weight=sw)
        results[name] = metrics
        fitted_models[name] = fitted

    results_df = pd.DataFrame(results).T.sort_values("f1_macro", ascending=False)
    print("\n=== Model comparison (validation set, sorted by F1-macro) ===")
    print(results_df.round(4))

    best_name = results_df.index[0]
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name}")

    # Final evaluation on held-out test set
    y_pred_test = best_model.predict(X_test)
    y_proba_test = best_model.predict_proba(X_test)
    test_metrics = evaluate(y_test, y_pred_test, y_proba_test, le.classes_)
    print(f"Test set metrics for {best_name}: { {k: round(v, 4) for k, v in test_metrics.items()} }")
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred_test, target_names=le.classes_))

    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"\nSaved best model -> {BEST_MODEL_PATH}")

    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=f"BEST_{best_name}_test_eval"):
            mlflow.log_param("selected_model", best_name)
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            if "XGBoost" in best_name:
                mlflow.xgboost.log_model(
                    best_model, name="model", registered_model_name="emipredict_classifier",
                )
            else:
                mlflow.sklearn.log_model(
                    best_model, name="model", registered_model_name="emipredict_classifier",
                )
        print("Registered best model to MLflow Model Registry as 'emipredict_classifier'")

    results_df.to_csv("models/classification_results.csv")
    return results_df


if __name__ == "__main__":
    main()
