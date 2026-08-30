"""
EMIPredict AI - Explainability & Fairness Analysis

Two things this script does:

1. Feature importance via permutation importance. True SHAP requires the
   `shap` package, which isn't installable in the offline dev sandbox this
   was built in - if you have internet, `pip install shap` and use
   shap.TreeExplainer instead for per-prediction attribution (permutation
   importance below only gives global, model-wide importance, not
   per-individual-prediction breakdowns like SHAP does).

2. A fairness audit: accuracy and approval-rate parity across gender, age,
   and education groups on the classifier's validation set.

Run:
    python analysis/explainability_fairness.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from train_utils import (
    load_featured_data, get_splits, fit_or_load_preprocessor, transform_splits,
    CLF_TARGET, REG_TARGET, ALL_NUMERIC, NOMINAL_CATEGORICAL, ORDINAL_CATEGORICAL,
)


def get_feature_names(preprocessor):
    return (
        ALL_NUMERIC
        + list(preprocessor.named_transformers_["nom"].named_steps["onehot"]
               .get_feature_names_out(NOMINAL_CATEGORICAL))
        + ORDINAL_CATEGORICAL
    )


def run_permutation_importance():
    df = load_featured_data()
    train_df, val_df, test_df = get_splits(df)
    preprocessor = fit_or_load_preprocessor(train_df, refit=False)
    feature_names = get_feature_names(preprocessor)

    rng = np.random.RandomState(42)

    print("=" * 60)
    print("CLASSIFICATION - Permutation Importance (top 15)")
    print("=" * 60)
    X_val = preprocessor.transform(val_df)
    le = joblib.load("models/label_encoder.joblib")
    y_val = le.transform(val_df[CLF_TARGET])
    clf = joblib.load("models/best_classifier.joblib")

    idx = rng.choice(len(X_val), size=min(8000, len(X_val)), replace=False)
    result = permutation_importance(clf, X_val[idx], y_val[idx], n_repeats=5,
                                     random_state=42, n_jobs=-1, scoring="f1_macro")
    ranked = sorted(zip(feature_names, result.importances_mean), key=lambda x: -x[1])
    for name, imp in ranked[:15]:
        print(f"  {name}: {imp:.4f}")

    print()
    print("=" * 60)
    print("REGRESSION - Permutation Importance (top 10)")
    print("=" * 60)
    y_val_reg = val_df[REG_TARGET].values
    reg = joblib.load("models/best_regressor.joblib")
    result = permutation_importance(reg, X_val[idx], y_val_reg[idx], n_repeats=5,
                                     random_state=42, n_jobs=-1, scoring="r2")
    ranked = sorted(zip(feature_names, result.importances_mean), key=lambda x: -x[1])
    for name, imp in ranked[:10]:
        print(f"  {name}: {imp:.4f}")

    return preprocessor, val_df, le, clf


def run_fairness_audit(preprocessor, val_df, le, clf):
    X_val = preprocessor.transform(val_df)
    y_val = le.transform(val_df[CLF_TARGET])
    pred = clf.predict(X_val)

    val_df = val_df.copy()
    val_df["pred"] = le.inverse_transform(pred)
    val_df["true"] = le.inverse_transform(y_val)
    val_df["correct"] = val_df["pred"] == val_df["true"]
    val_df["age_bin"] = pd.cut(val_df["age"], bins=[18, 30, 40, 50, 100],
                                labels=["18-30", "31-40", "41-50", "51+"])

    print()
    print("=" * 60)
    print("FAIRNESS AUDIT (validation set)")
    print("=" * 60)
    for group_col in ["gender", "education", "age_bin"]:
        print(f"\n--- By {group_col} ---")
        acc = val_df.groupby(group_col, observed=True)["correct"].mean().round(4)
        approval = val_df.groupby(group_col, observed=True)["pred"].apply(
            lambda s: (s == "Eligible").mean()
        ).round(4)
        summary = pd.DataFrame({"accuracy": acc, "approval_rate": approval})
        print(summary)


if __name__ == "__main__":
    preprocessor, val_df, le, clf = run_permutation_importance()
    run_fairness_audit(preprocessor, val_df, le, clf)
