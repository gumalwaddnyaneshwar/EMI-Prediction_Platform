"""EMIPredict AI - Model Performance & MLflow Dashboard Page"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import pandas as pd
from app_utils import inject_custom_css

st.set_page_config(page_title="Model Performance | EMIPredict AI", layout="wide")
inject_custom_css()
st.title("\U0001F4C8 Model Performance & Experiment Tracking")

st.write(
    "All models below were trained on the same train/validation/test split "
    "and tracked in MLflow (params, metrics, and model artifacts logged per run)."
)

tab1, tab2 = st.tabs(["Classification (EMI Eligibility)", "Regression (Max EMI)"])

with tab1:
    try:
        clf_results = pd.read_csv("models/classification_results.csv", index_col=0)
        clf_results = clf_results.sort_values("f1_macro", ascending=False)
        st.dataframe(clf_results.style.highlight_max(
            subset=["accuracy", "f1_macro", "roc_auc_ovr"], color="lightgreen"
        ), width="stretch")
        best = clf_results.index[0]
        st.success(f"**Selected model:** {best} "
                   f"(accuracy: {clf_results.loc[best, 'accuracy']:.1%}, "
                   f"F1-macro: {clf_results.loc[best, 'f1_macro']:.3f})")
        st.bar_chart(clf_results[["accuracy", "f1_macro", "roc_auc_ovr"]])
    except FileNotFoundError:
        st.warning("No classification results yet. Run `python src/train_classification.py`.")

with tab2:
    try:
        reg_results = pd.read_csv("models/regression_results.csv", index_col=0)
        reg_results = reg_results.sort_values("rmse")
        st.dataframe(reg_results.style.highlight_min(
            subset=["rmse", "mae", "mape"], color="lightgreen"
        ).highlight_max(subset=["r2"], color="lightgreen"), width="stretch")
        best = reg_results.index[0]
        st.success(f"**Selected model:** {best} "
                   f"(RMSE: \u20b9{reg_results.loc[best, 'rmse']:,.0f}, "
                   f"R\u00b2: {reg_results.loc[best, 'r2']:.3f})")
        st.bar_chart(reg_results[["rmse", "mae"]])
    except FileNotFoundError:
        st.warning("No regression results yet. Run `python src/train_regression.py`.")

st.divider()
st.subheader("MLflow Experiment Tracking")
st.markdown("""
For the full interactive MLflow UI — parameter comparisons, metric plots
across runs, and the model registry — run this from the project root:

```bash
mlflow ui
```

Then open **http://localhost:5000** in your browser. Two experiments are
tracked: `EMIPredict_Classification` and `EMIPredict_Regression`, each
containing one run per model plus a final `BEST_*_test_eval` run that
registers the selected model to the Model Registry.
""")
