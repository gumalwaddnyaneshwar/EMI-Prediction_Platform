"""
EMIPredict AI - Generate lightweight app data.

Run this AFTER feature_engineering.py, locally, once you have the full
data/processed/emi_dataset_featured.csv. It produces two small files that
DO get committed to GitHub and used by the deployed Streamlit app's
Explore Data page:

- data/processed/emi_dataset_sample.csv  (~6MB, 5% stratified sample)
- data/processed/dataset_summary.json    (~1KB, exact stats from full data)

The full featured/clean/raw CSVs (70-125MB each) are in .gitignore and are
NOT committed - GitHub hard-blocks files over 100MB, and shipping a 124MB
CSV to Streamlit Cloud would be slow and memory-heavy for every user's
session anyway. This script is how those two small, safe-to-commit files
get produced from the (git-ignored) full dataset.

Run:
    python src/generate_app_data.py
"""

import json
import pandas as pd

FULL_DATA_PATH = "data/processed/emi_dataset_featured.csv"
SAMPLE_OUT_PATH = "data/processed/emi_dataset_sample.csv"
SUMMARY_OUT_PATH = "data/processed/dataset_summary.json"

SAMPLE_FRACTION = 0.05
RANDOM_STATE = 42


def main():
    print(f"Loading {FULL_DATA_PATH} ...")
    df = pd.read_csv(FULL_DATA_PATH)
    print(f"Full dataset: {len(df):,} rows")

    print(f"Building {SAMPLE_FRACTION:.0%} stratified sample (by emi_eligibility)...")
    sample = df.groupby("emi_eligibility", group_keys=False).apply(
        lambda g: g.sample(frac=SAMPLE_FRACTION, random_state=RANDOM_STATE)
    ).reset_index(drop=True)
    sample.to_csv(SAMPLE_OUT_PATH, index=False)
    print(f"Saved sample -> {SAMPLE_OUT_PATH} ({len(sample):,} rows)")

    summary = {
        "n_records": len(df),
        "avg_credit_score": round(df["credit_score"].mean()),
        "avg_monthly_salary": round(df["monthly_salary"].mean()),
        "eligible_pct": round((df["emi_eligibility"] == "Eligible").mean() * 100, 1),
    }
    with open(SUMMARY_OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary -> {SUMMARY_OUT_PATH}: {summary}")


if __name__ == "__main__":
    main()
