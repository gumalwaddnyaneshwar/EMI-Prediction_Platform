"""
EMIPredict AI - Data Preprocessing Pipeline
Loads, cleans, and validates the raw EMI dataset.

Handles known data quality issues:
- Malformed numeric strings (e.g. "58.0.0" instead of "58.0") in age,
  monthly_salary, bank_balance
- Inconsistent categorical casing in gender (Male/male/M/MALE etc.)
- Out-of-range credit_score values (spec says 300-850)
- Missing values in education, monthly_rent, credit_score, bank_balance,
  emergency_fund
"""

import re
import numpy as np
import pandas as pd

RAW_PATH = "data/raw/emi_prediction_dataset.csv"
PROCESSED_PATH = "data/processed/emi_dataset_clean.csv"

# Columns that were exported with occasional doubled ".0.0" suffixes
MALFORMED_NUMERIC_COLS = ["age", "monthly_salary", "bank_balance"]

GENDER_MAP = {
    "male": "Male", "m": "Male", "MALE".lower(): "Male",
    "female": "Female", "f": "Female",
}

CREDIT_SCORE_MIN, CREDIT_SCORE_MAX = 300, 850


def fix_malformed_numeric(series: pd.Series) -> pd.Series:
    """Fix strings like '58.0.0' -> '58.0' before numeric conversion."""
    cleaned = series.astype(str).str.strip()
    # Collapse any run of repeated ".0" endings down to a single ".0"
    cleaned = cleaned.str.replace(r"(\.0){2,}$", ".0", regex=True)
    cleaned = cleaned.replace({"nan": np.nan, "None": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def standardize_gender(series: pd.Series) -> pd.Series:
    return series.str.strip().str.lower().map(GENDER_MAP).fillna(series)


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    initial_rows = len(df)

    # 1. Fix malformed numeric columns
    for col in MALFORMED_NUMERIC_COLS:
        df[col] = fix_malformed_numeric(df[col])

    # 2. Standardize gender casing
    df["gender"] = standardize_gender(df["gender"])

    # 3. Fix out-of-range credit_score (clip to valid domain per spec: 300-850)
    df.loc[df["credit_score"] > CREDIT_SCORE_MAX, "credit_score"] = np.nan
    df.loc[df["credit_score"] < CREDIT_SCORE_MIN, "credit_score"] = np.nan

    # 4. Handle missing values
    #    - education: categorical -> impute with mode
    df["education"] = df["education"].fillna(df["education"].mode()[0])
    #    - monthly_rent: NaN plausibly means "owns home / lives with family" -> 0
    df["monthly_rent"] = df["monthly_rent"].fillna(0)
    #    - credit_score: numeric -> impute with median
    df["credit_score"] = df["credit_score"].fillna(df["credit_score"].median())
    #    - bank_balance: numeric -> impute with median
    df["bank_balance"] = df["bank_balance"].fillna(df["bank_balance"].median())
    #    - emergency_fund: numeric -> impute with median
    df["emergency_fund"] = df["emergency_fund"].fillna(df["emergency_fund"].median())

    # 5. Enforce sane dtypes
    int_like = ["family_size", "dependents", "requested_tenure"]
    for col in int_like:
        df[col] = df[col].astype(int)

    # 6. Drop exact duplicate rows
    df = df.drop_duplicates()

    # 7. Drop rows where critical fields are still unusable (e.g. negative salary)
    df = df[df["monthly_salary"] > 0]
    df = df[df["age"].between(18, 100)]

    # 8. Drop rows with logically impossible values that pass basic type/range
    #    checks but are financially nonsensical:
    #    - current_emi_amount cannot exceed total monthly income
    #    - max_monthly_emi (the regression target) cannot exceed total income
    #    - a row can't be labeled "Eligible" while spending more than it earns
    #      (negative disposable income) - this is a label/data contradiction
    total_expenses = (
        df["monthly_rent"] + df["school_fees"] + df["college_fees"]
        + df["travel_expenses"] + df["groceries_utilities"]
        + df["other_monthly_expenses"] + df["current_emi_amount"]
    )
    disposable_income = df["monthly_salary"] - total_expenses

    impossible_current_emi = df["current_emi_amount"] > df["monthly_salary"]
    impossible_target_emi = df["max_monthly_emi"] > df["monthly_salary"]
    contradictory_label = (disposable_income < 0) & (df["emi_eligibility"] == "Eligible")

    invalid_mask = impossible_current_emi | impossible_target_emi | contradictory_label
    n_invalid = int(invalid_mask.sum())
    df = df[~invalid_mask]

    df = df.reset_index(drop=True)

    report = {
        "initial_rows": initial_rows,
        "final_rows": len(df),
        "rows_dropped": initial_rows - len(df),
        "rows_dropped_logically_invalid": n_invalid,
    }
    return df, report


def validate_data(df: pd.DataFrame) -> dict:
    """Post-cleaning quality assessment."""
    total_expenses = (
        df["monthly_rent"] + df["school_fees"] + df["college_fees"]
        + df["travel_expenses"] + df["groceries_utilities"]
        + df["other_monthly_expenses"] + df["current_emi_amount"]
    )
    disposable_income = df["monthly_salary"] - total_expenses

    checks = {
        "missing_values_total": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "credit_score_out_of_range": int(
            (~df["credit_score"].between(CREDIT_SCORE_MIN, CREDIT_SCORE_MAX)).sum()
        ),
        "negative_salary_rows": int((df["monthly_salary"] <= 0).sum()),
        "current_emi_exceeds_salary": int((df["current_emi_amount"] > df["monthly_salary"]).sum()),
        "max_emi_target_exceeds_salary": int((df["max_monthly_emi"] > df["monthly_salary"]).sum()),
        "eligible_with_negative_disposable_income": int(
            ((disposable_income < 0) & (df["emi_eligibility"] == "Eligible")).sum()
        ),
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "eligibility_class_balance": df["emi_eligibility"].value_counts(normalize=True).round(3).to_dict(),
    }
    return checks


if __name__ == "__main__":
    print("Loading raw data...")
    raw = load_raw()
    print(f"Raw shape: {raw.shape}")

    print("Cleaning data...")
    clean, report = clean_data(raw)
    print(f"Cleaning report: {report}")

    print("Validating...")
    checks = validate_data(clean)
    for k, v in checks.items():
        print(f"  {k}: {v}")

    clean.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved cleaned dataset -> {PROCESSED_PATH} ({clean.shape[0]:,} rows)")
