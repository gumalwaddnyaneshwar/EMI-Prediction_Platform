"""
EMIPredict AI - Feature Engineering
Creates derived financial ratios and risk features, then builds a
reusable scikit-learn preprocessing pipeline (encoding + scaling) that
is fit once and reused identically for training and for live inference
in the Streamlit app.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

PROCESSED_PATH = "data/processed/emi_dataset_clean.csv"
FEATURED_PATH = "data/processed/emi_dataset_featured.csv"
PREPROCESSOR_PATH = "models/feature_pipeline.joblib"

EDUCATION_ORDER = ["High School", "Graduate", "Post Graduate", "Professional"]

NOMINAL_CATEGORICAL = [
    "gender", "marital_status", "employment_type", "company_type",
    "house_type", "existing_loans", "emi_scenario",
]
ORDINAL_CATEGORICAL = ["education"]

BASE_NUMERIC = [
    "age", "monthly_salary", "years_of_employment", "monthly_rent",
    "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "current_emi_amount", "credit_score", "bank_balance", "emergency_fund",
    "requested_amount", "requested_tenure",
]

# Engineered features added on top of BASE_NUMERIC
ENGINEERED_NUMERIC = [
    "total_monthly_expenses", "disposable_income", "debt_to_income_ratio",
    "expense_to_income_ratio", "affordability_ratio", "savings_ratio",
    "employment_stability_score", "credit_risk_score", "financial_health_score",
    "implied_current_emi_burden_ratio",
]

ALL_NUMERIC = BASE_NUMERIC + ENGINEERED_NUMERIC


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Total fixed monthly obligations
    df["total_monthly_expenses"] = (
        df["monthly_rent"] + df["school_fees"] + df["college_fees"]
        + df["travel_expenses"] + df["groceries_utilities"]
        + df["other_monthly_expenses"] + df["current_emi_amount"]
    )

    # Disposable income after fixed obligations
    df["disposable_income"] = df["monthly_salary"] - df["total_monthly_expenses"]

    # Debt-to-income ratio (existing EMI burden vs income)
    df["debt_to_income_ratio"] = (
        df["current_emi_amount"] / df["monthly_salary"].replace(0, np.nan)
    ).fillna(0)

    # Expense-to-income ratio (all living costs vs income)
    df["expense_to_income_ratio"] = (
        df["total_monthly_expenses"] / df["monthly_salary"].replace(0, np.nan)
    ).fillna(0)

    # Affordability ratio: how much of income remains free relative to income
    df["affordability_ratio"] = (
        df["disposable_income"] / df["monthly_salary"].replace(0, np.nan)
    ).fillna(0)

    # Savings ratio: safety cushion relative to income
    df["savings_ratio"] = (
        (df["bank_balance"] + df["emergency_fund"]) / df["monthly_salary"].replace(0, np.nan)
    ).fillna(0)

    # Employment stability: longer tenure + government/MNC treated as more stable
    stability_weight = df["employment_type"].map(
        {"Government": 1.2, "Private": 1.0, "Self-employed": 0.8}
    ).fillna(1.0)
    df["employment_stability_score"] = df["years_of_employment"] * stability_weight

    # Credit risk score: normalized credit score combined with existing loan flag
    normalized_credit = (df["credit_score"] - 300) / (850 - 300)
    loan_penalty = df["existing_loans"].map({"Yes": -0.15, "No": 0.0}).fillna(0)
    df["credit_risk_score"] = (normalized_credit + loan_penalty).clip(0, 1)

    # Composite financial health score (0-1): blends affordability, savings, credit
    df["financial_health_score"] = (
        0.4 * df["affordability_ratio"].clip(-1, 1).add(1).div(2)
        + 0.3 * df["credit_risk_score"]
        + 0.3 * df["savings_ratio"].clip(0, 5).div(5)
    ).clip(0, 1)

    # How much of requested EMI capacity is already used by current EMI
    implied_new_emi = df["requested_amount"] / df["requested_tenure"].replace(0, np.nan)
    df["implied_current_emi_burden_ratio"] = (
        df["current_emi_amount"] / implied_new_emi.replace(0, np.nan)
    ).fillna(0).clip(0, 10)

    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    nominal_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    ordinal_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=[EDUCATION_ORDER],
                                    handle_unknown="use_encoded_value",
                                    unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, ALL_NUMERIC),
        ("nom", nominal_pipeline, NOMINAL_CATEGORICAL),
        ("ord", ordinal_pipeline, ORDINAL_CATEGORICAL),
    ])
    return preprocessor


if __name__ == "__main__":
    print("Loading cleaned data...")
    df = pd.read_csv(PROCESSED_PATH)

    print("Engineering features...")
    featured = engineer_features(df)
    featured.to_csv(FEATURED_PATH, index=False)
    print(f"Saved -> {FEATURED_PATH} with {len(ENGINEERED_NUMERIC)} new features")
    print(featured[ENGINEERED_NUMERIC].describe().T[["mean", "std", "min", "max"]])

    print("\nFitting preprocessing pipeline (encoding + scaling)...")
    preprocessor = build_preprocessor()
    preprocessor.fit(featured)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    transformed_shape = preprocessor.transform(featured.head(5)).shape
    print(f"Saved fitted pipeline -> {PREPROCESSOR_PATH}")
    print(f"Transformed feature vector width: {transformed_shape[1]}")
