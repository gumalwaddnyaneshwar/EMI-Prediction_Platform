"""Shared utilities for classification and regression training scripts."""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from feature_engineering import build_preprocessor, ALL_NUMERIC, NOMINAL_CATEGORICAL, ORDINAL_CATEGORICAL

FEATURED_PATH = "data/processed/emi_dataset_featured.csv"
PREPROCESSOR_PATH = "models/feature_pipeline.joblib"

FEATURE_COLS = ALL_NUMERIC + NOMINAL_CATEGORICAL + ORDINAL_CATEGORICAL
CLF_TARGET = "emi_eligibility"
REG_TARGET = "max_monthly_emi"

RANDOM_STATE = 42


def load_featured_data() -> pd.DataFrame:
    return pd.read_csv(FEATURED_PATH)


def get_splits(df: pd.DataFrame):
    """
    70/15/15 train/val/test split, stratified on emi_eligibility so class
    balance is preserved across all three sets for both the classification
    and regression problems (same rows used for both targets).
    """
    train_val, test = train_test_split(
        df, test_size=0.15, random_state=RANDOM_STATE, stratify=df[CLF_TARGET]
    )
    train, val = train_test_split(
        train_val, test_size=0.1765,  # 0.15/0.85 ~= 0.1765 -> 15% of original
        random_state=RANDOM_STATE, stratify=train_val[CLF_TARGET]
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def fit_or_load_preprocessor(train_df: pd.DataFrame, refit: bool = False):
    if refit:
        preprocessor = build_preprocessor()
        preprocessor.fit(train_df)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
    else:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    return preprocessor


def transform_splits(preprocessor, train_df, val_df, test_df):
    X_train = preprocessor.transform(train_df)
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)
    return X_train, X_val, X_test
