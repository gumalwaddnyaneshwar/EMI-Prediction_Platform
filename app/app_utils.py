"""
Shared helpers for the EMIPredict AI Streamlit app.
Kept separate from feature_engineering.py so the app can import it
without needing the sklearn training dependencies at module load time
beyond what's already required for inference.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = "models"
DATA_PATH = "data/processed/emi_dataset_sample.csv"
SUMMARY_PATH = "data/processed/dataset_summary.json"

EMI_SCENARIOS = [
    "E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI",
    "Personal Loan EMI", "Education EMI",
]
EDUCATION_LEVELS = ["High School", "Graduate", "Post Graduate", "Professional"]

# --- Design tokens -----------------------------------------------------
# Palette grounded in the product itself: navy/indigo is the "institution"
# base, teal/amber/coral is the SAME 3-way signal the model actually
# outputs (Eligible / High_Risk / Not_Eligible), reused consistently
# across every page rather than invented as decoration.
COLOR_INK_NAVY = "#0B1F3A"
COLOR_DEEP_INDIGO = "#16294F"
COLOR_TEAL = "#0FBF9F"      # Eligible
COLOR_AMBER = "#F5A623"     # High_Risk
COLOR_CORAL = "#E85D5D"     # Not_Eligible
COLOR_CLOUD = "#F7F9FC"
COLOR_SLATE = "#5B6B82"

ELIGIBILITY_COLORS = {
    "Eligible": COLOR_TEAL,
    "High_Risk": COLOR_AMBER,
    "Not_Eligible": COLOR_CORAL,
}


def inject_custom_css():
    """Call once near the top of every page. Loads the shared type system
    (Space Grotesk for headers, Inter for body, JetBrains Mono for financial
    figures/tabular numerals) and the card/button/sidebar restyle."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        color: {COLOR_INK_NAVY};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLOR_INK_NAVY} 0%, {COLOR_DEEP_INDIGO} 100%);
    }}
    section[data-testid="stSidebar"] * {{
        color: #E8EDF7 !important;
    }}
    section[data-testid="stSidebar"] a {{
        border-radius: 8px;
        transition: background 0.15s ease;
    }}
    section[data-testid="stSidebar"] a:hover {{
        background: rgba(15, 191, 159, 0.15) !important;
    }}

    /* Buttons */
    div.stButton > button, button[kind="primary"], button[kind="formSubmit"] {{
        background: linear-gradient(135deg, {COLOR_TEAL} 0%, #0AA085 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6em 1.2em;
        box-shadow: 0 4px 14px rgba(15, 191, 159, 0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    div.stButton > button:hover, button[kind="primary"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(15, 191, 159, 0.45);
    }}

    /* Metric-style stat cards (see render_stat_card) */
    .emi-stat-card {{
        background: white;
        border-left: 4px solid {COLOR_TEAL};
        border-radius: 12px;
        padding: 1.1em 1.3em;
        box-shadow: 0 6px 18px rgba(11, 31, 58, 0.08);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .emi-stat-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 10px 26px rgba(11, 31, 58, 0.14);
    }}
    .emi-stat-label {{
        font-size: 0.8em;
        color: {COLOR_SLATE};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }}
    .emi-stat-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.9em;
        font-weight: 700;
        color: {COLOR_INK_NAVY};
        margin-top: 0.15em;
    }}

    /* Hero banner (Home page) */
    .emi-hero {{
        background: linear-gradient(135deg, {COLOR_INK_NAVY} 0%, {COLOR_DEEP_INDIGO} 60%, #1E3A5F 100%);
        border-radius: 18px;
        padding: 2.4em 2.2em;
        margin-bottom: 1.6em;
        box-shadow: 0 14px 34px rgba(11, 31, 58, 0.28);
    }}
    .emi-hero h1 {{
        color: white !important;
        font-size: 2.3em;
        margin-bottom: 0.1em;
    }}
    .emi-hero p {{
        color: #C7D3E8;
        font-size: 1.05em;
        margin-bottom: 0;
    }}

    /* Eligibility result badge - color is the model's actual signal */
    .emi-badge {{
        display: inline-block;
        padding: 0.35em 0.9em;
        border-radius: 999px;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
        color: white;
    }}

    /* Dataframe polish */
    div[data-testid="stDataFrame"] {{
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 14px rgba(11, 31, 58, 0.08);
    }}
    </style>
    """, unsafe_allow_html=True)


def render_stat_card(label: str, value: str, accent: str = COLOR_TEAL) -> str:
    """Returns HTML for one KPI card. Use with st.markdown(..., unsafe_allow_html=True)."""
    return f"""
    <div class="emi-stat-card" style="border-left-color:{accent};">
        <div class="emi-stat-label">{label}</div>
        <div class="emi-stat-value">{value}</div>
    </div>
    """


def render_eligibility_badge(label: str) -> str:
    color = ELIGIBILITY_COLORS.get(label, COLOR_SLATE)
    return f'<span class="emi-badge" style="background:{color};">{label}</span>'
EMPLOYMENT_TYPES = ["Private", "Government", "Self-employed"]
COMPANY_TYPES = ["Mid-size", "MNC", "Startup", "Large Indian", "Small"]
HOUSE_TYPES = ["Rented", "Family", "Own"]


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(f"{MODELS_DIR}/feature_pipeline.joblib")
    classifier = joblib.load(f"{MODELS_DIR}/best_classifier.joblib")
    regressor = joblib.load(f"{MODELS_DIR}/best_regressor.joblib")
    label_encoder = joblib.load(f"{MODELS_DIR}/label_encoder.joblib")
    return preprocessor, classifier, regressor, label_encoder


@st.cache_data
def load_reference_data():
    """Loads a 5% stratified sample (~20K rows) for the Explore Data page.
    Kept small deliberately so it can ship in the GitHub repo and load fast
    on Streamlit Cloud - the full 400K-row dataset (124MB) is used for
    training locally but is too large to commit or serve from the deployed
    app. See load_dataset_summary() for headline stats computed on the
    FULL dataset, so those numbers stay exact even though charts sample."""
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_dataset_summary():
    import json
    with open(SUMMARY_PATH) as f:
        return json.load(f)


def engineer_single_record(raw: dict) -> pd.DataFrame:
    """Apply the same feature engineering formulas from
    src/feature_engineering.py to a single user-submitted record."""
    df = pd.DataFrame([raw])

    df["total_monthly_expenses"] = (
        df["monthly_rent"] + df["school_fees"] + df["college_fees"]
        + df["travel_expenses"] + df["groceries_utilities"]
        + df["other_monthly_expenses"] + df["current_emi_amount"]
    )
    df["disposable_income"] = df["monthly_salary"] - df["total_monthly_expenses"]
    df["debt_to_income_ratio"] = (
        df["current_emi_amount"] / df["monthly_salary"].replace(0, np.nan)
    ).fillna(0)
    df["expense_to_income_ratio"] = (
        df["total_monthly_expenses"] / df["monthly_salary"].replace(0, np.nan)
    ).fillna(0)
    df["affordability_ratio"] = (
        df["disposable_income"] / df["monthly_salary"].replace(0, np.nan)
    ).fillna(0)
    df["savings_ratio"] = (
        (df["bank_balance"] + df["emergency_fund"]) / df["monthly_salary"].replace(0, np.nan)
    ).fillna(0)

    stability_weight = df["employment_type"].map(
        {"Government": 1.2, "Private": 1.0, "Self-employed": 0.8}
    ).fillna(1.0)
    df["employment_stability_score"] = df["years_of_employment"] * stability_weight

    normalized_credit = (df["credit_score"] - 300) / (850 - 300)
    loan_penalty = df["existing_loans"].map({"Yes": -0.15, "No": 0.0}).fillna(0)
    df["credit_risk_score"] = (normalized_credit + loan_penalty).clip(0, 1)

    df["financial_health_score"] = (
        0.4 * df["affordability_ratio"].clip(-1, 1).add(1).div(2)
        + 0.3 * df["credit_risk_score"]
        + 0.3 * df["savings_ratio"].clip(0, 5).div(5)
    ).clip(0, 1)

    implied_new_emi = df["requested_amount"] / df["requested_tenure"].replace(0, np.nan)
    df["implied_current_emi_burden_ratio"] = (
        df["current_emi_amount"] / implied_new_emi.replace(0, np.nan)
    ).fillna(0).clip(0, 10)

    return df


def predict(raw: dict):
    preprocessor, classifier, regressor, label_encoder = load_artifacts()
    featured = engineer_single_record(raw)
    X = preprocessor.transform(featured)

    class_idx = classifier.predict(X)[0]
    class_proba = classifier.predict_proba(X)[0]
    eligibility = label_encoder.inverse_transform([class_idx])[0]
    proba_dict = dict(zip(label_encoder.classes_, class_proba))

    max_emi = float(regressor.predict(X)[0])

    return {
        "eligibility": eligibility,
        "probabilities": proba_dict,
        "max_monthly_emi": max_emi,
    }
