"""
EMIPredict AI - Main Streamlit entry point.
Run from the project root with: streamlit run app/Home.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
from app_utils import inject_custom_css, render_stat_card, COLOR_TEAL, COLOR_AMBER, COLOR_CORAL, COLOR_SLATE

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="\U0001F4B0",
    layout="wide",
)
inject_custom_css()

st.markdown("""
<div class="emi-hero">
    <h1>\U0001F4B0 EMIPredict AI</h1>
    <p>Intelligent Financial Risk Assessment Platform — real-time EMI eligibility
    and loan capacity prediction, trained on 400,000+ financial profiles
    across 5 lending scenarios.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(render_stat_card("Records Analyzed", "402,748", COLOR_TEAL), unsafe_allow_html=True)
with c2:
    st.markdown(render_stat_card("Input Features", "22 + 10", COLOR_SLATE), unsafe_allow_html=True)
with c3:
    st.markdown(render_stat_card("Classification Accuracy", "97.9%", COLOR_TEAL), unsafe_allow_html=True)
with c4:
    st.markdown(render_stat_card("Regression RMSE", "\u20b9541", COLOR_AMBER), unsafe_allow_html=True)

st.write("")
st.write("")

col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("### What this app does")
    st.markdown("""
- **Predict** — get real-time EMI eligibility and maximum safe monthly
  EMI amount for any applicant profile
- **Explore Data** — interactive charts on eligibility patterns, credit
  score distributions, and financial ratios across the dataset
- **Model Performance** — compare all trained classification and
  regression models, with MLflow experiment tracking details
- **Admin** — manage and inspect submitted applications
""")

    st.markdown("### How it works")
    st.markdown("""
Two models work together under the hood:
1. A **classification model** predicts whether an applicant is
   `Eligible`, `High_Risk`, or `Not_Eligible`
2. A **regression model** predicts the applicant's **maximum safe
   monthly EMI** in INR, based on their full financial profile

Both were selected from a pool of 4 candidate models each (Logistic/
Linear Regression, Random Forest, XGBoost, and Gradient Boosting),
trained and compared using MLflow experiment tracking.
""")

with col_right:
    st.markdown("### Eligibility signal")
    st.markdown(f"""
<div class="emi-stat-card" style="border-left-color:{COLOR_TEAL}; margin-bottom:0.7em;">
    <span class="emi-badge" style="background:{COLOR_TEAL};">Eligible</span>
    <div style="color:{COLOR_SLATE}; font-size:0.85em; margin-top:0.5em;">
        Low risk, comfortable EMI affordability
    </div>
</div>
<div class="emi-stat-card" style="border-left-color:{COLOR_AMBER}; margin-bottom:0.7em;">
    <span class="emi-badge" style="background:{COLOR_AMBER};">High Risk</span>
    <div style="color:{COLOR_SLATE}; font-size:0.85em; margin-top:0.5em;">
        Marginal case, higher interest rate recommended
    </div>
</div>
<div class="emi-stat-card" style="border-left-color:{COLOR_CORAL};">
    <span class="emi-badge" style="background:{COLOR_CORAL};">Not Eligible</span>
    <div style="color:{COLOR_SLATE}; font-size:0.85em; margin-top:0.5em;">
        High risk, loan not recommended at requested terms
    </div>
</div>
""", unsafe_allow_html=True)

st.info("Use the sidebar to navigate between pages.")
