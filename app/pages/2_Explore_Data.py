"""EMIPredict AI - Interactive Data Exploration Page"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import pandas as pd
import plotly.express as px
from app_utils import load_reference_data, load_dataset_summary, inject_custom_css

st.set_page_config(page_title="Explore Data | EMIPredict AI", layout="wide")
inject_custom_css()
st.title("\U0001F4CA Data Exploration")

try:
    df = load_reference_data()
    summary = load_dataset_summary()
except FileNotFoundError:
    st.error("Processed dataset not found. Run the preprocessing and "
              "feature engineering scripts first (see README).")
    st.stop()

st.caption(
    f"Full training dataset: {summary['n_records']:,} records. Charts and "
    f"filters below use a 5% stratified sample ({len(df):,} records) for "
    f"fast, lightweight loading in the deployed app."
)

st.sidebar.header("Filters")
scenario_filter = st.sidebar.multiselect(
    "EMI Scenario", options=sorted(df["emi_scenario"].unique()),
    default=sorted(df["emi_scenario"].unique()),
)
df_filtered = df[df["emi_scenario"].isin(scenario_filter)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records (full dataset)", f"{summary['n_records']:,}")
c2.metric("Avg Credit Score", f"{summary['avg_credit_score']:.0f}")
c3.metric("Avg Monthly Salary", f"\u20b9{summary['avg_monthly_salary']:,.0f}")
c4.metric("Eligible %", f"{summary['eligible_pct']:.1f}%")

tab1, tab2, tab3, tab4 = st.tabs([
    "Eligibility Patterns", "Financial Ratios", "Demographics", "Raw Data",
])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df_filtered, x="emi_eligibility",
                            color_discrete_sequence=["#0FBF9F"],
                            title="Eligibility Distribution")
        st.plotly_chart(fig, width="stretch")
    with col2:
        ct = pd.crosstab(df_filtered["emi_scenario"], df_filtered["emi_eligibility"], normalize="index")
        fig = px.bar(ct, barmode="stack", title="Eligibility Rate by Scenario")
        st.plotly_chart(fig, width="stretch")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df_filtered, x="emi_eligibility", y="credit_score",
                     title="Credit Score by Eligibility")
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = px.box(df_filtered, x="emi_eligibility", y="debt_to_income_ratio",
                     title="Debt-to-Income Ratio by Eligibility")
        st.plotly_chart(fig, width="stretch")

    fig = px.scatter(df_filtered.sample(min(5000, len(df_filtered))),
                      x="affordability_ratio", y="max_monthly_emi",
                      color="emi_eligibility", opacity=0.5,
                      title="Affordability Ratio vs Max Monthly EMI (5K sample)")
    st.plotly_chart(fig, width="stretch")

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df_filtered, x="age", color="emi_eligibility",
                            title="Age Distribution by Eligibility")
        st.plotly_chart(fig, width="stretch")
    with col2:
        edu_order = ["High School", "Graduate", "Post Graduate", "Professional"]
        ct = pd.crosstab(df_filtered["education"], df_filtered["emi_eligibility"],
                          normalize="index").reindex(edu_order)
        fig = px.bar(ct, barmode="stack", title="Eligibility Rate by Education")
        st.plotly_chart(fig, width="stretch")

with tab4:
    st.dataframe(df_filtered.head(500), width="stretch")
    st.caption(f"Showing first 500 of {len(df_filtered):,} filtered records.")
