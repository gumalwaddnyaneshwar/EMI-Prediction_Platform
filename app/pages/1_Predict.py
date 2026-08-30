"""EMIPredict AI - Real-time Prediction Page"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from app_utils import (predict, EMI_SCENARIOS, EDUCATION_LEVELS, EMPLOYMENT_TYPES,
                        COMPANY_TYPES, HOUSE_TYPES, inject_custom_css, render_eligibility_badge,
                        ELIGIBILITY_COLORS)

st.set_page_config(page_title="Predict | EMIPredict AI", layout="wide")
inject_custom_css()
st.title("\U0001F52E Real-Time EMI Prediction")
st.write("Fill in an applicant's profile to get an instant eligibility decision and recommended maximum EMI.")

with st.form("prediction_form"):
    st.subheader("Personal & Demographic")
    c1, c2, c3, c4 = st.columns(4)
    age = c1.number_input("Age", min_value=18, max_value=100, value=35)
    gender = c2.selectbox("Gender", ["Male", "Female"])
    marital_status = c3.selectbox("Marital Status", ["Single", "Married"])
    education = c4.selectbox("Education", EDUCATION_LEVELS, index=1)

    st.subheader("Employment & Income")
    c1, c2, c3, c4 = st.columns(4)
    monthly_salary = c1.number_input("Monthly Salary (INR)", min_value=0, value=50000, step=1000)
    employment_type = c2.selectbox("Employment Type", EMPLOYMENT_TYPES)
    years_of_employment = c3.number_input("Years of Employment", min_value=0.0, value=5.0, step=0.5)
    company_type = c4.selectbox("Company Type", COMPANY_TYPES)

    st.subheader("Housing & Family")
    c1, c2, c3, c4 = st.columns(4)
    house_type = c1.selectbox("House Type", HOUSE_TYPES)
    monthly_rent = c2.number_input("Monthly Rent (INR)", min_value=0, value=0, step=500)
    family_size = c3.number_input("Family Size", min_value=1, max_value=15, value=3)
    dependents = c4.number_input("Dependents", min_value=0, max_value=10, value=1)

    st.subheader("Monthly Expenses")
    c1, c2, c3, c4, c5 = st.columns(5)
    school_fees = c1.number_input("School Fees", min_value=0, value=0, step=500)
    college_fees = c2.number_input("College Fees", min_value=0, value=0, step=500)
    travel_expenses = c3.number_input("Travel Expenses", min_value=0, value=3000, step=500)
    groceries_utilities = c4.number_input("Groceries & Utilities", min_value=0, value=8000, step=500)
    other_monthly_expenses = c5.number_input("Other Expenses", min_value=0, value=3000, step=500)

    st.subheader("Financial Status & Credit History")
    c1, c2, c3, c4, c5 = st.columns(5)
    existing_loans = c1.selectbox("Existing Loans?", ["No", "Yes"])
    current_emi_amount = c2.number_input("Current EMI Amount", min_value=0, value=0, step=500)
    credit_score = c3.number_input("Credit Score", min_value=300, max_value=850, value=700)
    bank_balance = c4.number_input("Bank Balance", min_value=0, value=100000, step=5000)
    emergency_fund = c5.number_input("Emergency Fund", min_value=0, value=50000, step=5000)

    st.subheader("Loan Application Details")
    c1, c2, c3 = st.columns(3)
    emi_scenario = c1.selectbox("EMI Scenario", EMI_SCENARIOS)
    requested_amount = c2.number_input("Requested Amount (INR)", min_value=1000, value=200000, step=5000)
    requested_tenure = c3.number_input("Requested Tenure (months)", min_value=3, max_value=84, value=24)

    submitted = st.form_submit_button("Predict Eligibility", type="primary", width="stretch")

if submitted:
    raw = dict(
        age=age, gender=gender, marital_status=marital_status, education=education,
        monthly_salary=float(monthly_salary), employment_type=employment_type,
        years_of_employment=years_of_employment, company_type=company_type,
        house_type=house_type, monthly_rent=float(monthly_rent),
        family_size=family_size, dependents=dependents,
        school_fees=float(school_fees), college_fees=float(college_fees),
        travel_expenses=float(travel_expenses), groceries_utilities=float(groceries_utilities),
        other_monthly_expenses=float(other_monthly_expenses), existing_loans=existing_loans,
        current_emi_amount=float(current_emi_amount), credit_score=float(credit_score),
        bank_balance=float(bank_balance), emergency_fund=float(emergency_fund),
        emi_scenario=emi_scenario, requested_amount=float(requested_amount),
        requested_tenure=requested_tenure,
    )

    try:
        result = predict(raw)
    except FileNotFoundError:
        st.error(
            "Trained model files not found in `models/`. Run "
            "`python src/train_classification.py` and "
            "`python src/train_regression.py` first."
        )
        st.stop()

    st.divider()
    st.subheader("Result")

    accent = ELIGIBILITY_COLORS[result["eligibility"]]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(render_eligibility_badge(result["eligibility"]), unsafe_allow_html=True)
        st.write("")
        for cls, p in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
            st.write(f"{cls}: {p * 100:.1f}%")
            st.progress(min(float(p), 1.0))
    with col2:
        st.markdown(f"""
        <div class="emi-stat-card" style="border-left-color:{accent};">
            <div class="emi-stat-label">Recommended Maximum Monthly EMI</div>
            <div class="emi-stat-value">\u20b9{result['max_monthly_emi']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        if result["max_monthly_emi"] < (requested_amount / requested_tenure):
            st.write("")
            st.warning(
                f"Note: the requested amount implies an EMI of "
                f"\u20b9{requested_amount / requested_tenure:,.0f}/month, above the "
                f"recommended safe maximum. Consider a longer tenure or lower amount."
            )
