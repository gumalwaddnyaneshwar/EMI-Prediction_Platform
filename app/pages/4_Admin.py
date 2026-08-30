"""EMIPredict AI - Admin Data Management Page (CRUD)"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import pandas as pd
from app_utils import (predict, EMI_SCENARIOS, EDUCATION_LEVELS, EMPLOYMENT_TYPES,
                        COMPANY_TYPES, HOUSE_TYPES, inject_custom_css)

st.set_page_config(page_title="Admin | EMIPredict AI", layout="wide")
inject_custom_css()
st.title("\U0001F6E0\uFE0F Admin: Application Records")
st.write(
    "Manage submitted loan applications (Create, Read, Update, Delete). "
    "This is separate from the training dataset — it represents new "
    "applications processed through this platform."
)

RECORDS_PATH = "data/processed/admin_records.csv"
COLUMNS = [
    "applicant_name", "age", "gender", "monthly_salary", "credit_score",
    "emi_scenario", "requested_amount", "requested_tenure",
    "predicted_eligibility", "predicted_max_emi",
]


def load_records():
    if os.path.exists(RECORDS_PATH):
        return pd.read_csv(RECORDS_PATH)
    return pd.DataFrame(columns=COLUMNS)


def save_records(df):
    os.makedirs(os.path.dirname(RECORDS_PATH), exist_ok=True)
    df.to_csv(RECORDS_PATH, index=False)


records = load_records()

tab_create, tab_view, tab_manage = st.tabs(["\u2795 Create", "\U0001F4CB View All", "\u270F\uFE0F Update / Delete"])

with tab_create:
    st.subheader("Add New Application")
    with st.form("create_form"):
        c1, c2, c3 = st.columns(3)
        applicant_name = c1.text_input("Applicant Name")
        age = c2.number_input("Age", 18, 100, 35)
        gender = c3.selectbox("Gender", ["Male", "Female"])

        c1, c2, c3 = st.columns(3)
        monthly_salary = c1.number_input("Monthly Salary", 0, value=50000, step=1000)
        credit_score = c2.number_input("Credit Score", 300, 850, 700)
        emi_scenario = c3.selectbox("EMI Scenario", EMI_SCENARIOS)

        c1, c2 = st.columns(2)
        requested_amount = c1.number_input("Requested Amount", 1000, value=200000, step=5000)
        requested_tenure = c2.number_input("Requested Tenure (months)", 3, 84, 24)

        submitted = st.form_submit_button("Save & Predict", type="primary")

    if submitted:
        if not applicant_name.strip():
            st.error("Applicant name is required.")
        else:
            raw = dict(
                age=age, gender=gender, marital_status="Single", education="Graduate",
                monthly_salary=float(monthly_salary), employment_type="Private",
                years_of_employment=5.0, company_type="Mid-size", house_type="Rented",
                monthly_rent=0.0, family_size=2, dependents=0, school_fees=0.0,
                college_fees=0.0, travel_expenses=3000.0, groceries_utilities=8000.0,
                other_monthly_expenses=3000.0, existing_loans="No", current_emi_amount=0.0,
                credit_score=float(credit_score), bank_balance=100000.0, emergency_fund=50000.0,
                emi_scenario=emi_scenario, requested_amount=float(requested_amount),
                requested_tenure=requested_tenure,
            )
            try:
                result = predict(raw)
            except FileNotFoundError:
                st.error("Models not trained yet — run the training scripts first.")
                st.stop()

            new_row = pd.DataFrame([{
                "applicant_name": applicant_name, "age": age, "gender": gender,
                "monthly_salary": monthly_salary, "credit_score": credit_score,
                "emi_scenario": emi_scenario, "requested_amount": requested_amount,
                "requested_tenure": requested_tenure,
                "predicted_eligibility": result["eligibility"],
                "predicted_max_emi": round(result["max_monthly_emi"], 2),
            }])
            records = pd.concat([records, new_row], ignore_index=True)
            save_records(records)
            st.success(f"Saved application for {applicant_name} — "
                       f"predicted: {result['eligibility']}")

with tab_view:
    st.subheader(f"All Applications ({len(records)})")
    if records.empty:
        st.info("No applications yet. Add one in the Create tab.")
    else:
        st.dataframe(records, width="stretch")
        st.download_button("Download as CSV", records.to_csv(index=False),
                            file_name="emi_applications.csv", mime="text/csv")

with tab_manage:
    st.subheader("Update or Delete a Record")
    if records.empty:
        st.info("No records to manage yet.")
    else:
        idx = st.selectbox(
            "Select record by row index",
            options=records.index.tolist(),
            format_func=lambda i: f"{i}: {records.loc[i, 'applicant_name']} "
                                   f"({records.loc[i, 'predicted_eligibility']})",
        )
        row = records.loc[idx]
        col1, col2 = st.columns(2)
        with col1:
            new_salary = st.number_input("Monthly Salary", 0, value=int(row["monthly_salary"]))
            new_credit = st.number_input("Credit Score", 300, 850, int(row["credit_score"]))
        with col2:
            if st.button("Update Record"):
                records.loc[idx, "monthly_salary"] = new_salary
                records.loc[idx, "credit_score"] = new_credit
                save_records(records)
                st.success("Record updated.")
                st.rerun()
            if st.button("Delete Record", type="secondary"):
                records = records.drop(idx).reset_index(drop=True)
                save_records(records)
                st.success("Record deleted.")
                st.rerun()
