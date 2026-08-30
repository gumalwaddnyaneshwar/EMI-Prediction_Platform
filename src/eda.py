"""
EMIPredict AI - Exploratory Data Analysis
Generates key visualizations and a text summary of business insights
from the cleaned + featured dataset.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

FEATURED_PATH = "data/processed/emi_dataset_featured.csv"
OUTPUT_DIR = "notebooks/eda_outputs"

sns.set_style("whitegrid")


def eligibility_distribution(df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df["emi_eligibility"].value_counts().plot(
        kind="bar", ax=axes[0], color=["#2ca02c", "#ff7f0e", "#d62728"]
    )
    axes[0].set_title("EMI Eligibility Distribution")
    axes[0].set_ylabel("Count")

    pd.crosstab(df["emi_scenario"], df["emi_eligibility"], normalize="index").plot(
        kind="bar", stacked=True, ax=axes[1], colormap="viridis"
    )
    axes[1].set_title("Eligibility Rate by EMI Scenario")
    axes[1].set_ylabel("Proportion")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{outdir}/01_eligibility_distribution.png", dpi=110)
    plt.close()


def correlation_heatmap(df, outdir):
    numeric_cols = [
        "monthly_salary", "credit_score", "debt_to_income_ratio",
        "expense_to_income_ratio", "affordability_ratio", "savings_ratio",
        "credit_risk_score", "financial_health_score", "requested_amount",
        "max_monthly_emi",
    ]
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Between Key Financial Variables")
    plt.tight_layout()
    plt.savefig(f"{outdir}/02_correlation_heatmap.png", dpi=110)
    plt.close()
    return corr


def credit_score_vs_eligibility(df, outdir):
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="emi_eligibility", y="credit_score",
                order=["Not_Eligible", "High_Risk", "Eligible"])
    plt.title("Credit Score Distribution by Eligibility Outcome")
    plt.tight_layout()
    plt.savefig(f"{outdir}/03_credit_score_by_eligibility.png", dpi=110)
    plt.close()


def max_emi_by_scenario(df, outdir):
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="emi_scenario", y="max_monthly_emi")
    plt.xticks(rotation=30, ha="right")
    plt.title("Max Monthly EMI Capacity by Loan Scenario")
    plt.tight_layout()
    plt.savefig(f"{outdir}/04_max_emi_by_scenario.png", dpi=110)
    plt.close()


def demographic_patterns(df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(data=df, x="age", hue="emi_eligibility", multiple="stack",
                 bins=20, ax=axes[0], palette="viridis")
    axes[0].set_title("Age Distribution by Eligibility")

    edu_order = ["High School", "Graduate", "Post Graduate", "Professional"]
    pd.crosstab(df["education"], df["emi_eligibility"], normalize="index").loc[edu_order].plot(
        kind="bar", stacked=True, ax=axes[1], colormap="viridis"
    )
    axes[1].set_title("Eligibility Rate by Education Level")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(f"{outdir}/05_demographic_patterns.png", dpi=110)
    plt.close()


def generate_report(df, corr, outdir):
    lines = []
    lines.append("EMIPredict AI - Exploratory Data Analysis: Business Insights\n")
    lines.append("=" * 65 + "\n")

    total = len(df)
    elig_pct = (df["emi_eligibility"] == "Eligible").mean() * 100
    risk_pct = (df["emi_eligibility"] == "High_Risk").mean() * 100
    not_elig_pct = (df["emi_eligibility"] == "Not_Eligible").mean() * 100
    lines.append(f"Dataset: {total:,} records across 5 EMI scenarios.\n")
    lines.append(f"Eligibility split -> Eligible: {elig_pct:.1f}% | "
                  f"High_Risk: {risk_pct:.1f}% | Not_Eligible: {not_elig_pct:.1f}%\n")
    lines.append("The dataset is heavily imbalanced toward Not_Eligible applicants, "
                  "reflecting realistic conservative lending patterns.\n")

    strongest = corr["max_monthly_emi"].drop("max_monthly_emi").abs().sort_values(ascending=False)
    lines.append(f"\nStrongest predictors of max_monthly_emi: "
                  f"{', '.join(strongest.head(3).index)}.\n")

    by_scenario = df.groupby("emi_scenario")["emi_eligibility"].apply(
        lambda s: (s == "Eligible").mean() * 100
    ).sort_values(ascending=False)
    lines.append(f"\nEligibility rate by scenario (highest to lowest):\n")
    for scenario, pct in by_scenario.items():
        lines.append(f"  - {scenario}: {pct:.1f}% eligible\n")

    avg_credit_by_elig = df.groupby("emi_eligibility")["credit_score"].mean().round(0)
    lines.append(f"\nAverage credit score by outcome: {avg_credit_by_elig.to_dict()}\n")

    with open(f"{outdir}/eda_summary_report.txt", "w") as f:
        f.writelines(lines)
    print("".join(lines))


if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(FEATURED_PATH)

    print("Generating visualizations...")
    eligibility_distribution(df, OUTPUT_DIR)
    corr = correlation_heatmap(df, OUTPUT_DIR)
    credit_score_vs_eligibility(df, OUTPUT_DIR)
    max_emi_by_scenario(df, OUTPUT_DIR)
    demographic_patterns(df, OUTPUT_DIR)

    print("Generating summary report...")
    generate_report(df, corr, OUTPUT_DIR)

    print(f"\nAll EDA outputs saved to {OUTPUT_DIR}/")
