import pandas as pd

DATASET_PATH = r"C:\Users\Athira KJ\OneDrive\Desktop\personal finance\budget_dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("=" * 70)
print("CREDIT SCORE ANALYSIS")
print("=" * 70)

print("\nCredit score by financial health:")
print(
    df.groupby("financial_health_status")["credit_score"]
    .agg(["count", "mean", "min", "max"])
)

print("\nCorrelation with numeric features:")
print(
    df[
        [
            "monthly_income_usd",
            "monthly_expenses_usd",
            "savings_usd",
            "loan_amount_usd",
            "monthly_emi_usd",
            "debt_to_income_ratio",
            "savings_to_income_ratio",
            "expense_ratio",
            "credit_score"
        ]
    ].corr()["credit_score"]
    .sort_values(ascending=False)
)

print("\nCredit score sample:")
print(
    df[
        [
            "monthly_income_usd",
            "monthly_expenses_usd",
            "savings_usd",
            "has_loan",
            "loan_amount_usd",
            "monthly_emi_usd",
            "credit_score",
            "financial_health_status"
        ]
    ].head(20)
)