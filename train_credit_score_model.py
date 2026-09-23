import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# LOAD DATASET
# ============================================================

DATASET_PATH = r"C:\Users\Athira KJ\OneDrive\Desktop\personal finance\budget_dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("=" * 70)
print("DATASET")
print("=" * 70)

print("Shape:", df.shape)
print()


# ============================================================
# FEATURES
# ============================================================

features = [
    "monthly_income_usd",
    "monthly_expenses_usd",
    "savings_usd",
    "has_loan",
    "loan_amount_usd",
    "monthly_emi_usd",
    "debt_to_income_ratio",
    "savings_to_income_ratio",
    "expense_ratio"
]

target = "credit_score"


# ============================================================
# CHECK DATA
# ============================================================

print("Credit score statistics:")
print(df[target].describe())

print()


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

df = df.dropna(
    subset=features + [target]
).copy()


# ============================================================
# ENCODE LOAN
# ============================================================

loan_encoder = LabelEncoder()

df["has_loan_encoded"] = loan_encoder.fit_transform(
    df["has_loan"].astype(str)
)


# ============================================================
# PREPARE X
# ============================================================

X = df[
    [
        "monthly_income_usd",
        "monthly_expenses_usd",
        "savings_usd",
        "has_loan_encoded",
        "loan_amount_usd",
        "monthly_emi_usd",
        "debt_to_income_ratio",
        "savings_to_income_ratio",
        "expense_ratio"
    ]
].copy()


# ============================================================
# PREPARE Y
# ============================================================

y = df[target].astype(float)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("=" * 70)
print("TRAIN / TEST")
print("=" * 70)

print("Training:", len(X_train))
print("Testing :", len(X_test))


# ============================================================
# SCALE
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# MODELS
# ============================================================

models = {

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=3,
        random_state=42
    )
}


# ============================================================
# TRAIN AND COMPARE
# ============================================================

results = {}

for name, model in models.items():

    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)

    model.fit(
        X_train_scaled,
        y_train
    )

    predictions = model.predict(
        X_test_scaled
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results[name] = {
        "model": model,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }

    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 4))


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_name = min(
    results,
    key=lambda name: results[name]["mae"]
)

best_model = results[best_name]["model"]

print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Model:", best_name)
print("MAE :", round(results[best_name]["mae"], 2))
print("RMSE:", round(results[best_name]["rmse"], 2))
print("R2  :", round(results[best_name]["r2"], 4))


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    "credit_score_model.pkl"
)

joblib.dump(
    scaler,
    "credit_score_scaler.pkl"
)

joblib.dump(
    loan_encoder,
    "credit_score_loan_encoder.pkl"
)


print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print("credit_score_model.pkl")
print("credit_score_scaler.pkl")
print("credit_score_loan_encoder.pkl")


# ============================================================
# TEST SAMPLE PREDICTIONS
# ============================================================

sample_predictions = best_model.predict(
    X_test_scaled[:20]
)

print("\n")
print("=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

for actual, predicted in zip(
    y_test.iloc[:20],
    sample_predictions
):

    print(
        "Actual:",
        round(float(actual), 2),
        " | Predicted:",
        round(float(predicted), 2)
    )