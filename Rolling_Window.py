"""
Rolling 6-month training / 2-month testing temporal validation
for SOFA, APACHE II, and Combined models.

- Trains logistic regression on 6 months of data
- Tests on the next 2 months
- Advances the window by 1 month each time
- Computes AUC and Brier score for each window
"""

import pandas as pd
from datetime import timedelta
from dateutil.relativedelta import relativedelta

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss


# -----------------------------
# 1. Load and prepare the data
# -----------------------------
# Adjust the path and sheet name as needed
FILE_PATH = "Final_Modelling_Data_First_Admission_End_Of_May.xlsx"

df = pd.read_excel(FILE_PATH)

# Ensure date is datetime & sort chronologically
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
df = df.sort_values("Date of Admission").reset_index(drop=True)

# Encode outcome: 1 = Deceased, 0 = Alive (adjust if your labels differ)
df["y"] = (df["Outcome"].str.lower() == "deceased").astype(int)

# Optionally drop rows with missing predictors
df = df.dropna(subset=["SOFA", "Apache II", "y", "Date of Admission"])


# --------------------------------------------------
# 2. Define rolling-window generator (6m train, 2m test)
# --------------------------------------------------
def rolling_windows(
    df,
    date_col="Date of Admission",
    months_train=6,
    months_test=2,
    min_train=80,
    min_test=40,
):
    """
    Generate (train_start, train_end, test_start, test_end) tuples
    for rolling 6-month train and 2-month test windows.
    Windows advance by 1 month at a time.

    train: [train_start, train_end)
    test:  [test_start, test_end)
    """
    res = []
    start_date = df[date_col].min().normalize()
    last_date = df[date_col].max().normalize()

    cutoff = start_date
    while True:
        train_start = cutoff
        train_end = cutoff + relativedelta(months=months_train)

        test_start = train_end
        test_end = train_end + relativedelta(months=months_test)

        # Stop if test window extends beyond the data
        if test_start > last_date or test_end > (last_date + timedelta(days=1)):
            break

        tr_mask = (df[date_col] >= train_start) & (df[date_col] < train_end)
        te_mask = (df[date_col] >= test_start) & (df[date_col] < test_end)

        n_tr = tr_mask.sum()
        n_te = te_mask.sum()

        if n_tr >= min_train and n_te >= min_test:
            res.append((train_start, train_end, test_start, test_end))

        # Advance by 1 month
        cutoff = cutoff + relativedelta(months=1)

    return res


windows = rolling_windows(df)
print(f"Number of rolling windows: {len(windows)}")


# --------------------------------------------------
# 3. Function to fit and evaluate a model in a window
# --------------------------------------------------
def evaluate_window(df, feature_cols, y_col, window, date_col="Date of Admission"):
    """
    Fit logistic regression on the training window and
    compute AUC + Brier for train and test sets.
    """
    train_start, train_end, test_start, test_end = window

    tr_mask = (df[date_col] >= train_start) & (df[date_col] < train_end)
    te_mask = (df[date_col] >= test_start) & (df[date_col] < test_end)

    X_train = df.loc[tr_mask, feature_cols].values
    y_train = df.loc[tr_mask, y_col].values

    X_test = df.loc[te_mask, feature_cols].values
    y_test = df.loc[te_mask, y_col].values

    # Basic logistic regression (no penalty tuning)
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)

    p_train = model.predict_proba(X_train)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    result = {
        "Train_Start": train_start.date(),
        "Train_End": (train_end - timedelta(days=1)).date(),
        "Test_Start": test_start.date(),
        "Test_End": (test_end - timedelta(days=1)).date(),
        "Train_n": len(y_train),
        "Test_n": len(y_test),
        "Train_Mortality_%": round(100 * y_train.mean(), 1),
        "Test_Mortality_%": round(100 * y_test.mean(), 1),
        "AUC_Train": roc_auc_score(y_train, p_train),
        "AUC_Test": roc_auc_score(y_test, p_test),
        "Brier_Train": brier_score_loss(y_train, p_train),
        "Brier_Test": brier_score_loss(y_test, p_test),
    }
    return result


# --------------------------------------------------
# 4. Run models for SOFA, APACHE II, and Combined
# --------------------------------------------------
all_rows = []

model_specs = [
    ("SOFA", ["SOFA"]),
    ("APACHE II", ["Apache II"]),
    ("Combined", ["SOFA", "Apache II"]),
]

for model_name, cols in model_specs:
    print(f"Evaluating model: {model_name}")
    for w in windows:
        res = evaluate_window(df, cols, "y", w)
        res["Model"] = model_name
        all_rows.append(res)

results = pd.DataFrame(all_rows)

# Sort nicely
results = results[
    [
        "Model",
        "Train_Start",
        "Train_End",
        "Test_Start",
        "Test_End",
        "Train_n",
        "Test_n",
        "Train_Mortality_%",
        "Test_Mortality_%",
        "AUC_Train",
        "AUC_Test",
        "Brier_Train",
        "Brier_Test",
    ]
].sort_values(["Model", "Train_Start"])

print(results)


# --------------------------------------------------
# 5. Summary table (mean / range across windows)
# --------------------------------------------------
summary = (
    results.groupby("Model")[["AUC_Test", "Brier_Test", "Test_Mortality_%"]]
    .agg(["mean", "std", "min", "max"])
    .round(3)
)

print("\nSummary across all rolling windows:\n")
print(summary)


# --------------------------------------------------
# 6. Optional: save to Excel/CSV
# --------------------------------------------------
results.to_excel("Rolling_Temporal_Results_All_Models.xlsx", index=False)
summary.to_excel("Rolling_Temporal_Summary_By_Model.xlsx")
