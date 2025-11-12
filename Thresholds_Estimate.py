import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

# ----------------------------
# Load Data
# ----------------------------
#File with columns "Sex","Age","APACHE II score", "SOFA score"
file_path = "Final_Modelling_Data_First_Admission_End_Of_May.xlsx"
data = pd.read_excel(file_path)

# Development set (rows 0–758)
dev_data = data.iloc[0:759].copy()

# Columns
outcome_col = "Outcome"
sofa_col = "SOFA"
apache_col = "Apache II"

# Recode outcome to binary
y = dev_data[outcome_col].map({"Alive": 0, "Deceased": 1}).values
X_SOFA = dev_data[[sofa_col]].values
X_APACHE = dev_data[[apache_col]].values
X_COMBINED = dev_data[[sofa_col, apache_col]].values

# ----------------------------
# Helper Functions
# ----------------------------
log_reg = LogisticRegression(solver="liblinear")

def optimal_threshold(y_true, y_prob):
    """Return threshold that maximizes Youden J."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    J = tpr - fpr
    return thr[np.argmax(J)]

def summarize_thresholds(thresholds):
    return {
        "n_runs": len(thresholds),
        "mean": float(np.mean(thresholds)),
        "median": float(np.median(thresholds)),
        "q1": float(np.percentile(thresholds, 25)),
        "q3": float(np.percentile(thresholds, 75)),
        "min": float(np.min(thresholds)),
        "max": float(np.max(thresholds))
    }

# ----------------------------
# 1. Classic Bootstrap (forced balance)
# ----------------------------
n_bootstraps = 10000
rng = np.random.default_rng(seed=42)

thresholds_sofa_boot, thresholds_apache_boot, thresholds_combined_boot = [], [], []

#thr_boot = {"SOFA": [], "Apache II": [], "Combined": []}

while len(thresholds_sofa_boot) < n_bootstraps:
    idx = rng.choice(len(dev_data), size=len(dev_data), replace=True)
    if len(np.unique(y[idx])) < 2:
        continue
    
    # SOFA
    model = log_reg.fit(X_SOFA[idx], y[idx])
    probs = model.predict_proba(X_SOFA[idx])[:, 1]
    thresholds_sofa_boot.append(optimal_threshold(y[idx], probs))
    
    # Apache II
    model = log_reg.fit(X_APACHE[idx], y[idx])
    probs = model.predict_proba(X_APACHE[idx])[:, 1]
    thresholds_apache_boot.append(optimal_threshold(y[idx], probs))
    
    # Combined
    model = log_reg.fit(X_COMBINED[idx], y[idx])
    probs = model.predict_proba(X_COMBINED[idx])[:, 1]
    thresholds_combined_boot.append(optimal_threshold(y[idx], probs))

summary_bootstrap = {
    "SOFA": summarize_thresholds(thresholds_sofa_boot),
    "Apache II": summarize_thresholds(thresholds_apache_boot),
    "Combined": summarize_thresholds(thresholds_combined_boot)
}

# ----------------------------
# 2. 80/20 Splits (forced balance)
# ----------------------------
n_splits = 10000
thresholds_sofa_split, thresholds_apache_split, thresholds_combined_split = [], [], []

#thr_boot = {"SOFA": [], "Apache II": [], "Combined": []}
for i in range(n_splits):
    while True:
        X_train, X_val, y_train, y_val = train_test_split(
            X_COMBINED, y, test_size=0.2, random_state=None
        )
        if len(np.unique(y_train)) == 2 and len(np.unique(y_val)) == 2:
            break
    
    # SOFA
    model = log_reg.fit(X_train[:, [0]], y_train)
    probs = model.predict_proba(X_val[:, [0]])[:, 1]
    thresholds_sofa_split.append(optimal_threshold(y_val, probs))
    
    # Apache II
    model = log_reg.fit(X_train[:, [1]], y_train)
    probs = model.predict_proba(X_val[:, [1]])[:, 1]
    thresholds_apache_split.append(optimal_threshold(y_val, probs))
    
    # Combined
    model = log_reg.fit(X_train, y_train)
    probs = model.predict_proba(X_val)[:, 1]
    thresholds_combined_split.append(optimal_threshold(y_val, probs))

summary_splits = {
    "SOFA": summarize_thresholds(thresholds_sofa_split),
    "Apache II": summarize_thresholds(thresholds_apache_split),
    "Combined": summarize_thresholds(thresholds_combined_split)
}

def _median_or_nan(arr):
    arr = np.asarray(arr, dtype=float)
    return float(np.median(arr)) if arr.size else float("nan")

# Sanity checks (optional but helpful)
required_lists = {
    "thresholds_sofa_boot":        'Bootstrap/SOFA',
    "thresholds_apache_boot":      'Bootstrap/APACHE II',
    "thresholds_combined_boot":    'Bootstrap/Combined',
    "thresholds_sofa_split":       '80/20 Split/SOFA',
    "thresholds_apache_split":     '80/20 Split/APACHE II',
    "thresholds_combined_split":   '80/20 Split/Combined',
}
_missing = [k for k in required_lists if k not in globals()]
if _missing:
    raise NameError(f"These threshold lists were not found (make sure they are created above): {_missing}")

rows = [
    {"method": "Bootstrap",   "model": "SOFA",       "median_threshold": _median_or_nan(thresholds_sofa_boot)},
    {"method": "Bootstrap",   "model": "APACHE II",  "median_threshold": _median_or_nan(thresholds_apache_boot)},
    {"method": "Bootstrap",   "model": "Combined",   "median_threshold": _median_or_nan(thresholds_combined_boot)},
    {"method": "80/20 Split", "model": "SOFA",       "median_threshold": _median_or_nan(thresholds_sofa_split)},
    {"method": "80/20 Split", "model": "APACHE II",  "median_threshold": _median_or_nan(thresholds_apache_split)},
    {"method": "80/20 Split", "model": "Combined",   "median_threshold": _median_or_nan(thresholds_combined_split)},
]

out_df = pd.DataFrame(rows, columns=["method", "model", "median_threshold"])
out_df.to_csv("median_thresholds.csv", index=False)
print("\nMedian thresholds (saved to median_thresholds.csv):")
print(out_df.to_string(index=False))
# ====================================================================
