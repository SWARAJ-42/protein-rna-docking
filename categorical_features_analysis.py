import json
import pandas as pd
import numpy as np

# === CONFIG ===
input_path = "./datasets/PRDBv3_preprocessed.json"   # <-- Replace with your actual JSON file path
output_path = "categorical_feature_analysis.csv"

# === LOAD JSON ===
with open(input_path, "r") as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)
n_rows = len(df)
print(f"Loaded {n_rows} records.")

# === Identify candidate categorical fields ===
# We'll exclude clearly numeric fields
def is_numeric_series(s):
    try:
        pd.to_numeric(s.dropna().astype(str).str.replace('+','', regex=False))
        return True
    except Exception:
        return False

categorical_candidates = []
for col in df.columns:
    if df[col].dtype == 'object' and not is_numeric_series(df[col]):
        categorical_candidates.append(col)

print(f"Found {len(categorical_candidates)} potential categorical fields.")

# === Analyze each categorical feature ===
records = []
for col in categorical_candidates:
    series = df[col].astype(str)
    n_unique = series.nunique(dropna=True)
    most_common_value = series.value_counts().index[0]
    most_common_count = series.value_counts().iloc[0]
    percent_unique = n_unique / n_rows * 100

    records.append({
        "feature": col,
        "n_unique": n_unique,
        "most_common_value": most_common_value[:80],  # truncate for readability
        "most_common_count": most_common_count,
        "percent_unique": round(percent_unique, 2)
    })

# === Save and show summary ===
summary_df = pd.DataFrame(records).sort_values("percent_unique", ascending=False)
summary_df.to_csv(output_path, index=False)
print("\n=== Summary of Categorical Feature Cardinality ===")
print(summary_df.to_string(index=False))
print(f"\nSaved to {output_path}")
