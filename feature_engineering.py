import pandas as pd
import numpy as np
import json
import re
import ast
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# === CONFIG ===
input_path = "./datasets/PRDBv3_preprocessed.json"
output_path = "./PRDBv3_engineered.json"
high_cardinality_threshold = 15

# === HELPER FUNCTIONS ===
def safe_eval_arithmetic(value):
    if pd.isna(value) or value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        expr = value.strip()
        if expr.lower() == 'none':
            return np.nan
        if re.fullmatch(r"[0-9+\-*/. ]+", expr):
            try:
                return float(ast.literal_eval(expr))
            except:
                return np.nan
        try:
            return float(expr)
        except:
            return np.nan
    return np.nan

def bin_numeric(series, bins=5, labels=None):
    if series.isnull().all():
        return series
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)

def cluster_categorical(series, n_clusters=5):
    """
    Cluster high-cardinality categorical feature into n_clusters.
    Returns cluster labels as integers.
    """
    series = series.fillna("None").astype(str)
    le = LabelEncoder()
    encoded = le.fit_transform(series).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(encoded)
    return cluster_labels

# === LOAD DATA ===
with open(input_path, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
n_rows = len(df)
print(f"Loaded {n_rows} rows.")

# === STEP 0: Ensure missing columns exist ===
required_cols = [
    "C_RNA_source_organism", "C_pro_source_organism",
    "C_RNA_name", "C_RNA_chain",
    "U_RNA_name", "U_RNA_chain",
    "C_resolution","U_pro_resolution","U_RNA_resolution"
]
for col in required_cols:
    if col not in df.columns:
        df[col] = None

# === STEP 1: Convert numeric strings to float ===
numeric_fields = [
    "rcsbpdb_C_RNA_length", "C_RNA_seq_full_length", "C_RNA_seq_length",
    "rcsbpdb_U_pro_seq_length", "U_pro_seq_full_length", "U_pro_seq_length",
    "rcsbpdb__U_RNA_seq_length", "U_RNA_seq_full_length", "U_RNA_seq_length",
    "C_pro_seq_full_length", "C_pro_seq_length",
    "C_resolution", "U_pro_resolution", "U_RNA_resolution"
]
for col in numeric_fields:
    if col in df.columns:
        df[col] = df[col].apply(safe_eval_arithmetic)

# === STEP 2: Derived features (before one-hot / clustering) ===
df["has_tRNA"] = df["C_RNA_name"].str.contains("tRNA", case=False, na=False).astype(int)
df["has_dsRNA"] = df["C_RNA_name"].str.contains("dsRNA", case=False, na=False).astype(int)
df["has_RNA"] = df["C_RNA_name"].notna().astype(int)

df["organism_match"] = df.apply(
    lambda row: int(row.get("C_pro_source_organism") == row.get("C_RNA_source_organism")),
    axis=1
)

df["RNA_chain_count"] = df["C_RNA_chain"].apply(lambda x: len(str(x).strip()) if pd.notna(x) else 0)
df["pro_seq_missing"] = df["C_pro_seq_full_length"] - df["C_pro_seq_length"]
df["RNA_seq_missing"] = df["C_RNA_seq_full_length"] - df["C_RNA_seq_length"]

def avg_res(row):
    vals = []
    for col in ["C_resolution","U_pro_resolution","U_RNA_resolution"]:
        try:
            v = float(row[col])
            vals.append(v)
        except:
            continue
    return np.mean(vals) if vals else np.nan
df["avg_resolution"] = df.apply(avg_res, axis=1)

# === STEP 3: Bin numeric fields ===
resolution_cols = ["C_resolution","U_pro_resolution","U_RNA_resolution"]
for col in resolution_cols:
    if col in df.columns:
        df[f"{col}_bin"] = bin_numeric(df[col], bins=5, labels=[1,2,3,4,5])

length_cols = [
    "C_pro_seq_length","C_pro_seq_full_length",
    "C_RNA_seq_length","C_RNA_seq_full_length",
    "U_pro_seq_length","U_pro_seq_full_length",
    "U_RNA_seq_length","U_RNA_seq_full_length"
]
for col in length_cols:
    if col in df.columns:
        df[f"{col}_bin"] = bin_numeric(df[col], bins=5, labels=[1,2,3,4,5])

# === STEP 4: Handle categorical features ===
# Identify high-cardinality vs low-cardinality
categorical_cols = [
    "C_chain_PR","C_RNA_source_organism","U_RNA_macromolecule_name","U_RNA_structure_title",
    "U_RNA_PDB","C_RNA_chain","C_pro_chain","U_RNA_resolution","supp_U_pro_chain",
    "U_PRO_chain","U_RNA_chain","U_RNA_source_organism","Flexible_class",
    "Structural_class","Docking_case","Binding_affinity"
]

low_cardinality_cols = [c for c in categorical_cols if df[c].nunique() <= high_cardinality_threshold]
high_cardinality_cols = [c for c in categorical_cols if df[c].nunique() > high_cardinality_threshold]

# 4a. Cluster high-cardinality categorical features
for col in high_cardinality_cols:
    df[f"{col}_cluster"] = cluster_categorical(df[col], n_clusters=5)

# 4b. One-hot encode low-cardinality categorical features
df[low_cardinality_cols] = df[low_cardinality_cols].astype(str).fillna("None")
df = pd.get_dummies(df, columns=low_cardinality_cols, prefix=low_cardinality_cols)

# === STEP 6: Remove unnecessary string columns safely ===
# Remove original categorical columns that were clustered or one-hot encoded
string_cols_to_remove = [
    "C_pro_name",
    "C_structure_title",
    "C_RNA_name",
    "U_pro_structure_title",
    "U_pro_macromolecule_name",
    "U_RNA_name",
    "C_pro_source_organism",
    "U_pro_PDB",
    "U_pro_source_organism"
]
columns_to_remove = [col for col in categorical_cols if col in df.columns] + string_cols_to_remove  # all original categorical columns
df.drop(columns=columns_to_remove, inplace=True)

# === STEP 5: Save processed dataset ===
df.to_json(output_path, orient="records", indent=4)
print(f"Processed dataset saved to {output_path}")

