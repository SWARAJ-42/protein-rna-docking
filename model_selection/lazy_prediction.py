import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lazypredict.Supervised import LazyClassifier
from imblearn.over_sampling import SMOTE
import joblib

# ---------------- CONFIG ----------------
DATA_PATH = "./PRDBv3_engineered_v1.json"
TARGET_COL = "Binding_affinity"
CATEGORICAL_LIKE = [
    "Structural_class", "Docking_case", "Flexible_class",
    "has_tRNA", "is_NMR_structure", "has_dsRNA",
    "organism_match", "RNA_chain_count"
]
DROP_COLS = []
USE_SMOTE = False
RANDOM_STATE = 42
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ---------------------------------------

# Load data
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"File not found: {DATA_PATH}")

df = pd.read_json(DATA_PATH, orient="records")
print(f"Loaded dataset: {df.shape}")

df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

if TARGET_COL not in df.columns:
    raise KeyError(f"Target column '{TARGET_COL}' not found. Columns available: {list(df.columns)}")

# Convert declared categorical-like columns
for c in CATEGORICAL_LIKE:
    if c in df.columns:
        df[c] = df[c].astype("category")

print("\nData types summary:")
print(df.dtypes)

print("\nTarget distribution:")
print(df[TARGET_COL].value_counts(), "\nProportions:\n", df[TARGET_COL].value_counts(normalize=True))

# Prepare X, y
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Detect numeric and categorical columns
num_cols = X.select_dtypes(include=["number", "float64", "int64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["category", "object", "bool"]).columns.tolist()

print(f"\nNumeric features: {len(num_cols)}, Categorical features: {len(cat_cols)}")
print("Sample numeric columns:", num_cols[:10])
print("Sample categorical columns:", cat_cols[:10])

# Preprocessors
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ],
    remainder="drop"
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Preprocess
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# SMOTE (optional)
if USE_SMOTE:
    print("\nApplying SMOTE to training set...")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_prep, y_train = sm.fit_resample(X_train_prep, y_train)
    print("Post-SMOTE class counts:\n", pd.Series(y_train).value_counts())

# LazyPredict
print("\nRunning LazyClassifier (this may take a while)...")
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train_prep, X_test_prep, y_train, y_test)

print("\nTop models:")
print(models.head(20))

# Save outputs
models_path = os.path.join(OUTPUT_DIR, "lazypredict_models_PRDBv3_v1.csv")
models.to_csv(models_path, index=True)
print(f"\nSaved LazyPredict results to: {models_path}")

preproc_path = os.path.join(OUTPUT_DIR, "prdbv3_preprocessor_v1.joblib")
joblib.dump(preprocessor, preproc_path)
print(f"Saved preprocessor to: {preproc_path}")

# Save train/test data (optional)
X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train_raw.csv"), index=False)
X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test_raw.csv"), index=False)
y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)
print(f"Saved train/test splits to: {OUTPUT_DIR}")
