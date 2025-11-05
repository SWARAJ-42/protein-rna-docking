import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np

# === Step 1: Load dataset ===
df = pd.read_json("./datasets/PRDBv3_preprocessed.json")

# === Step 2: Keep biologically meaningful features ===
keep_features = [
    'Structural_class', 'Flexible_class', 'Docking_case', 
    'has_tRNA', 'has_dsRNA', 'organism_match', 'is_NMR_structure',
    'RNA_chain_count', 'pro_seq_missing', 'RNA_seq_missing',
    'C_pro_seq_length', 'C_RNA_seq_length', 'U_pro_seq_length', 'U_RNA_seq_length',
    'avg_resolution', 'Binding_affinity'
]
df = df[keep_features].copy()

# === Step 3: Encode Binding_affinity ===
df['Binding_affinity'] = df['Binding_affinity'].map({'yes': 1, 'no': 0})

# === Step 4: Encode categorical variables ===
categorical_cols = ['Structural_class', 'Flexible_class', 'Docking_case']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])

# === Step 5: Handle missing values ===
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)

# === Step 6: Compute correlations ===
corr_matrix = df.corr(numeric_only=True)

# === Step 7: Get top 15 absolute correlations with Binding_affinity ===
corr_target = corr_matrix['Binding_affinity'].drop('Binding_affinity')
top_features = corr_target.abs().sort_values(ascending=False).head(20).index.tolist()

# === Step 8: Derived dataset ===
derived_df = df[top_features + ['Binding_affinity']]

# === Step 9: Plot and save heatmap ===
plt.figure(figsize=(12, 10))
sns.heatmap(
    derived_df.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={'label': 'Correlation'}
)
plt.title("Top 15 Correlated Features with Binding Affinity", fontsize=16)
plt.tight_layout()
plt.savefig("top15_correlation_heatmap.png")
plt.close()

# === Step 10: Save derived dataset as JSON ===
derived_df.to_json("derived_top15_features.json", orient="records", indent=4)

print("✅ Saved:")
print("- Derived dataset → derived_top15_features.json")
print("- Heatmap → top15_correlation_heatmap.png")

# === Optional: Show top correlation values ===
print("\nTop 15 correlations with Binding_affinity:")
print(corr_target.loc[top_features].sort_values(key=abs, ascending=False))
