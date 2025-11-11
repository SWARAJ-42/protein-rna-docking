import pandas as pd
import json

# File paths
json_path = "./PRDBv3_engineered_v1.json"
csv_path = "./supplimentary_data.csv"
output_path = "./PRDBv3_engineered_v2.json"

# Load files
with open(json_path, "r") as f:
    data = json.load(f)

csv_data = pd.read_csv(csv_path)

# Get feature columns (excluding the first one, protein name)
feature_columns = csv_data.columns[1:]

# Iterate through JSON and CSV simultaneously
for i, row in csv_data.iterrows():
    for col in feature_columns:
        # Assign value or 0 if NaN
        value = row[col]
        data[i][col] = 0 if pd.isna(value) else value

# Save merged JSON
with open(output_path, "w") as f:
    json.dump(data, f, indent=4)

output_path
