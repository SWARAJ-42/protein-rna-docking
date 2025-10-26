import pandas as pd
import re
import ast
import argparse
import json
import os


def safe_eval_arithmetic(value):
    """
    Safely evaluate arithmetic-like strings (e.g., '13+13' -> 26).
    Returns float or None.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        expr = value.strip()
        # Only allow digits and arithmetic operators
        if re.fullmatch(r"[0-9+\-*/. ]+", expr):
            try:
                return float(ast.literal_eval(expr))
            except Exception:
                return None
    return None

def strip_string_fields(df):
    """
    Strip leading and trailing spaces from all string fields in the DataFrame.
    This helps clean inconsistent inputs like '  Homo sapiens ' -> 'Homo sapiens'.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def preprocess_structure_data(df):
    """
    Preprocess the structural dataset:
    - Fix arithmetic-like numeric strings
    - Add derived biological features
    """
    df = df.copy()

    df = strip_string_fields(df)

    # Fields that may contain arithmetic expressions
    numeric_fields = [
        "rcsbpdb_C_RNA_length", "C_RNA_seq_full_length", "C_RNA_seq_length",
        "rcsbpdb_U_pro_seq_length", "U_pro_seq_full_length", "U_pro_seq_length",
        "rcsbpdb__U_RNA_seq_length", "U_RNA_seq_full_length", "U_RNA_seq_length",
        "C_pro_seq_full_length", "C_pro_seq_length", "rcsbpdb_C_pro_seq_length"
    ]

    for col in numeric_fields:
        if col in df.columns:
            df[col] = df[col].apply(safe_eval_arithmetic)

    # Derived binary/categorical features
    df["has_tRNA"] = df["C_RNA_name"].fillna("").str.contains("tRNA", case=False).astype(int)
    df["has_dsRNA"] = df["C_RNA_name"].fillna("").str.contains("dsRNA", case=False).astype(int)
    df["has_RNA"] = df["C_RNA_name"].notna().astype(int)
    df["has_RNA_source"] = df["C_RNA_source_organism"].notna().astype(int)
    df["organism_match"] = (df["C_pro_source_organism"] == df["C_RNA_source_organism"]).astype(int)
    df["is_NMR_structure"] = df["U_pro_resolution"].astype(str).str.upper().eq("NMR").astype(int)

    # RNA chain count (number of chain letters, e.g., "DE" -> 2)
    df["RNA_chain_count"] = df["C_RNA_chain"].fillna("").apply(lambda x: len(x.strip()))

    # Sequence completeness
    df["pro_seq_missing"] = df["C_pro_seq_full_length"] - df["C_pro_seq_length"]
    df["RNA_seq_missing"] = df["C_RNA_seq_full_length"] - df["C_RNA_seq_length"]

    # Average resolution across available numeric ones
    def avg_res(row):
        vals = []
        for col in ["C_resolution", "U_pro_resolution", "U_RNA_resolution"]:
            try:
                v = float(row[col])
                vals.append(v)
            except Exception:
                continue
        return sum(vals)/len(vals) if vals else None

    df["avg_resolution"] = df.apply(avg_res, axis=1)

    return df


def main(input_path, output_path):
    # Load JSON data
    print(f"Loading data from {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} records.")

    # Preprocess
    print("Preprocessing data ...")
    clean_df = preprocess_structure_data(df)

    # Save to output
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".csv":
        clean_df.to_csv(output_path, index=False)
    else:
        clean_df.to_json(output_path, orient="records", indent=4)

    print(f"Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess protein-RNA structural dataset JSON file.")
    parser.add_argument("--input", "-i", required=True, help="Path to input JSON file (array of objects).")
    parser.add_argument("--output", "-o", required=True, help="Path to save cleaned output (JSON or CSV).")
    args = parser.parse_args()

    main(args.input, args.output)
