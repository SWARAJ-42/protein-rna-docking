import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# -------------------------
# Utility Functions
# -------------------------

def load_dataset(dataset_path):
    """Load dataset from JSON file and return as a DataFrame."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def preprocess_data(df):
    """Separate features (X) and target (y)."""
    X = df.drop(columns=['Binding_affinity'])
    y = df['Binding_affinity']
    return X, y

def train_model(X_train, y_train, criterion, max_depth, min_samples_split, random_state):
    """Train DecisionTreeClassifier with given hyperparameters."""
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1_Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan,
    }
    report = classification_report(y_test, y_pred, digits=4)
    return metrics, report

def save_results(params, metrics, report, output_path):
    """Save results and metrics to a text file."""
    with open(output_path, "w") as f:
        f.write("=== DecisionTreeClassifier Training Summary ===\n\n")
        f.write("Model Settings:\n")
        for k, v in params.items():
            f.write(f"{k}: {v}\n")

        f.write("\n=== Evaluation Metrics ===\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

        f.write("\n--- Detailed Classification Report ---\n")
        f.write(report)

    print(f"Saved results to {output_path}")

# -------------------------
# Main Grid Search Logic
# -------------------------

def main():
    DATA_PATH = "../PRDBv3_engineered_v1.json"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Define hyperparameter grid
    CRITERIA = ["gini", "entropy", "log_loss"]
    MAX_DEPTHS = [6, 12, 18, 24, 30]
    MIN_SAMPLES_SPLITS = [2, 3, 4, 5]

    # Load and prepare dataset
    df = load_dataset(DATA_PATH)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results_summary = []

    # Run through all combinations
    for crit in CRITERIA:
        for depth in MAX_DEPTHS:
            for split in MIN_SAMPLES_SPLITS:
                print(f"\nTraining with criterion={crit}, max_depth={depth}, min_samples_split={split}...")

                model = train_model(X_train, y_train, crit, depth, split, RANDOM_STATE)
                metrics, report = evaluate_model(model, X_test, y_test)

                params = {
                    "criterion": crit,
                    "max_depth": depth,
                    "min_samples_split": split,
                    "test_size": TEST_SIZE,
                    "random_state": RANDOM_STATE
                }

                filename = f"./outputs/output_c{crit}_d{depth}_s{split}.txt"
                save_results(params, metrics, report, filename)

                results_summary.append({
                    "criterion": crit,
                    "max_depth": depth,
                    "min_samples_split": split,
                    **metrics
                })

    # Summarize all results
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv("./decisiontree_grid_results.csv", index=False)
    print("\nAll results saved to decisiontree_grid_results.csv")

    # Identify best performing configuration
    best_model = results_df.sort_values(
        by=["F1_Score", "ROC_AUC"], ascending=False
    ).iloc[0]

    print("\nBest Model Configuration:")
    print(best_model)

    with open("./best_decisiontree_model.txt", "w") as f:
        f.write("=== BEST DECISION TREE MODEL SUMMARY ===\n\n")
        for k, v in best_model.items():
            f.write(f"{k}: {v}\n")

    print("\nBest model summary saved to best_decisiontree_model.txt")

# -------------------------
# Entry Point
# -------------------------

if __name__ == "__main__":
    main()
