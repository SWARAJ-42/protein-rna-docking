import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
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
    """Load dataset from a JSON file and return as DataFrame."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def preprocess_data(df):
    """Separate features (X) and target (y)."""
    X = df.drop(columns=['Binding_affinity'])
    y = df['Binding_affinity']
    return X, y

def train_model(X_train, y_train, n_estimators, max_depth, random_state):
    """Train ExtraTreesClassifier with given hyperparameters."""
    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and return metrics."""
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
    """Save model parameters and evaluation results to a text file."""
    with open(output_path, "w") as f:
        f.write("=== ExtraTreesClassifier Training Summary ===\n\n")
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

    # Hyperparameter grid
    N_ESTIMATORS = [100, 200, 300]
    MAX_DEPTHS = [10, 20, 30]

    # Load dataset
    df = load_dataset(DATA_PATH)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Results collector
    results_summary = []

    for n_est in N_ESTIMATORS:
        for depth in MAX_DEPTHS:
            print(f"\nTraining with n_estimators={n_est}, max_depth={depth} ...")

            model = train_model(X_train, y_train, n_est, depth, RANDOM_STATE)
            metrics, report = evaluate_model(model, X_test, y_test)

            params = {
                "n_estimators": n_est,
                "max_depth": depth,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE
            }

            output_file = f"./outputs/output_n{n_est}_d{depth}.txt"
            save_results(params, metrics, report, output_file)

            results_summary.append({
                "n_estimators": n_est,
                "max_depth": depth,
                **metrics
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv("./outputs/grid_results_summary.csv", index=False)
    print("\nAll results saved to grid_results_summary.csv")

    # Identify best combination (based on F1 Score primarily, tie-breaker ROC_AUC)
    best_model = results_df.sort_values(
        by=["F1_Score", "ROC_AUC"], ascending=False
    ).iloc[0]

    print("\nBest Model Configuration:")
    print(best_model)

    with open("./outputs/best_model_summary.txt", "w") as f:
        f.write("=== BEST EXTRA TREES MODEL SUMMARY ===\n\n")
        for k, v in best_model.items():
            f.write(f"{k}: {v}\n")

    print("\nBest model summary saved to best_model_summary.txt")

# -------------------------
# Entry Point
# -------------------------

if __name__ == "__main__":
    main()
