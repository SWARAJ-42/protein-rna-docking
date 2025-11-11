import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from lightgbm import LGBMClassifier

def load_dataset(dataset_path):
    """Load dataset from JSON file."""
    with open(dataset_path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def preprocess_data(df):
    """Scale features for boosting models."""
    X = df.drop(columns=["Binding_affinity"])
    y = df["Binding_affinity"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def train_model(X_train, y_train, num_leaves, max_depth, learning_rate, n_estimators):
    """Train LGBMClassifier with given hyperparameters."""
    model = LGBMClassifier(
        objective="binary",
        num_leaves=num_leaves,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model and compute standard classification metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1_Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_proba)
    }

    report = classification_report(y_test, y_pred, digits=4)
    return metrics, report

def save_results(params, metrics, report, output_path):
    """Save individual model results to text file."""
    with open(output_path, "w") as f:
        f.write("=== LGBMClassifier Training Summary ===\n\n")
        f.write("Model Settings:\n")
        for k, v in params.items():
            f.write(f"{k}: {v}\n")

        f.write("\n=== Evaluation Metrics ===\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

        f.write("\n--- Detailed Classification Report ---\n")
        f.write(report)

    print(f"Saved: {output_path}")

def main():
    DATA_PATH = "../PRDBv3_engineered_v1.json"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Hyperparameter grid
    NUM_LEAVES = [15, 31, 63]
    MAX_DEPTHS = [5, 10, -1]  # -1 means no limit
    LEARNING_RATES = [0.01, 0.05, 0.1]
    N_ESTIMATORS = [100, 200]

    os.makedirs("outputs", exist_ok=True)

    df = load_dataset(DATA_PATH)
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results = []

    for num_leaves in NUM_LEAVES:
        for max_depth in MAX_DEPTHS:
            for lr in LEARNING_RATES:
                for n_est in N_ESTIMATORS:
                    print(f"Training: num_leaves={num_leaves}, max_depth={max_depth}, lr={lr}, n_estimators={n_est}")

                    model = train_model(X_train, y_train, num_leaves, max_depth, lr, n_est)
                    metrics, report = evaluate_model(model, X_test, y_test)

                    params = {
                        "num_leaves": num_leaves,
                        "max_depth": max_depth,
                        "learning_rate": lr,
                        "n_estimators": n_est,
                        "test_size": TEST_SIZE,
                        "random_state": RANDOM_STATE
                    }

                    filename = f"outputs/output_leaves{num_leaves}_depth{max_depth}_lr{lr}_nest{n_est}.txt"
                    save_results(params, metrics, report, filename)

                    results.append({
                        "num_leaves": num_leaves,
                        "max_depth": max_depth,
                        "learning_rate": lr,
                        "n_estimators": n_est,
                        **metrics
                    })

    # Save summary results
    results_df = pd.DataFrame(results)
    results_df.to_csv("lgbmclassifier_results.csv", index=False)
    print("All results saved to lgbmclassifier_results.csv")

    # Determine best configuration
    best = results_df.sort_values(by=["F1_Score", "ROC_AUC"], ascending=False).iloc[0]
    print("\nBest configuration:")
    print(best)

    with open("best_lgbmclassifier_model.txt", "w") as f:
        f.write("=== BEST LGBM CLASSIFIER MODEL ===\n\n")
        for k, v in best.items():
            f.write(f"{k}: {v}\n")

    print("Best model summary saved to best_lgbmclassifier_model.txt")

if __name__ == "__main__":
    main()
