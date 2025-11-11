import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

def load_dataset(dataset_path):
    """Load dataset from JSON file."""
    with open(dataset_path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def preprocess_data(df):
    """Scale features for GaussianNB."""
    X = df.drop(columns=["Binding_affinity"])
    y = df["Binding_affinity"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def train_model(X_train, y_train, var_smoothing, priors=None):
    """Train Gaussian Naive Bayes model."""
    model = GaussianNB(var_smoothing=var_smoothing, priors=priors)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model and compute standard metrics."""
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
    """Save model configuration, metrics, and classification report."""
    with open(output_path, "w") as f:
        f.write("=== GaussianNB Training Summary ===\n\n")
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
    DATA_PATH = "../PRDBv3_engineered_v2.json"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Hyperparameter grid
    VAR_SMOOTHINGS = [1e-9, 1e-8, 1e-7, 1e-6]
    PRIORS = [None, [0.3, 0.7], [0.5, 0.5]]

    os.makedirs("outputs", exist_ok=True)

    df = load_dataset(DATA_PATH)
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results = []

    for var_smoothing in VAR_SMOOTHINGS:
        for priors in PRIORS:
            print(f"Training GaussianNB: var_smoothing={var_smoothing}, priors={priors}")

            model = train_model(X_train, y_train, var_smoothing, priors)
            metrics, report = evaluate_model(model, X_test, y_test)

            params = {
                "var_smoothing": var_smoothing,
                "priors": priors,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE
            }

            filename = f"outputs/output_varsmooth{var_smoothing}_priors{priors}.txt".replace(" ", "")
            save_results(params, metrics, report, filename)

            results.append({
                "var_smoothing": var_smoothing,
                "priors": str(priors),
                **metrics
            })

    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv("gaussiannb_results.csv", index=False)
    print("All results saved to gaussiannb_results.csv")

    # Identify best configuration
    best = results_df.sort_values(by=["F1_Score", "ROC_AUC"], ascending=False).iloc[0]
    print("\nBest configuration:")
    print(best)

    with open("best_gaussiannb_model.txt", "w") as f:
        f.write("=== BEST GAUSSIAN NAIVE BAYES MODEL ===\n\n")
        for k, v in best.items():
            f.write(f"{k}: {v}\n")

    print("Best model summary saved to best_gaussiannb_model.txt")

if __name__ == "__main__":
    main()
