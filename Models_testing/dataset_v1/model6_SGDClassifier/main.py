import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
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
    """Scale features for SGD-based linear models."""
    X = df.drop(columns=["Binding_affinity"])
    y = df["Binding_affinity"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def train_model(X_train, y_train, loss, penalty, alpha, max_iter):
    """Train SGDClassifier with given hyperparameters."""
    model = SGDClassifier(
        loss=loss,
        penalty=penalty,
        alpha=alpha,
        max_iter=max_iter,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Compute evaluation metrics."""
    y_pred = model.predict(X_test)

    # Attempt to compute decision scores for ROC_AUC
    try:
        y_score = model.decision_function(X_test)
        roc = roc_auc_score(y_test, y_score)
    except Exception:
        roc = np.nan

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1_Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc
    }

    report = classification_report(y_test, y_pred, digits=4)
    return metrics, report

def save_results(params, metrics, report, output_path):
    """Save per-run results to text file."""
    with open(output_path, "w") as f:
        f.write("=== SGDClassifier Training Summary ===\n\n")
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
    LOSSES = ["hinge", "log_loss", "modified_huber"]
    PENALTIES = ["l2", "l1", "elasticnet"]
    ALPHAS = [0.0001, 0.001, 0.01]
    MAX_ITERS = [1000, 2000]

    os.makedirs("outputs", exist_ok=True)

    df = load_dataset(DATA_PATH)
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results = []

    for loss in LOSSES:
        for penalty in PENALTIES:
            for alpha in ALPHAS:
                for iters in MAX_ITERS:
                    print(f"Training SGDClassifier: loss={loss}, penalty={penalty}, alpha={alpha}, max_iter={iters}")

                    model = train_model(X_train, y_train, loss, penalty, alpha, iters)
                    metrics, report = evaluate_model(model, X_test, y_test)

                    params = {
                        "loss": loss,
                        "penalty": penalty,
                        "alpha": alpha,
                        "max_iter": iters,
                        "test_size": TEST_SIZE,
                        "random_state": RANDOM_STATE
                    }

                    filename = f"outputs/output_loss{loss}_pen{penalty}_alpha{alpha}_iter{iters}.txt"
                    save_results(params, metrics, report, filename)

                    results.append({
                        "loss": loss,
                        "penalty": penalty,
                        "alpha": alpha,
                        "max_iter": iters,
                        **metrics
                    })

    # Save summary results
    results_df = pd.DataFrame(results)
    results_df.to_csv("sgdclassifier_results.csv", index=False)
    print("All results saved to sgdclassifier_results.csv")

    # Select best configuration
    best = results_df.sort_values(by=["F1_Score", "ROC_AUC"], ascending=False).iloc[0]
    print("\nBest configuration:")
    print(best)

    with open("best_sgdclassifier_model.txt", "w") as f:
        f.write("=== BEST SGD CLASSIFIER MODEL ===\n\n")
        for k, v in best.items():
            f.write(f"{k}: {v}\n")

    print("Best model summary saved to best_sgdclassifier_model.txt")

if __name__ == "__main__":
    main()
