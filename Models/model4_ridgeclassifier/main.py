import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
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
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def preprocess_data(df):
    """Separate features and target; scale features for RidgeClassifier."""
    X = df.drop(columns=["Binding_affinity"])
    y = df["Binding_affinity"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def train_model(X_train, y_train, alpha, solver, fit_intercept):
    """Train RidgeClassifier with given hyperparameters."""
    model = RidgeClassifier(alpha=alpha, solver=solver, fit_intercept=fit_intercept)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model and compute metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1_Score": f1_score(y_test, y_pred, zero_division=0)
    }

    # ROC_AUC needs probabilities; RidgeClassifier doesn't have predict_proba
    try:
        y_score = model.decision_function(X_test)
        metrics["ROC_AUC"] = roc_auc_score(y_test, y_score)
    except Exception:
        metrics["ROC_AUC"] = np.nan

    report = classification_report(y_test, y_pred, digits=4)
    return metrics, report

def save_results(params, metrics, report, output_path):
    """Save individual model results to text file."""
    with open(output_path, "w") as f:
        f.write("=== RidgeClassifier Training Summary ===\n\n")
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

    # Define hyperparameter grid
    ALPHAS = [0.1, 1.0, 10.0]
    SOLVERS = ["auto", "saga", "lsqr"]
    FIT_INTERCEPTS = [True, False]

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Load and preprocess dataset
    df = load_dataset(DATA_PATH)
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results = []

    for alpha in ALPHAS:
        for solver in SOLVERS:
            for fit_int in FIT_INTERCEPTS:
                print(f"Training RidgeClassifier: alpha={alpha}, solver={solver}, fit_intercept={fit_int}")

                model = train_model(X_train, y_train, alpha, solver, fit_int)
                metrics, report = evaluate_model(model, X_test, y_test)

                params = {
                    "alpha": alpha,
                    "solver": solver,
                    "fit_intercept": fit_int,
                    "test_size": TEST_SIZE,
                    "random_state": RANDOM_STATE
                }

                filename = f"outputs/output_alpha{alpha}_solver{solver}_intercept{fit_int}.txt"
                save_results(params, metrics, report, filename)

                results.append({
                    "alpha": alpha,
                    "solver": solver,
                    "fit_intercept": fit_int,
                    **metrics
                })

    # Save summary results
    results_df = pd.DataFrame(results)
    results_df.to_csv("ridgeclassifier_results.csv", index=False)
    print("All results saved to ridgeclassifier_results.csv")

    # Find the best configuration
    best = results_df.sort_values(by=["F1_Score", "ROC_AUC"], ascending=False).iloc[0]
    print("\nBest configuration:")
    print(best)

    with open("best_ridgeclassifier_model.txt", "w") as f:
        f.write("=== BEST RIDGE CLASSIFIER MODEL ===\n\n")
        for k, v in best.items():
            f.write(f"{k}: {v}\n")

    print("Best model summary saved to best_ridgeclassifier_model.txt")

if __name__ == "__main__":
    main()
