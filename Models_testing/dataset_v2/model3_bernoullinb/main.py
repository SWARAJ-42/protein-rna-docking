import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import Binarizer, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

def load_dataset(dataset_path):
    """Load dataset from JSON file."""
    with open(dataset_path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def preprocess_data(df):
    """Scale and binarize features for BernoulliNB."""
    X = df.drop(columns=["Binding_affinity"])
    y = df["Binding_affinity"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    binarizer = Binarizer(threshold=0.5)
    X_bin = binarizer.fit_transform(X_scaled)

    return X_bin, y

def train_model(X_train, y_train, alpha, fit_prior):
    """Train BernoulliNB with chosen parameters."""
    model = BernoulliNB(alpha=alpha, fit_prior=fit_prior)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Compute standard classification metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1_Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
    }
    report = classification_report(y_test, y_pred, digits=4)
    return metrics, report

def save_results(params, metrics, report, filename):
    """Save results to text file."""
    with open(filename, "w") as f:
        f.write("=== BernoulliNB Training Summary ===\n\n")
        f.write("Model Settings:\n")
        for k, v in params.items():
            f.write(f"{k}: {v}\n")
        f.write("\n=== Evaluation Metrics ===\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n--- Detailed Classification Report ---\n")
        f.write(report)
    print(f"Saved: {filename}")

def main():
    DATA_PATH = "../PRDBv3_engineered_v2.json"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Hyperparameter grid
    ALPHAS = [0.1, 0.5, 1.0, 2.0, 3.0]
    FIT_PRIORS = [True, False]

    # Load and preprocess
    df = load_dataset(DATA_PATH)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results = []

    for alpha in ALPHAS:
        for fit_prior in FIT_PRIORS:
            print(f"Training with alpha={alpha}, fit_prior={fit_prior}")
            model = train_model(X_train, y_train, alpha, fit_prior)
            metrics, report = evaluate_model(model, X_test, y_test)

            params = {
                "alpha": alpha,
                "fit_prior": fit_prior,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
            }

            filename = f"./outputs/output_alpha{alpha}_prior{fit_prior}.txt"
            save_results(params, metrics, report, filename)

            results.append({"alpha": alpha, "fit_prior": fit_prior, **metrics})

    # Summarize and select best
    results_df = pd.DataFrame(results)
    results_df.to_csv("./bernoullinb_results.csv", index=False)
    print("All results saved to bernoullinb_results.csv")

    best = results_df.sort_values(by=["F1_Score", "ROC_AUC"], ascending=False).iloc[0]
    print("\nBest configuration:")
    print(best)

    with open("./best_bernoullinb_model.txt", "w") as f:
        f.write("=== BEST BERNOULLI NAIVE BAYES MODEL ===\n\n")
        for k, v in best.items():
            f.write(f"{k}: {v}\n")
    print("Best model summary saved to best_bernoullinb_model.txt")

if __name__ == "__main__":
    main()
