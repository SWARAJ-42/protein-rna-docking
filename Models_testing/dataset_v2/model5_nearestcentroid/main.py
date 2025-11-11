import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid
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
    """Scale numerical features for NearestCentroid."""
    X = df.drop(columns=["Binding_affinity"])
    y = df["Binding_affinity"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def train_model(X_train, y_train, metric, shrink_threshold):
    """Train NearestCentroid classifier."""
    model = NearestCentroid(metric=metric, shrink_threshold=shrink_threshold)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Compute metrics for NearestCentroid."""
    y_pred = model.predict(X_test)

    # No predict_proba method; we can skip ROC_AUC or approximate via distance if needed
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1_Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": np.nan  # not applicable directly
    }

    report = classification_report(y_test, y_pred, digits=4)
    return metrics, report

def save_results(params, metrics, report, output_path):
    """Save training settings and metrics."""
    with open(output_path, "w") as f:
        f.write("=== NearestCentroid Training Summary ===\n\n")
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
    METRICS = ["euclidean", "manhattan"]
    SHRINK_THRESHOLDS = [None, 0.1, 0.5, 1.0]

    os.makedirs("outputs", exist_ok=True)

    df = load_dataset(DATA_PATH)
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results = []

    for metric in METRICS:
        for shrink in SHRINK_THRESHOLDS:
            print(f"Training NearestCentroid: metric={metric}, shrink_threshold={shrink}")

            model = train_model(X_train, y_train, metric, shrink)
            metrics, report = evaluate_model(model, X_test, y_test)

            params = {
                "metric": metric,
                "shrink_threshold": shrink,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE
            }

            filename = f"outputs/output_metric{metric}_shrink{shrink}.txt"
            save_results(params, metrics, report, filename)

            results.append({
                "metric": metric,
                "shrink_threshold": shrink,
                **metrics
            })

    # Save all results
    results_df = pd.DataFrame(results)
    results_df.to_csv("nearestcentroid_results.csv", index=False)
    print("All results saved to nearestcentroid_results.csv")

    # Determine best configuration
    best = results_df.sort_values(by=["F1_Score", "Accuracy"], ascending=False).iloc[0]
    print("\nBest configuration:")
    print(best)

    with open("best_nearestcentroid_model.txt", "w") as f:
        f.write("=== BEST NEAREST CENTROID MODEL ===\n\n")
        for k, v in best.items():
            f.write(f"{k}: {v}\n")

    print("Best model summary saved to best_nearestcentroid_model.txt")

if __name__ == "__main__":
    main()
