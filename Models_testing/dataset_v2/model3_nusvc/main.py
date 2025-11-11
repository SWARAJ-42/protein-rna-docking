import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC
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
    """Scale numerical features for NuSVC."""
    X = df.drop(columns=["Binding_affinity"])
    y = df["Binding_affinity"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def train_model(X_train, y_train, kernel, nu, gamma, degree):
    """Train NuSVC with given hyperparameters."""
    model = NuSVC(
        kernel=kernel,
        nu=nu,
        gamma=gamma,
        degree=degree,
        probability=True,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
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
    """Save model and evaluation details."""
    with open(output_path, "w") as f:
        f.write("=== NuSVC Training Summary ===\n\n")
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
    KERNELS = ["linear", "rbf", "poly"]
    NUS = [0.25, 0.5, 0.75]
    GAMMAS = ["scale", "auto"]
    DEGREES = [2, 3]

    os.makedirs("outputs", exist_ok=True)

    df = load_dataset(DATA_PATH)
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results = []

    for kernel in KERNELS:
        for nu in NUS:
            for gamma in GAMMAS:
                for degree in DEGREES:
                    print(f"Training NuSVC: kernel={kernel}, nu={nu}, gamma={gamma}, degree={degree}")

                    model = train_model(X_train, y_train, kernel, nu, gamma, degree)
                    metrics, report = evaluate_model(model, X_test, y_test)

                    params = {
                        "kernel": kernel,
                        "nu": nu,
                        "gamma": gamma,
                        "degree": degree,
                        "test_size": TEST_SIZE,
                        "random_state": RANDOM_STATE
                    }

                    filename = f"outputs/output_kernel{kernel}_nu{nu}_gamma{gamma}_deg{degree}.txt"
                    save_results(params, metrics, report, filename)

                    results.append({
                        "kernel": kernel,
                        "nu": nu,
                        "gamma": gamma,
                        "degree": degree,
                        **metrics
                    })

    # Save summary results
    results_df = pd.DataFrame(results)
    results_df.to_csv("nusvc_results.csv", index=False)
    print("All results saved to nusvc_results.csv")

    # Find best configuration
    best = results_df.sort_values(by=["F1_Score", "ROC_AUC"], ascending=False).iloc[0]
    print("\nBest configuration:")
    print(best)

    with open("best_nusvc_model.txt", "w") as f:
        f.write("=== BEST NuSVC MODEL ===\n\n")
        for k, v in best.items():
            f.write(f"{k}: {v}\n")

    print("Best model summary saved to best_nusvc_model.txt")

if __name__ == "__main__":
    main()
