## **RandomForestClassifier Model Implementation**

### **Script: `train_randomforest_grid.py`**

```python
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
    """Scale numerical features for the RandomForest model."""
    X = df.drop(columns=["Binding_affinity"])
    y = df["Binding_affinity"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def train_model(X_train, y_train, n_estimators, max_depth, min_samples_split, min_samples_leaf):
    """Train RandomForestClassifier with given hyperparameters."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
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
    """Save model configuration, metrics, and report."""
    with open(output_path, "w") as f:
        f.write("=== RandomForestClassifier Training Summary ===\n\n")
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
    N_ESTIMATORS = [100, 200, 300]
    MAX_DEPTHS = [10, 20, 30]
    MIN_SAMPLES_SPLITS = [2, 4]
    MIN_SAMPLES_LEAFS = [1, 2]

    os.makedirs("outputs", exist_ok=True)

    df = load_dataset(DATA_PATH)
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results = []

    for n_est in N_ESTIMATORS:
        for depth in MAX_DEPTHS:
            for split in MIN_SAMPLES_SPLITS:
                for leaf in MIN_SAMPLES_LEAFS:
                    print(f"Training RandomForest: n_estimators={n_est}, max_depth={depth}, "
                          f"min_samples_split={split}, min_samples_leaf={leaf}")

                    model = train_model(X_train, y_train, n_est, depth, split, leaf)
                    metrics, report = evaluate_model(model, X_test, y_test)

                    params = {
                        "n_estimators": n_est,
                        "max_depth": depth,
                        "min_samples_split": split,
                        "min_samples_leaf": leaf,
                        "test_size": TEST_SIZE,
                        "random_state": RANDOM_STATE
                    }

                    filename = f"outputs/output_n{n_est}_d{depth}_split{split}_leaf{leaf}.txt"
                    save_results(params, metrics, report, filename)

                    results.append({
                        "n_estimators": n_est,
                        "max_depth": depth,
                        "min_samples_split": split,
                        "min_samples_leaf": leaf,
                        **metrics
                    })

    # Save summary
    results_df = pd.DataFrame(results)
    results_df.to_csv("randomforest_results.csv", index=False)
    print("All results saved to randomforest_results.csv")

    # Identify best model
    best = results_df.sort_values(by=["F1_Score", "ROC_AUC"], ascending=False).iloc[0]
    print("\nBest configuration:")
    print(best)

    with open("best_randomforest_model.txt", "w") as f:
        f.write("=== BEST RANDOM FOREST MODEL ===\n\n")
        for k, v in best.items():
            f.write(f"{k}: {v}\n")

    print("Best model summary saved to best_randomforest_model.txt")

if __name__ == "__main__":
    main()
```

---

### **Hyperparameter Grid Summary**

| Parameter           | Values Tested |
| ------------------- | ------------- |
| `n_estimators`      | 100, 200, 300 |
| `max_depth`         | 10, 20, 30    |
| `min_samples_split` | 2, 4          |
| `min_samples_leaf`  | 1, 2          |
| `test_size`         | 0.2           |
| `random_state`      | 42            |

This results in **36 total model configurations** (3×3×2×2).

---

### **Output Structure**

```
/RandomForest/
   train_randomforest_grid.py
   /outputs/
       output_n100_d10_split2_leaf1.txt
       output_n200_d20_split4_leaf2.txt
       ...
   randomforest_results.csv
   best_randomforest_model.txt
   README.md
```

---

### **Brief Documentation (for README.md)**

**Description:**
Trains and evaluates a Random Forest Classifier on the PRDBv3 dataset. Performs a grid search across forest size, tree depth, and splitting rules to determine the best configuration for binding affinity classification.

**Hyperparameter Grid:**

* n_estimators: 100, 200, 300
* max_depth: 10, 20, 30
* min_samples_split: 2, 4
* min_samples_leaf: 1, 2

**Outputs:**

* Per-run results: `outputs/output_n*_d*_split*_leaf*.txt`
* Summary: `randomforest_results.csv`
* Best configuration: `best_randomforest_model.txt`

**Pseudocode:**

```
load dataset
scale features
split into train/test
for n_estimators in [100, 200, 300]:
    for max_depth in [10, 20, 30]:
        for min_samples_split in [2, 4]:
            for min_samples_leaf in [1, 2]:
                train RandomForestClassifier
                evaluate (Accuracy, Precision, Recall, F1, ROC_AUC)
                save results
aggregate results to CSV
select best by F1 > ROC_AUC
save best model configuration
```
