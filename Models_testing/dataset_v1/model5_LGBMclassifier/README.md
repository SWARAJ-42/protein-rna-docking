### **LGBM Classifier**

**Description:**
Trains and evaluates a LightGBM classifier using boosting-based gradient trees for binary classification of RNA-protein binding affinity.
A full hyperparameter grid search is performed across tree complexity, learning rate, and boosting depth parameters.

**Hyperparameter Grid:**

* num_leaves: 15, 31, 63
* max_depth: 5, 10, -1
* learning_rate: 0.01, 0.05, 0.1
* n_estimators: 100, 200

**Outputs:**

* Per-run reports: `outputs/output_leaves*_depth*_lr*_nest*.txt`
* Summary table: `lgbmclassifier_results.csv`
* Best model configuration: `best_lgbmclassifier_model.txt`

**Pseudocode:**

```
load dataset
scale features
split into train/test
for num_leaves in [15, 31, 63]:
    for max_depth in [5, 10, -1]:
        for learning_rate in [0.01, 0.05, 0.1]:
            for n_estimators in [100, 200]:
                train LGBMClassifier
                evaluate (Accuracy, Precision, Recall, F1, ROC_AUC)
                save results
aggregate results to CSV
select best model by F1 > ROC_AUC
save best configuration
```