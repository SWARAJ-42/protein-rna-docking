### **Ridge Classifier**

**Description:**
Trains and evaluates a Ridge Classifier on the PRDBv3 dataset using standardized feature scaling and a grid of regularization and solver options.

**Hyperparameter Grid:**

* alpha: 0.1, 1.0, 10.0
* solver: auto, saga, lsqr
* fit_intercept: True, False

**Outputs:**

* Individual results: `outputs/output_alpha*_solver*_intercept*.txt`
* Summary table: `ridgeclassifier_results.csv`
* Best model record: `best_ridgeclassifier_model.txt`

**Pseudocode:**

```
load dataset
scale features
split into train/test
for alpha in [0.1, 1.0, 10.0]:
    for solver in [auto, saga, lsqr]:
        for fit_intercept in [True, False]:
            train RidgeClassifier
            evaluate (Accuracy, Precision, Recall, F1, ROC_AUC)
            save results
select best model by F1 > ROC_AUC
save summary
```
