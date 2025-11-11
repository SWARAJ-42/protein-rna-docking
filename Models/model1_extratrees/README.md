### **ExtraTreesClassifier Model**

**Description:**
This script trains and evaluates an `ExtraTreesClassifier` on the PRDBv3 dataset.
It performs a grid search over tree count and maximum depth to identify the best-performing configuration.

**Hyperparameter Grid:**

| Parameter      | Values Tested  |
| -------------- | -------------- |
| `n_estimators` | 100, 200, 300  |
| `max_depth`    | 10, 20, 30     |
| `test_size`    | 0.2 (constant) |
| `random_state` | 42 (constant)  |

**Outputs:**

* Individual result files: `output_n{n_est}_d{depth}.txt`
* Summary file: `grid_results_summary.csv`
* Best model summary: `best_model_summary.txt`

**Pseudo Code:**

```
load dataset
split into train/test
for n_estimators in [100, 200, 300]:
    for max_depth in [10, 20, 30]:
        train ExtraTreesClassifier
        evaluate (Accuracy, Precision, Recall, F1, ROC_AUC)
        save results
select best by F1 > ROC_AUC
save summary
```