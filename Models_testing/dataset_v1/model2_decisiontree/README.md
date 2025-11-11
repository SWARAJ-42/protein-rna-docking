### **DecisionTreeClassifier Model**

**Description:**
This script evaluates a `DecisionTreeClassifier` using a full grid of splitting criteria, tree depths, and minimum split sizes.
The goal is to find the optimal depth and splitting rules for classification accuracy.

**Hyperparameter Grid:**

| Parameter           | Values Tested           |
| ------------------- | ----------------------- |
| `criterion`         | gini, entropy, log_loss |
| `max_depth`         | 6, 12, 18, 24, 30       |
| `min_samples_split` | 2, 3, 4, 5              |
| `test_size`         | 0.2 (constant)          |
| `random_state`      | 42 (constant)           |

**Outputs:**

* Per-run reports: `output_c{criterion}_d{depth}_s{split}.txt`
* Summary table: `decisiontree_grid_results.csv`
* Best model record: `best_decisiontree_model.txt`

**Pseudo Code:**

```
load dataset
split into train/test
for criterion in [gini, entropy, log_loss]:
    for max_depth in [6, 12, 18, 24]:
        for min_samples_split in [2, 3, 4]:
            train DecisionTreeClassifier
            evaluate (Accuracy, Precision, Recall, F1, ROC_AUC)
            save results
select best by F1 > ROC_AUC
save summary
```