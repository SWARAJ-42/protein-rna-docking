### **SVC Model**

**Description:**
This script trains and evaluates a Support Vector Classifier on the PRDBv3 dataset using different kernel functions, regularization strengths, and kernel coefficient settings. It identifies the configuration achieving the best balance between generalization and classification performance.

**Hyperparameter Grid:**

* kernel: linear, rbf, poly
* C: 0.1, 1, 10
* gamma: scale, auto

**Outputs:**

* Individual runs: `outputs/output_kernel*_C*_gamma*.txt`
* Summary: `svc_results.csv`
* Best configuration: `best_svc_model.txt`

**Pseudocode:**

```
load dataset
scale features
split into train/test
for kernel in [linear, rbf, poly]:
    for C in [0.1, 1, 10]:
        for gamma in [scale, auto]:
            train SVC
            evaluate (Accuracy, Precision, Recall, F1, ROC_AUC)
            save results
aggregate results to CSV
select best by F1 > ROC_AUC
save best configuration
```
