### **NuSVC**

**Description:**
Trains and evaluates a Nu-Support Vector Classifier on the PRDBv3 dataset using various kernel functions, margin parameters (`nu`), and kernel coefficients (`gamma`, `degree`). Identifies the best-performing combination based on F1 and ROC_AUC.

**Hyperparameter Grid:**

* kernel: linear, rbf, poly
* nu: 0.25, 0.5, 0.75
* gamma: scale, auto
* degree: 2, 3

**Outputs:**

* Individual results: `outputs/output_kernel*_nu*_gamma*_deg*.txt`
* Summary: `nusvc_results.csv`
* Best model: `best_nusvc_model.txt`

**Pseudocode:**

```
load dataset
scale features
split into train/test
for kernel in [linear, rbf, poly]:
    for nu in [0.25, 0.5, 0.75]:
        for gamma in [scale, auto]:
            for degree in [2, 3]:
                train NuSVC
                evaluate (Accuracy, Precision, Recall, F1, ROC_AUC)
                save results
aggregate all results to CSV
select best by F1 > ROC_AUC
save best model configuration
```