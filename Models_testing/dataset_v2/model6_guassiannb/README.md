### **Gaussian naive Bias**

**Description:**
Trains and evaluates a Gaussian Naive Bayes model on the PRDBv3 dataset.
Tests multiple variance smoothing levels and prior distributions to optimize probabilistic classification performance.

**Hyperparameter Grid:**

* var_smoothing: 1e-9, 1e-8, 1e-7, 1e-6
* priors: None, [0.3, 0.7], [0.5, 0.5]

**Outputs:**

* Per-run results: `outputs/output_varsmooth*_priors*.txt`
* Summary file: `gaussiannb_results.csv`
* Best model: `best_gaussiannb_model.txt`

**Pseudocode:**

```
load dataset
scale features
split into train/test
for var_smoothing in [1e-9, 1e-8, 1e-7, 1e-6]:
    for priors in [None, [0.3, 0.7], [0.5, 0.5]]:
        train GaussianNB
        evaluate (Accuracy, Precision, Recall, F1, ROC_AUC)
        save results
aggregate results to CSV
select best model by F1 > ROC_AUC
save best configuration
```