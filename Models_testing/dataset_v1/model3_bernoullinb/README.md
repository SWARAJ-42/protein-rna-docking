### **Bernoulli Naive Bayes Model**

**Description:**
This script runs a Bernoulli Naive Bayes classifier after scaling and binarizing input features.
It tests multiple smoothing (`alpha`) values and prior learning options to determine the optimal probabilistic configuration.

**Hyperparameter Grid:**

| Parameter      | Values Tested      |
| -------------- | ------------------ |
| `alpha`        | 0.1, 0.5, 1.0, 2.0, 3.0 |
| `fit_prior`    | True, False        |
| `test_size`    | 0.2 (constant)     |
| `random_state` | 42 (constant)      |

**Outputs:**

* Model reports: `output_alpha{X}_prior{Y}.txt`
* Summary results: `bernoullinb_results.csv`
* Best configuration: `best_bernoullinb_model.txt`

**Pseudo Code:**

```
load dataset
scale and binarize features
split into train/test
for alpha in [0.1, 0.5, 1.0, 2.0]:
    for fit_prior in [True, False]:
        train BernoulliNB
        evaluate (Accuracy, Precision, Recall, F1, ROC_AUC)
        save results
select best by F1 > ROC_AUC
save summary
```