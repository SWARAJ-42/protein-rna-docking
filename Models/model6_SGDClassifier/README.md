### **SGD classifier Model**

**Description:**
Trains an SGDClassifier (Stochastic Gradient Descent-based linear model) on the PRDBv3 dataset, exploring different loss functions, regularization penalties, learning strengths, and iteration limits. Evaluates each setup and identifies the best-performing configuration.

**Hyperparameter Grid:**

* loss: hinge, log_loss, modified_huber
* penalty: l2, l1, elasticnet
* alpha: 0.0001, 0.001, 0.01
* max_iter: 1000, 2000

**Outputs:**

* Individual results: `outputs/output_loss*_pen*_alpha*_iter*.txt`
* Summary: `sgdclassifier_results.csv`
* Best configuration: `best_sgdclassifier_model.txt`

**Pseudocode:**

```
load dataset
scale features
split into train/test
for loss in [hinge, log_loss, modified_huber]:
    for penalty in [l2, l1, elasticnet]:
        for alpha in [0.0001, 0.001, 0.01]:
            for max_iter in [1000, 2000]:
                train SGDClassifier
                evaluate (Accuracy, Precision, Recall, F1, ROC_AUC)
                save results
aggregate results
select best by F1 > ROC_AUC
save summary and best config
```