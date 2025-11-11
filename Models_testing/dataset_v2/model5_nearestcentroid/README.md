### **Nearest Centroid**

**Description:**
Trains and evaluates a Nearest Centroid classifier on the PRDBv3 dataset. The algorithm classifies samples based on the closest class centroid in feature space.
A small grid search is conducted over distance metrics and shrinkage thresholds.

**Hyperparameter Grid:**

* metric: euclidean, manhattan
* shrink_threshold: None, 0.1, 0.5, 1.0

**Outputs:**

* Per-run results: `outputs/output_metric*_shrink*.txt`
* Summary file: `nearestcentroid_results.csv`
* Best configuration: `best_nearestcentroid_model.txt`

**Pseudocode:**

```
load dataset
scale features
split into train/test
for metric in [euclidean, manhattan]:
    for shrink_threshold in [None, 0.1, 0.5, 1.0]:
        train NearestCentroid
        evaluate (Accuracy, Precision, Recall, F1)
        save results
aggregate all results
select best by F1 > Accuracy
save summary and best configuration
```