# 🧬 Binding Affinity Feature Correlation & Selection Script

**Purpose:**
This script identifies and visualizes the **most biologically relevant features correlated with protein–RNA binding affinity**.
It performs preprocessing, encoding, correlation analysis, feature selection, and visualization — saving both a **derived dataset** and a **correlation heatmap**.

---

## 📘 Overview of Workflow

| Step                            | Description                                                                                                          |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **1. Load dataset**             | Imports the input JSON dataset containing protein–RNA complex details.                                               |
| **2. Feature selection**        | Keeps biologically meaningful fields relevant to structural or interaction properties.                               |
| **3. Target encoding**          | Converts `Binding_affinity` (“yes” / “no”) into binary form (1 / 0).                                                 |
| **4. Categorical encoding**     | Transforms text-based categorical fields (`Structural_class`, etc.) into numerical codes using `LabelEncoder`.       |
| **5. Missing value handling**   | Replaces missing numeric values with the mean and categorical ones with the mode.                                    |
| **6. Correlation computation**  | Calculates a full correlation matrix among all numeric features.                                                     |
| **7. Top feature selection**    | Identifies the 15 features with the **highest absolute correlation** (positive or negative) with `Binding_affinity`. |
| **8. Derived dataset creation** | Builds a smaller dataset containing only the selected 15 features and the target.                                    |
| **9. Visualization**            | Creates and saves a heatmap showing correlations among the selected features.                                        |
| **10. Output saving**           | Saves: (a) `derived_top15_features.json` for modeling and (b) `top15_correlation_heatmap.png` for analysis.          |

---

## 🧩 Key Feature Explanation

| Feature                                                                        | Type        | Description                             | Biological Relevance                                          |
| ------------------------------------------------------------------------------ | ----------- | --------------------------------------- | ------------------------------------------------------------- |
| `Structural_class`                                                             | categorical | Protein structure type (A–D)            | Indicates protein fold/family; can influence docking ability. |
| `Flexible_class`                                                               | categorical | Protein flexibility level (R, S, etc.)  | Flexibility may affect conformational binding.                |
| `Docking_case`                                                                 | categorical | Type of docking scenario (UU, UB, etc.) | Directly tied to binding mode.                                |
| `has_tRNA`                                                                     | binary      | Whether complex includes tRNA           | RNA type can define binding strength.                         |
| `has_dsRNA`                                                                    | binary      | Whether complex includes dsRNA          | Similar effect as above.                                      |
| `organism_match`                                                               | binary      | Protein and RNA from same organism      | Cross-species pairs may reduce affinity.                      |
| `is_NMR_structure`                                                             | binary      | Structure obtained via NMR or X-ray     | May reflect structural accuracy, affecting affinity.          |
| `RNA_chain_count`                                                              | numeric     | Number of RNA chains in complex         | Proxy for complex size.                                       |
| `pro_seq_missing`, `RNA_seq_missing`                                           | numeric     | Missing residues or bases               | Missing sequence parts can weaken binding.                    |
| `C_pro_seq_length`, `C_RNA_seq_length`, `U_pro_seq_length`, `U_RNA_seq_length` | numeric     | Sequence lengths                        | Larger molecules may have greater binding surfaces.           |
| `avg_resolution`                                                               | numeric     | Average structure resolution            | Lower = better quality structure.                             |

---

## 🧮 Correlation Analysis Logic

* Uses **Pearson correlation** (default in Pandas) for numeric relationships.
* Calculates the correlation of each feature with the binary target `Binding_affinity`.
* Takes the **top 15** features with the largest absolute correlation values:
  [
  |corr(feature, Binding_affinity)| \rightarrow \text{Ranked descending}
  ]
* Both positive and negative correlations are considered — as both can be biologically meaningful.

---

## 📊 Outputs Generated

| File                                | Description                                                                                                            |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **`derived_top15_features.json`**   | JSON file containing only the top 15 correlated features and `Binding_affinity`. Used for modeling or deeper analysis. |
| **`top15_correlation_heatmap.png`** | Visual correlation matrix among top features. Red = positive correlation, Blue = negative correlation.                 |

---

## ⚙️ Dependencies

* **Python 3.8+**
* **Pandas** – data manipulation
* **Seaborn** & **Matplotlib** – visualization
* **Scikit-learn** – for label encoding

Install via:

```bash
pip install pandas seaborn matplotlib scikit-learn
```

---

## 🧠 Interpretation Guide

* A **red block** in the heatmap indicates that the feature **increases with higher binding affinity**.
* A **blue block** indicates that the feature **decreases with binding affinity** (negative correlation).
* The **top 15 derived features** are those with the strongest linear relationship to `Binding_affinity`.

---

## 📈 Example Usage

```bash
python analyze_binding_affinity.py
```

**Output:**

```
✅ Saved:
- Derived dataset → derived_top15_features.json
- Heatmap → top15_correlation_heatmap.png

Top 15 correlations with Binding_affinity:
Flexible_class          0.52
Docking_case           -0.47
has_tRNA                0.41
...
```

---

## 🧾 Summary

This script acts as a **feature selection and exploratory analysis tool** for protein–RNA binding affinity studies.
It enables:

* Quick identification of influential biological properties
* Clean dataset extraction for model input
* Clear visualization of correlation structure