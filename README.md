# Random Forest Classifier — Diabetes Prediction

> Ensemble classifier applied to the Pima Indians Diabetes dataset, building on the same EDA pipeline as the Decision Tree project — a `RandomForestClassifier(n_estimators=60)` achieves **76.0% accuracy** without any hyperparameter tuning, surpassing the tuned single tree's 71.4% and demonstrating the ensemble advantage.

---

## Problem

Predict whether a patient has diabetes based on diagnostic measurements. Same clinical context as the Decision Tree project: an interpretable vs. powerful model comparison on identical data — the core question being how much accuracy the ensemble gains over a single decision boundary.

## Dataset

- **Source:** Pima Indians Diabetes dataset (768 rows × 9 features)
- **Target:** `Outcome` — 1 = diabetes, 0 = no diabetes (65% / 35% class split)
- **Features used after cleaning:** Glucose, BMI, Age, Pregnancies (top 4 by SelectKBest f_classif)

The full EDA and preprocessing pipeline is identical to the Decision Tree project:

| Step | Action |
|---|---|
| Drop impossible columns | Insulin (48.7% zeros) and SkinThickness (29.6% zeros) removed |
| Zero imputation | Group-stratified median imputation on Glucose, BloodPressure, BMI |
| Outlier capping | IQR method on Pregnancies, DiabetesPedigreeFunction, Age |
| Scaling | StandardScaler |
| Feature selection | SelectKBest (f_classif, k=4) → Glucose, BMI, Age, Pregnancies |
| Split | 80/20 stratified (614 train / 154 test) |

## Model Results

| Model | Accuracy | Notes |
|---|---|---|
| Single Decision Tree (tuned, max_depth=5) | 71.4% | GridSearchCV, 10-fold CV |
| **Random Forest (n_estimators=60)** | **76.0%** | Default params, no tuning |

A Random Forest with default settings beats a carefully tuned single tree by **+4.6 percentage points**. This is the ensemble effect in practice: 60 trees each trained on a random bootstrap sample and random feature subset, their majority votes averaging out individual errors.

## Key Takeaways

- **Ensembles beat single trees without tuning:** The Random Forest achieves higher accuracy than the GridSearchCV-tuned Decision Tree without any hyperparameter optimisation. The diversity of 60 independently grown trees is more valuable than perfect tuning of one.
- **Bootstrap + feature randomness = reduced variance:** Each tree in the forest sees a different random sample of training data and a random subset of features at each split. This decorrelates the trees — their errors don't line up — so the ensemble vote is more reliable than any individual tree.
- **Accuracy vs. interpretability:** The single Decision Tree can be plotted and inspected — a clinician can follow the exact path from features to prediction. The Random Forest is a black box of 60 trees. The 4.6% accuracy gain has a real cost in explainability.

## Tech Stack

`Python` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## Run It Locally

```bash
git clone https://github.com/matthewkane-ml/ML_RandomForest_MTK.git
cd ML_RandomForest_MTK
pip install -r requirements.txt
jupyter notebook src/RandomForest.ipynb
```

The trained model is saved to `models/` via `pickle`.

## What I'd Do Next

- Run `GridSearchCV` on the Random Forest (`n_estimators`, `max_depth`, `min_samples_split`, `max_features`) to find the accuracy ceiling with proper tuning
- Plot **feature importances** from the forest — which of the 4 features does the ensemble lean on most heavily? Compare against the single tree's split structure
- Add **SHAP values** for individual predictions — this partially recovers interpretability from the ensemble without sacrificing accuracy

---

**Author:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [GitHub portfolio](https://github.com/matthewkane-ml)
