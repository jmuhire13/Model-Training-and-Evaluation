# Diabetes Risk Prediction and CDC Health Indicators

Binary classification task: predict whether a person has diabetes using self-reported health survey data.

## Dataset

- **Source:** CDC BRFSS 2014 via [UCI ML Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- **Size:** 253,680 respondents, 21 features + 6 engineered = 27 total
- **Class imbalance:** ~86% non-diabetic, ~14% diabetic (addressed with SMOTE / class weights)

## Models Compared (11 experiments)

| Family | Models |
|--------|--------|
| Classical ML | Logistic Regression, Random Forest, XGBoost |
| Deep Learning | Sequential DNN (shallow & deep), Functional API (residual) |

**Best model:** XGBoost (tuned), **AUC 0.82** on the held-out test set.  
Deep learning models reached comparable AUC (~0.82) but offered no advantage over the gradient boosting approach on this tabular dataset.

## Key Findings

- Gradient boosting outperformed all other models despite deep learning's added complexity.
- Unlimited-depth Random Forest severely overfits (train AUC 0.9999, test AUC 0.78), depth-limiting fixes this.
- Top predictors: General Health self-rating, BMI, High Blood Pressure, and the engineered `Cardio_Risk_Score`.
- Lowering the classification threshold improves recall (catching more diabetic cases) at the cost of precision, the right trade-off for a clinical screening tool.

## Run It

Open `CDC_Diabetes_ML_Project.ipynb` in Jupyter or Google Colab. The notebook is fully self-contained and it fetches the dataset automatically via `ucimlrepo`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jmuhire13/Model-Training-and-Evaluation/blob/main/CDC_Diabetes_ML_Project.ipynb)
