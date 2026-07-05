# Home Credit Default Risk — Machine Learning Project

**Course:** AI331 – Machine Learning | **Section:** M11

**Team:**
- Raed Abdulsalam Al-Maghdoowi
- Youssef Hluil Al-Juhani 
- Majed Mohammed Al-Harbi 

## Overview

This project tackles the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview) dataset from Kaggle. The goal is to predict whether a client will repay a loan or default, helping financial institutions extend safe credit to clients with limited or no credit history.

## Dataset

- **Source:** Kaggle — Home Credit Default Risk (`application_train.csv`)
- **Size:** 307,511 samples, 122 features
- **Target:** Binary (`0` = will repay, `1` = will default)
- **Class balance:** ~92% class 0, ~8% class 1 (highly imbalanced)

> Note: the raw CSV is not included in this repo due to size/licensing. Download it from Kaggle and place it locally before running the notebook.

## Methodology

1. **Preprocessing**
   - Split numeric vs. categorical columns
   - Impute missing numeric values with the median
   - Impute missing categorical values with `"Unknown"`
   - Feature engineering (e.g. `DAYS_BIRTH` → `YEARS_AGE`)
   - Scale numeric features (`StandardScaler`) and encode categoricals (`OrdinalEncoder`) via `ColumnTransformer`

2. **Models trained**
   - Decision Tree
   - Random Forest (`n_estimators=500`)
   - Logistic Regression

   Each model was trained twice — with default parameters and with `class_weight='balanced'` — to address class imbalance.

3. **Hyperparameter tuning**
   - `RandomizedSearchCV` applied to Logistic Regression (best-performing model), tuning `C` and `solver`, optimizing for **Recall**.

4. **Evaluation metrics**
   - Precision, Recall, F1-score, AUC-ROC (Recall chosen as the primary metric, since false negatives — approving a high-risk client — are the costliest error)

## Results Summary

| Model | Accuracy | Recall (class 1) | AUC |
|---|---|---|---|
| Decision Tree | 86% | 0.15 | 0.54 |
| Random Forest | 92% | 0.00 | 0.74 |
| Logistic Regression (balanced) | 69% | 0.67 | 0.75 |
| Logistic Regression (tuned) | 69% | 0.67 | **0.747** |

**Key finding:** accuracy is misleading on imbalanced data. Logistic Regression with `class_weight='balanced'` gave the best real-world performance, correctly flagging 67% of clients likely to default — the most valuable outcome for risk-sensitive lending decisions.

## Repository Structure

```
.
├── home_credit_default_risk.ipynb   # Full notebook: preprocessing, modeling, evaluation
├── report/                          # Project report (PDF)
├── requirements.txt                 # Python dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
jupyter notebook home_credit_default_risk.ipynb
```

## References

- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow
- [IBM: What is Machine Learning?](https://www.ibm.com/think/machine-learning)
- [GeeksforGeeks: Random Forest Algorithm](https://www.geeksforgeeks.org/machine-learning/random-forest-algorithm-in-machine-learning/)
- [Scikit-learn documentation](https://scikit-learn.org)
- [Evidently AI: Classification Metrics Guide](https://www.evidentlyai.com/classification-metrics/accuracy-precision-recall#what-is-recall)
- [Google Developers: Accuracy, Precision, Recall](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall)
