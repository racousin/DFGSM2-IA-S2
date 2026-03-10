# DFGSM2 — Introduction to Machine Learning for Medical Students

3 practical sessions (TP) for 2nd-year medical students. Each session builds on the previous one, following a real ML methodology: **Explore → Visualize → Prepare → Train → Evaluate**.

## Structure

| Session | Topic | Notebook |
|---------|-------|----------|
| **Session 1** | Classification & Logistic Regression (Breast Cancer) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/L2Math/blob/main/session1/tp1.ipynb) |
| **Session 2** | Data Preparation & Random Forest (Breast Cancer with noise) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/L2Math/blob/main/session2/tp1.ipynb) |
| **Session 3** | Neural Networks & Model Comparison (Heart Disease) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/L2Math/blob/main/session3/tp1.ipynb) |

## Session Progression

### Session 1 — Classification & Logistic Regression
Clean Breast Cancer Wisconsin dataset. Learn the classification pipeline: explore data, visualize distributions, train logistic regression, evaluate with accuracy/precision/recall/F1, confusion matrix, cross-validation.

### Session 2 — Data Preparation & Random Forest
Same dataset but with realistic noise (missing values, outliers, categorical features). Learn data preparation: imputation, outlier handling (IQR), one-hot encoding, scaling. Introduce Random Forest and hyperparameter tuning with GridSearchCV.

### Session 3 — Neural Networks & Model Comparison
New dataset (Heart Disease). Apply the full pipeline from sessions 1-2. Introduce MLPClassifier (neural network), experiment with architectures and loss curves. Final comparison of all 3 models: Logistic Regression vs Random Forest vs Neural Network.

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```
