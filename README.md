# L2 Data Science & Text Mining — Practicals

Course for L2 Math students (35 students). 6 sessions + 1 project.

## Structure

| Session | Topic | Notebook |
|---------|-------|----------|
| **Session 1** | Foundations of ML | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/L2Math/blob/main/session1/tp1.ipynb) |
| **Session 2** | Non-Parametric Models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/L2Math/blob/main/session2/tp2.ipynb) |
| **Session 3** | Ensemble Methods | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/L2Math/blob/main/session3/tp3.ipynb) |
| **Session 4** | Neural Networks & Tuning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/L2Math/blob/main/session4/tp4.ipynb) |
| **Session 5** | Text Extraction & Classification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/L2Math/blob/main/session5/tp5.ipynb) |
| **Session 6** | Embeddings & Word Representations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/L2Math/blob/main/session6/tp6.ipynb) |
| **Project** | Temperature Prediction (24h) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/L2Math/blob/main/project/project.ipynb) |

## Content Summary

### Session 1 — Foundations of ML
Linear regression, logistic regression, MSE/MAE/R², precision/recall/F1/AUC, preprocessing (scaling, encoding, imputation), train/test split, k-fold cross-validation, sklearn pipelines.

### Session 2 — Non-Parametric Models
KNN (effect of k, scaling), Decision Trees (depth, overfitting), SVM (linear vs RBF kernel), decision boundary visualization, model comparison on Wine dataset.

### Session 3 — Ensemble Methods
Bias-variance tradeoff (empirical), Random Forest (bagging, n_estimators, feature importance), Gradient Boosting (learning rate, sequential correction), full model comparison with CV.

### Session 4 — Neural Networks & Tuning
MLP with sklearn (architectures, activations), loss curves, GridSearchCV, RandomizedSearchCV, comparison with ensemble methods.

### Session 5 — Text Extraction & Classification
Tokenization, stopwords, Bag of Words, TF-IDF, text classification pipeline on 20 Newsgroups, model comparison (LogReg, NB, SVM), feature importance for text.

### Session 6 — Embeddings
Pretrained Word2Vec exploration (analogies, similarity), document embeddings (mean pooling), embeddings vs TF-IDF comparison, conceptual intro to transformers.

### Project — Temperature Prediction
Predict 24h temperature for 10 cities from 1000 cities x 2 weeks hourly data. Teams of 2-3, evaluated on ML-Arena. Grading: 50% performance + 50% presentation.

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```
