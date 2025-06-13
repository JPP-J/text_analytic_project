# 📝 Text Analytic Project
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/text_analytic_project?style=flat-square)
![Python](https://img.shields.io/badge/Python-97.8%25-blue?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/text_analytic_project?style=flat-square)

This repository contains the code and report for **Jidapa's Text Analytic Project**, which applies various machine learning techniques to **text classification and clustering** tasks. Implementations were done using both **Python** and **RapidMiner**.


## 📄 Report

🔗 [Text Classification and Clustering Report](https://drive.google.com/file/d/1hwf-yAWtEjgi-CYBG3U80RKYMJwHa6mS/view?usp=sharing)

The report covers:
- Text preprocessing: tokenization, stemming, stop-word removal
- Vectorization: Term Frequency-Inverse Document Frequency (TF-IDF), Term Occurrence
- ML Models for classification and clustering
- Results from both Python scripts and RapidMiner workflows



## 🧪 Methods Used

| Task             | Python                            | RapidMiner                         |
|------------------|-----------------------------------|------------------------------------|
| **Clustering**   | `K-Means`                         | `K-Means`, `K-Medoids`             |
| **Classification** | `Naive Bayes`, `k-NN`, `SVM`     | `Naive Bayes`, `k-NN`, `SVM`       |



## 🗂 File Structure

| File                                                   | Description                                                  |
|--------------------------------------------------------|--------------------------------------------------------------|
| [`text_setting.py`](text_setting.py)                   | Preprocessing setup: tokenization, stop-word removal, etc.   |
| [`text_exploration.py`](text_exploration.py)           | Exploratory word analysis for each category                  |
| [`text_clustering.py`](text_clustering.py)             | K-Means clustering on text vector data                       |
| [`text_classification.py`](text_classification.py)     | Training classification models (Naive Bayes, k-NN, SVM)      |
| [`text_clsf_usage_model.py`](text_clsf_usage_model.py) | Re-using trained classification model for prediction         |


## 📊 Clustering Results

**K-Means Clustering Metrics**:
- **Silhouette Score**: `0.0145`
- **Adjusted Rand Index**: `0.7696`
- **Davies-Bouldin Index**: `7.8209`

**Cluster Distribution (true label counts per cluster)**:

| Cluster | Business | Entertainment | Politics | Sport | Tech |
|---------|----------|----------------|----------|-------|------|
| 0       | 496      | 30             | 127      | 1     | 16   |
| 1       | 4        | 0              | 284      | 0     | 0    |
| 2       | 1        | 347            | 0        | 1     | 5    |
| 3       | 8        | 5              | 3        | 0     | 363  |
| 4       | 1        | 4              | 3        | 509   | 17   |



## 🤖 Classification Results

### ✅ Naive Bayes
- **Accuracy**: `96.7%`
- **Macro F1-score**: `0.97`

| Class          | Precision | Recall | F1-score |
|----------------|-----------|--------|----------|
| Business       | 0.96      | 0.98   | 0.97     |
| Entertainment  | 0.99      | 0.93   | 0.96     |
| Politics       | 0.91      | 0.99   | 0.95     |
| Sport          | 0.99      | 0.99   | 0.99     |
| Tech           | 0.98      | 0.93   | 0.95     |



### ✅ k-NN
- **Accuracy**: `71.4%`
- **Macro F1-score**: `0.63`

| Class          | Precision | Recall | F1-score |
|----------------|-----------|--------|----------|
| Business       | 0.87      | 0.81   | 0.84     |
| Entertainment  | 0.00      | 0.00   | 0.00     |
| Politics       | 0.66      | 0.92   | 0.77     |
| Sport          | 0.60      | 0.97   | 0.74     |
| Tech           | 0.81      | 0.78   | 0.80     |


### ✅ SVM (Best Performer)
- **Test Accuracy**: `98.35%`
- **Best Params**: `C=1`, `gamma=scale`, `kernel=linear`
- **Cross-validated Accuracy**: `97.69%`

| Class          | Precision | Recall | F1-score |
|----------------|-----------|--------|----------|
| Business       | 0.99      | 0.98   | 0.98     |
| Entertainment  | 0.97      | 0.99   | 0.98     |
| Politics       | 0.97      | 0.99   | 0.98     |
| Sport          | 1.00      | 0.99   | 1.00     |
| Tech           | 0.99      | 0.97   | 0.98     |



## 📌 Summary

- SVM performed best in classification with almost perfect precision, recall, and F1-scores.
- Naive Bayes also achieved excellent performance.
- Clustering via K-Means showed reasonable grouping but had limited silhouette score, indicating sparse separation.
- RapidMiner and Python both support flexible text analysis workflows from preprocessing to evaluation.



## 📁 Requirements

- Python 3.8+
- `scikit-learn`, `pandas`, `numpy`, `nltk`, `matplotlib`
