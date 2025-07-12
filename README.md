# 📝 Text Analytic Project
![Last Commit](https://img.shields.io/github/last-commit/JPP-J/text_analytic_project?style=flat-square)
![Python](https://img.shields.io/badge/Python-97.8%25-blue?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/text_analytic_project?style=flat-square)

This repository contains the code and report for **Jidapa's Text Analytic Project**

## 📌 Overview

This project applies various **machine learning techniques** for **text classification and clustering** to analyze and categorize textual data effectively. Using both **Python** and **RapidMiner**, it explores preprocessing, feature extraction, and modeling approaches to provide insights into text-based datasets.

### 🧩 Problem Statement

Classifying and grouping text data accurately is essential for tasks such as content categorization, sentiment analysis, and information retrieval. However, text data is often noisy, high-dimensional, and sparse, which makes traditional analysis difficult. This project aims to develop robust machine learning models that can automate text classification and clustering to improve understanding and usability of textual information.

### 🔍 Approach

- **Preprocessing:** Tokenization, stemming, stop-word removal to clean text data.
- **Feature Extraction:** Vectorization using TF-IDF and term occurrence to convert text into numeric form.
- **Clustering:** K-Means and K-Medoids to group documents based on similarity.
- **Classification:** Naive Bayes, k-NN, and SVM models to label text with predefined categories.
- **Evaluation:** Use of metrics like silhouette score for clustering and accuracy/F1-score for classification to assess performance.

### 🎢 Processes

1. Data preprocessing and cleaning.
2. Exploratory text analysis.
3. Feature vectorization (TF-IDF, term occurrence).
4. Clustering and classification model training.
5. Model evaluation and comparison.
6. Deployment of workflows in Python and RapidMiner.

### 🎯 Results & Impact

- High classification accuracy with SVM (98.35%), supporting reliable automated text categorization.  
- Clustering provided meaningful groupings, though with some overlap.  
- Facilitates faster and more consistent analysis of large text datasets, reducing manual effort.  
- Enables quicker categorization and real-time updates for news and social media content, improving responsiveness.  
- Offers flexible pipelines usable by data scientists and analysts for various text analytics tasks.

### ⚙️ Development Challenges

- Handling noisy and high-dimensional textual data effectively.
- Choosing optimal feature extraction methods for different algorithms.
- Balancing interpretability and performance between classical ML models.
- Integrating Python and RapidMiner workflows for complementary analysis.


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

---

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
