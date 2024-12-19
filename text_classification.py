
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer #TF-IDF
from sklearn.feature_extraction.text import CountVectorizer #TO, T F
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from utils.text_processes import preprocess_text, tokenize_text
from utils.clsf_extended import SVM_parameters
import joblib

# Part1: Load dataset
path = "https://drive.google.com/uc?id=1-pp62M_iZB-3ZZTzMI_TWosJOV-a_6OA"
df = pd.read_csv(path)

print(f'example data:\n{df.head()}')
print(f'shape of data: {df.shape}')
print(f'columns name: {df.columns.values}')
print("\n")

# --------------------------------------------------------------------------------------
# Part2: text processing
# Apply the preprocessing function to the 'text' column : tokenization, lower cases, stopword and stemming
df['processed_words'] = df['data'].apply(preprocess_text)
# print(df[['data', 'processed_words']][0:5])

# Apply the preprocessing function to the 'text' column : tokenization and lower cases - optional
df['tokenize_words'] = df['data'].apply(tokenize_text)
# print(df[['data', 'tokenize_words']][0:5])

# --------------------------------------------------------------------------------------
# Part3: Text Vectorize
df['processed_words'] = df['processed_words'].apply(lambda x : " ".join(x))
x = df['processed_words'] # column to clustering

# Step 1: Vectorize the processed text using vectorize
to_vectorizer = CountVectorizer() # or pruning > CountVectorizer(min_df=2, max_df=0.95)
tf_vectorizer = TfidfVectorizer(use_idf=False)  # Set use_idf=False for pure TF
tfidf_vectorizer = TfidfVectorizer()

X_to = to_vectorizer.fit_transform(x)
X_tf = tf_vectorizer.fit_transform(x)
X_tfidf = tfidf_vectorizer.fit_transform(x)

# Assign X and y for analysis
X = X_tfidf
y = df['labels']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --------------------------------------------------------------------------------------
# Part4: Apply models
# Model-1: Naive Bayes
# Initialize and train the model
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Make predictions
y_pred_nb = nb_classifier.predict(X_test)

# Evaluate the model
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# Saved model for use later
joblib.dump(nb_classifier, 'nb_clsf_model.pkl')

# --------------------------------------------------------------------------------------
# Model-2: k-NN
# Standardize data
scaler = StandardScaler(with_mean=False)  # with_mean=False because X is sparse
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
knn_classifier = KNeighborsClassifier(n_neighbors=90)
knn_classifier.fit(X_train_scaled, y_train)

# Make predictions
y_pred_knn = knn_classifier.predict(X_test_scaled)

# Evaluate the model
print("\nk-NN Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# Saved model for use later
joblib.dump(knn_classifier, 'knn_clsf_model.pkl')

# --------------------------------------------------------------------------------------
# Model-3: SVM
# Initialize and train the model
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm_classifier.predict(X_test)

# Evaluate the model
print("\nSVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Saved model for use later
joblib.dump(svm_classifier, 'vsm_clsf_model.pkl')

# --------------------------------------------------------------------------------------
# Model-4: SVM (best parameters)
# Best parameters: {'C': 1, 'gamma': 'scale', 'kernel': 'linear'}
# Results No different from above SVM - optional

# Initialize and train the model
best_svm_classifier = SVM_parameters(X_train, y_train)
y_pred_best_svm = best_svm_classifier.predict(X_test)

# Evaluate the model
print("Test Accuracy:", accuracy_score(y_test, y_pred_best_svm))
print(classification_report(y_test, y_pred_best_svm))