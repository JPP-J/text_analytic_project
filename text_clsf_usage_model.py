import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from utils.text_processes import load_vectorize_text

# Load the saved model
nb_clsf_loaded = joblib.load('nb_clsf_model.pkl')
knn_clsf_loaded = joblib.load('knn_clsf_model.pkl')
vsm_clsf_loaded = joblib.load('vsm_clsf_model.pkl')


# Use the model to make predictions
path = "https://drive.google.com/uc?id=1-pp62M_iZB-3ZZTzMI_TWosJOV-a_6OA"

# text processes and get X and y from df
X_tfidf, y = load_vectorize_text(path, col_x='data', col_y='labels', type_vector='tfidf')
X = X_tfidf
y = y

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Prediction with model
y_pred = vsm_clsf_loaded.predict(X_test)  #SVM

# Evaluate the model
print("SVM Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))