
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer #TF-IDF
from sklearn.feature_extraction.text import CountVectorizer #TO, T F
from sklearn.cluster import KMeans
from utils.text_processes import preprocess_text, tokenize_text
from utils.evaluate_clustering import evaluate_clustering
from utils.plot_clustering import plot_pca_2d, plot_pca_3d, plot_cluster_label_percentage

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
# Part3: Text Vectorization
df['processed_words'] = df['processed_words'].apply(lambda x : " ".join(x))
x = df['processed_words']                       # column to clustering

# Vectorize the processed text using vectorize
to_vectorizer = CountVectorizer()               # or pruning > CountVectorizer(min_df=2, max_df=0.95)
tf_vectorizer = TfidfVectorizer(use_idf=False)  # Set use_idf=False for pure TF
tfidf_vectorizer = TfidfVectorizer()

X_to = to_vectorizer.fit_transform(x)
X_tf = tf_vectorizer.fit_transform(x)
X_tfidf = tfidf_vectorizer.fit_transform(x)

# Select X for analysis (csr matrix)
X = X_tfidf

# --------------------------------------------------------------------------------------
# Part4: Text clustering with k-means
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Get the cluster labels
df['cluster'] = kmeans.labels_

# Evaluation model
s, ari, db = evaluate_clustering(X, df['cluster'], df['labels'])
print(f'\nKmean Clustering: \n'
      f'Silhouette Score: {s}'
      f'\nAdjusted Rand Index: {ari}'
      f'\nDavies-Bouldin Index: {db}')

# --------------------------------------------------------------------------------------
# Part5: Plot cluster
# Reduce the dimensions to 2D for visualization using PCA
plot_pca_2d(X, df_cluster=df['cluster'], title='K-Means Clustering of Text Data (2D PCA)', n_components=2)
plt.show()

# Reduce to 3D using PCA
plot_pca_3d(X, df_cluster=df['cluster'], title='K-Means Clustering of Text Data (3D PCA)',n_components=3)
plt.show()

# --------------------------------------------------------------------------------------
# Part6: plot cluster and categories
# Assuming 'df' contains your processed data with 'cluster' and 'label' columns
# Calculate proportion of each label in each cluster
cluster_label_distribution = df.groupby(['cluster', 'labels']).size().unstack(fill_value=0)
print(cluster_label_distribution)

# Calculate percentage distribution for each cluster
cluster_label_percentage = cluster_label_distribution.div(cluster_label_distribution.sum(axis=1), axis=0) * 100
print(cluster_label_percentage)

# Plot the percentage distribution of labels in each cluster as a stacked bar plot
plot_cluster_label_percentage(cluster_label_percentage, 'Proportion of Each Label in Clusters')

plt.show()