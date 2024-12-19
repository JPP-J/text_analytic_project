import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score



# Function to evaluate clustering
def evaluate_clustering(X, cluster_lab, cat_lab):
    silhouette_avg = silhouette_score(X, cluster_lab)
    ari_score = adjusted_rand_score(cat_lab, cluster_lab)
    dbi_score = davies_bouldin_score(X.toarray(), cluster_lab)
    return silhouette_avg, ari_score , dbi_score

# s = silhouette_score(X_tfidf, df['cluster']) #silhouette_score
# print(f'Silhouette Score: {s}')
#
# # If you have ground truth labels and want to evaluate clustering against actual labels:
# # For example, adjusted Rand Index (requires labels)
# ari_score = adjusted_rand_score(df['labels'], df['cluster'])
# print(f'Adjusted Rand Index: {ari_score}')
#
#
# # Compute the Davies-Bouldin Index Suitable For DBSCAN
# dbi_score = davies_bouldin_score(X_tfidf.toarray(), df['cluster'])
# print(f'Davies-Bouldin Index: {dbi_score}')