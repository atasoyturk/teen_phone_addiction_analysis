
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

from ml.preprocessing import addiction_df_create
from ml.preprocessing import feature_histogram, apply_pca



def clustering_by_all(path, k_range):

    addiction_df = addiction_df_create(path).reset_index(drop=True)
    feature_histogram(addiction_df)
    #non of them right skewed graph, no need to log1p transformation.

    #addiction_df = normalize_features(addiction_df) 
    
    if addiction_df.isna().any().any():
        print("Warning: Missing values detected in the dataset. Filling with mean...")
        addiction_df = addiction_df.fillna(addiction_df.mean(numeric_only=True))
        
    
    df_pca, pca_model, scaler = apply_pca(addiction_df, n_components=2)

    k_values = list(k_range)
    wcss = []
    silhouette_scores = []
    all_labels = {}  


    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df_pca) 

        wcss.append(kmeans.inertia_)
        # inertia_: toplam kare uzaklık (WCSS)
        
        if k == 1:
            score = 0
        else:
            score = silhouette_score(df_pca, kmeans.labels_)

        silhouette_scores.append(score)
        all_labels[k] = labels  

    best_k = k_values[silhouette_scores.index(max(silhouette_scores))]
    addiction_df["Cluster"] = all_labels[best_k]

    return addiction_df,k_values, wcss, silhouette_scores, best_k



def standardization_process(path, n_clusters):
    
    addiction_df = addiction_df_create(path)
    #addiction_df = normalize_features(addiction_df)
    
    if addiction_df.isna().any().any():
        print("Warning: Missing values detected in the dataset. Filling with mean...")
        addiction_df = addiction_df.fillna(addiction_df.mean(numeric_only=True))
    
    df_pca, _, _ = apply_pca(addiction_df, n_components=2)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_pca)
    
    df_pca = df_pca.copy()
    df_pca['Cluster'] = labels

    cluster_means = df_pca.groupby('Cluster')[df_pca.columns[:-1]].mean()

    return df_pca, cluster_means

