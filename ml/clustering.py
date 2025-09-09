
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .preprocessing import addiction_df_create
from .preprocessing import feature_histogram, feature_corr, apply_pca



def clustering_by_all(path, features, k_range):

    addiction_df = addiction_df_create(path, features).reset_index(drop=True)
    
    feature_corr(addiction_df, features)
    feature_histogram(addiction_df, features)

    #addiction_df = normalize_features(addiction_df) 
    
    if addiction_df.isna().any().any():
        print("Warning: Missing values detected in the dataset. Filling with mean...")
        addiction_df = addiction_df.fillna(addiction_df.mean(numeric_only=True))
        
    
    df_pca, _, _ = apply_pca(addiction_df, n_components=2)

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
    
    df_pca["Cluster"] = addiction_df["Cluster"].values

    return addiction_df,k_values, wcss, silhouette_scores, best_k, df_pca



def standardization_process(path, features, n_clusters):
    
    addiction_df = addiction_df_create(path, features).reset_index(drop=True)
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

