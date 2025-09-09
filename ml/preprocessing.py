from utils.data_loader import load_data
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def addiction_df_create(path, features):
    
    df = load_data(path)
    
    addiction_df = df[features].copy()
    addiction_df.dropna(inplace=True)
    
    return addiction_df



def feature_corr(df, features):
    
    #firstly, i should check the correlations between addiction metrics, if there exist higher values then 0.8, the metrics with high f score should be stay in df
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt= ".2f")
    plt.title('Addiction Metrics Correlation Matrix', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    #after the graph, i see all of them have unique distinctiveness, so addiction metrics stay same for now.



def feature_histogram(addiction_df, features):
        
    df = addiction_df.copy()  # ← Bu satır başta olmalı
    df[features].hist(bins=20, figsize=(12, 8))
        
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    #non of them right skewed graph, no need to log1p transformation.





#def normalize_features(df):
    df = df.copy()
    
    # log transform (çok büyük outlier’ları bastırır)
    
    if 'Phone_Checks_Per_Day' in df.columns:
        df['Phone_Checks_Per_Day'] = np.log1p(df['Phone_Checks_Per_Day'])
    else:
        print("Uyarı: 'Phone_Checks_Per_Day' sütunu bulunamadı!")
        print("Mevcut sütunlar:", df.columns.tolist())
    
    if 'Depression_Level' in df.columns:
        df['Depression_Level'] = np.log1p(df['Depression_Level'])
    else:
        print("Uyarı: 'Depression_Level' sütunu bulunamadı!")


    return df


def apply_pca(df, n_components=2):
    
    #PCA works with variance so datas should be scaled. If not, big metrics will be dominate in pca
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    pca = PCA(n_components=n_components, random_state=42)
    df_pca = pca.fit_transform(df_scaled)
    
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(df_pca, columns=pca_columns, index=df.index)
    
    
    return df_pca, pca, scaler


