from ml.clustering import standardization_process
from scipy.stats import f_oneway
''' 
    f testi (ANOVA), kümeler arasındaki ayırıcı faktörün (metrik) gerçekten ayırıcı mı yoksa şans eseri mi olduğunu araştırır.
f testi sadece bir metrik için test yapar!
H0 ve H1 hipotezi oluştururuz başta (H0 kümelerin ayırıcı faktörü arasında anlmalı fark yoktur H1 ise anlamlı fark vardır
f_oneway iki tane çıktı üretir : f değeri ve p değeri 
f<0.05 çıkarsa ayırıcı metrik doğruluğu yüksektir (güvenilebilir), ama >0.05 çıkarsa o metriğin ayırıcılığı şans eseri olabilir.
Ayrıca p<0.001 çıkarsa istatiksel olarak da çıkan hipotezin doğrıluğu yüksektir

'''
import pandas as pd
from ml.preprocessing import normalize_features, apply_pca, addiction_df_create


def standardization_info(path, best_k):
    df_pca, cluster_means = standardization_process(path, n_clusters=best_k)
    print("Cluster Means:\n", cluster_means)
    
    print("\nCluster Distribution:")
    distribution = df_pca['Cluster'].value_counts().sort_index()
    print(f"\n{distribution}")
    
    if (distribution < 2).any():
        print("Warning: One or more clusters have fewer than 2 samples, which may cause issues in F-test.")

    
def f_test(path, best_k):
    
    addiction_df = addiction_df_create(path)
    addiction_df = normalize_features(addiction_df)
    
    #tek any() kullanılırsa tüm sütunların true-false degeri gorunur. Yani birden fazla oldugu icin booelan bir deger olmaz
    #any().any() olursa sütunları mantıksal OR gibi calısır ve tek bir boolean ifade verir
    #tek boolena ifade veridig icin de, bu sayede if clause kullanılabilir.
    if addiction_df.isna().any().any():
        print("Warning: Missing values detected in the dataset. Filling with mean...")
        addiction_df = addiction_df.fillna(addiction_df.mean(numeric_only=True))
    
    df_clustered, _ = standardization_process(path, n_clusters=best_k)
    addiction_df['Cluster'] = df_clustered['Cluster']

    features = [
        "Daily_Usage_Hours",
        "Phone_Checks_Per_Day",
        "Screen_Time_Before_Bed",
        "Time_on_Social_Media",
        "Sleep_Hours",
        "Exercise_Hours",
        "Time_on_Gaming",
        "Anxiety_Level",
        "Depression_Level",
        "Self_Esteem",
        "Family_Communication",
        "Social_Interactions",
    ]

    print("F-test results:\n")
    
    for feat in features:
        clusters_data = [addiction_df[addiction_df['Cluster'] == i][feat] for i in range(best_k)]
        #clusters_data şu an liste halindedir, eğer bunu atarsak f_oneway 1 parametre alır ve yanlış olur (nested list)
        '''
        [
          [10, 15, 12, 8, 9],      # Cluster 0'daki telefon kontrolleri  
          [45, 50, 48, 52, 44],    # Cluster 1'deki telefon kontrolleri
          [80, 85, 90, 88, 92]     # Cluster 2'deki telefon kontrolleri
        ]
        '''
        #* operatörü listeyi "unpack" eder:
        # f_oneway(*clsuters_data) ile Fonksiyon şunu görür: ([1,2,3], [4,5,6], [7,8,9])
        # → 3 parametre (her biri ayrı grup) - DOĞRU!
        f_stat, p_val = f_oneway(*clusters_data)
        print(f"{feat:25}: F={f_stat:.2f}, p={p_val:.2e}\n")