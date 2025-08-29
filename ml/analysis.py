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


def standardization_info(path):
    
    addiction_df, cluster_means = standardization_process(path)
    print(cluster_means)
    
    print("Cluster Distribution:")
    print(addiction_df['Cluster'].value_counts().sort_index()) 
    #sort_index ile kişi sayısı fazla olan küme üstte yazılır
    
    
def f_test(path):
    
    addiction_df, _ = standardization_process(path)  # Sadece veri lazım
    
    features = ['Daily_Usage_Hours', 'Phone_Checks_Per_Day', 'Screen_Time_Before_Bed', 'Time_on_Social_Media']
    
    for feat in features:
        c0 = addiction_df[addiction_df['Cluster']==0][feat]
        c1 = addiction_df[addiction_df['Cluster']==1][feat]
        # c0 ve c1 birer series'dir (df degil)
        f_stat, p_val = f_oneway(c0, c1)
        print(f"{feat:25}: F={f_stat:8.2f}, p={p_val:.2e}")
