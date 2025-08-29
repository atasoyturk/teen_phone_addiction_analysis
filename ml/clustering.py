
from sklearn.cluster import KMeans 
#KMeans, sklearn içinde kümeleme (clustering) algoritmalarının bulunduğu modüldür.
#Bu modülde yer alan, K-Ortalamalar (K-Means) adlı kümeleme algoritmasıdır.
#k-means algoritması yalnızca sayısal (numerik) veriler üzerinde çalışır.
'''
    Amaç, veri noktalarını belirli sayıda (K sayısı kadar) kümelere ayırarak 
her küme içindeki verilerin birbirine mümkün olduğunca benzer, farklı kümelerdeki verilerin ise birbirinden farklı olmasını sağlamaktır.
Küme sayısı (K) önceden belirlenmelidir.
Uzaklık ölçümü genellikle Öklid uzaklığı kullanılır.
'''

'''
K-Means Nasıl Çalışır? (Adım Adım)

1-K değerini seç: Kaç küme oluşturmak istediğinizi belirleyin (örneğin K=3).
2-Başlangıç merkezleri seç: Rastgele K adet merkez (centroid) seç.
3-Veri noktalarını kümelerine ata: Her veri noktasını en yakın merkeze göre bir kümeye ata.
4-Merkezleri güncelle: Her kümenin merkezini (ortalamasını) hesapla.
5-Adım 3 ve 4'ü tekrar et: Merkezler sabitlenene kadar (yani değişmeyene kadar) devam et.
6-Sonuç: Her veri noktası bir kümeye atanmış olur.
'''

from sklearn.metrics import silhouette_score
#Bir noktanın kendi kümesine ne kadar iyi ait olduğunu ve diğer kümelerden ne kadar farklı olduğunu ölçer.
'''
Her veri noktasının:
Kendi kümesindeki diğer noktalara olan ortalama uzaklık (a: cohesion),
En yakın başka kümeye olan ortalama uzaklık (b: separation)
s = (b-a)/max(a,b)

+1'e yakın: Mükemmel kümeleme (nokta doğru kümede)
0'a yakın: Kümeler arası sınır belirsiz
-1'e yakın: Nokta yanlış kümede olabilir
'''

'''
Elbow Method ile Bağlantısı: Her iki yöntem de optimal küme sayısını bulmak için kullanılır, ancak farklı yaklaşımlar

Elbow Method:

Within-Cluster Sum of Squares (WCSS) kullanır: 
WCSS, bir küme içindeki noktaların, o kümenin merkezine (centroid) olan uzaklıklarının karelerinin toplamıdır.
k=1 ---> WCSS Çok yüksek, Tüm veri bir kümede → noktalar merkezden uzak
k=2 ---> WCSS Daha düşük, 2 kümeye bölünmüş, her küme daha yoğun
k=10 ---> WCSS Çok düşük, 10 kümeye bölünmüş, her küme çok küçük, merkeze yakın

Grafiğe baktığında, WCSS’in hızla düştüğü ama sonra "dirsek" yaptığı bir nokta olur.
Grafikteki "dirsek" noktasını arar
Ancak dirsek belirsizse, Silhouette Score ile desteklenmelidir.
'''
# İdeal yaklaşım her iki yöntemi birlikte kullanmaktır

from sklearn.preprocessing import StandardScaler
#StandardScaler, sklearn (scikit-learn) kütüphanesinin preprocessing modülünde yer alan bir sınıftır.
'''
    Verideki her bir özellik (feature) için z-score hesaplanır. 
Bu işlem sonucunda her ozelligin ortalaması 0 standart sapması 1 olur (Z-score Table)
Tüm veri aynı ölçekte (scale) olur.

Makine öğrenmesi algoritmaları (özellikle mesafe temelli olanlar: k-Means, k-NN, SVM, PCA, logistic regression) 
büyük değerlerle daha küçük değerleri aynı kabul etmez. aralığı fazla olan metrik mesafeyi daha fazla etkiler → Yanlış sonuç!
Standardizasyon ile her iki özelliği de aynı ölçeğe getiririz. Örn. gelirin 100.000 olması ile yaşın 65 olması "aynı ağırlıkta" değerlendirilir.
'''

import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

from ml.preprocessing import addiction_df_create


def clustering_by_all(path, k_range):

    # Tüm özelliklerle Elbow ve Silhouette analizi yapar.

    addiction_df = addiction_df_create(path)
    scaler = StandardScaler()
    addiction_df_scaled = scaler.fit_transform(addiction_df)
    '''
    scaler.fit() ile verinin ortalaması 0, standart sapması 1 olacak şekilde dönüştürülür.(z-score table)
    scaler.transform() ile verinin her bir özelliği (feature) için z-score hesaplanır.
    fit_transform() ile bu iki işlem tek adımda yapılır.
    '''

    k_values = list(k_range)
    wcss = []
    silhouette_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(addiction_df_scaled)
        #kmeans.fit() ile model veriye uygulanır.

        wcss.append(kmeans.inertia_)
        # inertia_: toplam kare uzaklık (WCSS)
        
        if k == 1:
            score = 0
        else:
            score = silhouette_score(addiction_df_scaled, kmeans.labels_)
        silhouette_scores.append(score)
        
    return k_values, wcss, silhouette_scores


def clustering_by_phone_checks_for_elbow(path, k_range):
    """
    Sadece Phone_Checks_Per_Day ile Elbow ve Silhouette analizi yapar.
    Bu fonksiyon, optimal K seçimi için kullanılır.
    """
    addiction_df = addiction_df_create(path)
    data = addiction_df[['Phone_Checks_Per_Day']].copy()
    data.dropna(inplace=True)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    k_values = list(k_range)
    wcss = []
    silhouette_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data_scaled)
        
        wcss.append(kmeans.inertia_)
        #kmeans.inertia_ ile toplam kare uzaklık (WCSS) hesaplanır.

        if k == 1:
            score = 0
        else:
            score = silhouette_score(data_scaled, kmeans.labels_)
        silhouette_scores.append(score)
        
    return k_values, wcss, silhouette_scores


def clustering_by_phone_checks(path, n_clusters=2, random_state=42):
    """
    Sadece Phone_Checks_Per_Day ile KMeans modeli kurar (final model).
    """
    addiction_df = addiction_df_create(path)
    by_phonechecks = addiction_df[['Phone_Checks_Per_Day']].copy()
    by_phonechecks.dropna(inplace=True)
    
    scaler = StandardScaler()
    by_phonechecks_scaled = scaler.fit_transform(by_phonechecks)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(by_phonechecks_scaled)
    #kmeans.fit() ile model veriye uygulanır. kmeans.predict() ile grup etiketleri tahmin edilir.
    #fit_predict() ile hem model eğitilir hem de etiketler tahmin edilir.

    
    sil_score = silhouette_score(by_phonechecks_scaled, labels)
    
    by_phonechecks_result_df = by_phonechecks.copy()
    by_phonechecks_result_df['Cluster'] = labels
    
    print(f"Clustering is applied only by Phone_Checks_Per_Day.")
    print(f"Silhouette Score: {sil_score:.3f}")
    
    return by_phonechecks_result_df, sil_score, kmeans, scaler


def standardization_process(path):
    """
    Tüm özelliklerle KMeans (K=2) uygular, kümelenmiş veri ve ortalama döner.
    """
    addiction_df = addiction_df_create(path)
    
    scaler = StandardScaler()
    addiction_df_scaled = scaler.fit_transform(addiction_df)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(addiction_df_scaled)
    
    addiction_df = addiction_df.copy()
    addiction_df['Cluster'] = labels
    
    cluster_means = addiction_df.groupby('Cluster')[addiction_df.columns[:-1]].mean()
    
    return addiction_df, cluster_means