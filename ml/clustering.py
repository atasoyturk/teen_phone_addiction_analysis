
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
from ml.preprocessing import normalize_features, apply_pca



def clustering_by_all(path, k_range):

    # Tüm özelliklerle Elbow ve Silhouette analizi yapar.
    #clusteringden once her zaman ölçeklendirme yapılmalıdır.
    
    addiction_df = addiction_df_create(path).reset_index(drop=True)
    addiction_df = normalize_features(addiction_df) 
    #Phone_Checks_per_Day ve Depression_Level cok baskınlık saglıyor, log transform ile outlier'şarı bastırıoruz
    
    if addiction_df.isna().any().any():
        print("Warning: Missing values detected in the dataset. Filling with mean...")
        addiction_df = addiction_df.fillna(addiction_df.mean(numeric_only=True))
        #eğer boş hucre varsa, ortalama ile doldurur.   
        #numeric_only=True, eğer boş hucre string bir ifade ise doldurmaz orayı
    
    '''
    addiction_df.isna()
    Her hücrede eksik değer (NaN) varsa True, yoksa False döner.
        A      B      C
    0  False  False  False
    1  False   True  False
    2  False  False   True
    
    İlk .any() (sütun bazında)
    A    False
    B     True
    C     True
    dtype: bool
    
     İkinci .any() (genel kontrol)
    Şimdi bu Series'i alır ve "bunların arasında en az bir True var mı?" 
    False OR True OR True = True
    
    '''

    df_pca, pca_model, scaler = apply_pca(addiction_df, n_components=2)

    k_values = list(k_range)
    wcss = []
    silhouette_scores = []
    all_labels = {}  # her k için label sakla


    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df_pca) 
        #fit_predict ile hem df'e kmeams uygular hem de etiketleri tahmin eder.

        wcss.append(kmeans.inertia_)
        # inertia_: toplam kare uzaklık (WCSS)
        
        if k == 1:
            score = 0
        else:
            score = silhouette_score(df_pca, kmeans.labels_)

        silhouette_scores.append(score)
        all_labels[k] = labels  # her k için sakla

    best_k = k_values[silhouette_scores.index(max(silhouette_scores))]
    #bir arraydaki max elemanın indeksi, array.index(max(array))
    addiction_df["Cluster"] = all_labels[best_k]

    return addiction_df,k_values, wcss, silhouette_scores, best_k



def standardization_process(path, n_clusters=3):
    
    addiction_df = addiction_df_create(path)
    addiction_df = normalize_features(addiction_df)
    
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




def clustering_by_dailyusage_for_elbow(path, k_range):
    
    addiction_df = addiction_df_create(path)
    addiction_df = normalize_features(addiction_df)

    data = addiction_df[['Daily_Usage_Hours']].copy()
    data.dropna(inplace=True)
    #Sadece daily usage hoursa bakarak optimal k bakıyoruz

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
    Tüm ilgili kolonları (Daily_Usage_Hours, Phone_Checks_Per_Day,
    Screen_Time_Before_Bed, Time_on_Social_Media) ve Cluster etiketini döndürür.
    """
    addiction_df = addiction_df_create(path)

    # Sadece Phone_Checks_Per_Day ile KMeans uygula
    phone_checks = addiction_df[['Phone_Checks_Per_Day']].copy()
    phone_checks.dropna(inplace=True)

    scaler = StandardScaler()
    phone_checks_scaled = scaler.fit_transform(phone_checks)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(phone_checks_scaled)

    sil_score = silhouette_score(phone_checks_scaled, labels)

    # Cluster kolonunu orijinal dataframe'e ekle
    by_phonechecks_result_df = addiction_df[['Daily_Usage_Hours',
                                             'Phone_Checks_Per_Day',
                                             'Screen_Time_Before_Bed',
                                             'Time_on_Social_Media']].copy()
    by_phonechecks_result_df['Cluster'] = labels

    print(f"Clustering is applied only by Phone_Checks_Per_Day.")
    print(f"Silhouette Score: {sil_score:.3f}")

    # ✅ Sadece clusterlı dataframe döndür
    return by_phonechecks_result_df


