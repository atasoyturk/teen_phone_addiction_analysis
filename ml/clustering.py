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

from ml.preprocessing import addiction_df_create



def standardization_process(path):
   
    addiction_df = addiction_df_create(path)
    
    scaler = StandardScaler()
    addiction_df_scaled = scaler.fit_transform(addiction_df)
    #StandardScaler classından bir nesne olustrduk 
    #Bu nesne, ileride veriyi nasıl dönüştüreceğini "öğrenmek" için kullanılacak.
    '''
    .fit_transform() = .fit() + .transform()
    scaler.fit() her sütun icin ortalama ve standart sapma hesaplar
    scaler.transform() ise fit() ile hesaplanan her değeri : (değer- ortalama)/ std. sapma değerine çevirir.
    (Z-Score'a cevirir yani her değeri )
    
    '''
    
    kmeans = KMeans(n_clusters=2, random_state=42) 

    #yeni bir kmeans modeli olusturuyoruz, n_clusters kume sayısını ifade eder
    #KMeans() bir model nesnesi oluşturur, henüz veri üzerinde çalışmaz.
    '''
    Başlangıcta secilen merkezler (centroid) rastegele oldugu icin birden fazla model olsuutruldugunda 
    eğer random_state kullanılmazsa her seferinde farklı merkezler seçilir bu da farklı kümelenmelerin cıkmasına neden olur.
    random_state =42 olmasınn onemi yok, onemli olan her seferinde aynı random_state değerini kullanmaktır.
    
    '''
    
    labels = kmeans.fit_predict(addiction_df_scaled) 
    #

    addiction_df = addiction_df.copy()
    addiction_df['Cluster'] = labels
    #addiction_df dataframeine Cluster kolonunu ekledik artık her öğrenci 0 ya da 1 değerine sahip.
    
    cluster_means = addiction_df.groupby('Cluster')[addiction_df.columns[:-1]].mean()
    #addiction_df.columns[:-1] son kolon haric tüm kolonları ifade eder (Cluster kolonu haric).  ([:-n] son n kolon haric tümü)
    #Cluster'e gore gruplayıp her Cluster icin, Cluster kolonu haric tüm kolonlarınıın ortalamaısnı alır
    
    return addiction_df, cluster_means