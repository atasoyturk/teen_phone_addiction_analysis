from utils.data_loader import load_data
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def addiction_df_create(path):
    
    df = load_data(path)
    features = [
        # Usage features
        "Daily_Usage_Hours",
        "Phone_Checks_Per_Day",
        "Screen_Time_Before_Bed",
        "Time_on_Social_Media",
        "Sleep_Hours",
        "Exercise_Hours",
        "Time_on_Gaming",
        
        # Psychological / social features
        "Anxiety_Level",
        "Depression_Level",
        "Self_Esteem",
        "Family_Communication",
        "Social_Interactions",
        
    ]

    addiction_df = df[features].copy()
    addiction_df.dropna(inplace=True)
    
    return addiction_df



def normalize_features(df):
    df = df.copy()
    
    # log transform (çok büyük outlier’ları bastırır)
    #bu log(1 + x) anlamına gelir.
    #Sağa çarpık (right-skewed) verileri daha normal dağılıma yaklaştırır
    #Çok yüksek değerler (outliers) daha makul seviyelere çekilir, Örnek: 1000 kontrol → log(1001) ≈ 6.9
    #np.logp ile daha guvenli sonuclar --> np.logp(0) = log(1+0) = 0 (eger ki np.log olsaydı hata verirdi)
    
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
    
    '''
    PCA (Principal Component Analysis)
    PCA, boyut azaltma ve veri sıkıştırma tekniğidir. Yüksek boyutlu veriyi daha az boyutta, ama en önemli bilgileri koruyarak temsil eder.
    '''
    
    #PCA varyansa dayalı çalıştığı için büyük değerler baskın olur. O yüzden scale edilmelidir.
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    '''
    Principal Components (Ana Bileşenler)

    PC1: Verinin en çok değiştiği yön (en fazla bilgi)
    PC2: PC1'e dik, ikinci en çok bilgi
    PC3, PC4... devam eder
    '''
    # PCA uygulama
    pca = PCA(n_components=n_components, random_state=42)
    df_pca = pca.fit_transform(df_scaled)
    #fit() : Eigenvalue ve eigenvector'leri bulur, Açıklanan varyans oranlarını hesaplar, Dönüşüm matrisini oluşturur
    #transform(): Orijinal veriyi yeni koordinat sistemine dönüştürür, Principal Component skorlarını hesaplar
    #Yani fit_transform() aslında PCA'nın "beyninini eğitip", sonra o eğitilmiş beyinle veriyi dönüştürme işlemidir.

    #print(pca.explained_variance_ratio_.sum())
    # yüzde kaç bilgi korunmuş

    # PCA sonuçlarını dataframe olarak döndür
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(df_pca, columns=pca_columns, index=df.index)

    return df_pca, pca, scaler


