import pandas as pd

from ml.model import random_forest_with_oversampling
from utils.data_loader import load_data 
from utils.plots import general_plotting, cluster_plots, classification_plots
from utils.stats import general_stats
from ml.analysis import standardization_info, f_test
from ml.clustering import clustering_by_phone_checks_for_elbow, clustering_by_all

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN

import os
import plotly.io as pio
pio.renderers.default = "browser"

def main():
    
    #Clustering: "Veri nasıl gruplanıyor?" (Keşifsel), Classification: "X durumunda Y sonucu ne olacak?" (Tahmin)
    '''
    Clustering: "Bu sınıfta 3 arkadaş grubu var" (sosyal grup) 
    (Unsupervised yani hedef degisken yok, algoritma kmeans, değerlendirme wcss ve silhouette )
    
    Classification: "Bu sınıfta matematik başarılı/başarısız 2 grup var" (akademik durum) 
    (supervised yani hedef değişken var (matematik notu), değerlendirme accuracy, f1-score, precision, recall)

    '''

    path = r"C:\Users\User\Desktop\lectures\teen_phone_addiction\data\teen_phone_addiction_dataset.csv"
    print(f"File exists: {os.path.exists(path)}")
    df = load_data(path)
    
    if df is not None:
        general_stats(df)
        general_plotting(df)

        all_df, k_values, wcss, silhouette_scores, best_k = clustering_by_all(path, range(2,10))
        best_k = k_values[silhouette_scores.index(max(silhouette_scores))]
        print(f"Optimal k: {best_k}")
        
        standardization_info(path, best_k)
        f_test(path, best_k)



        x = df[[
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
            
        ]]
        #print(f"Min: {df['Addiction_Level'].min()}")
        #print(f"Max: {df['Addiction_Level'].max()}")
        #print(f"Unique values: {sorted(df['Addiction_Level'].unique())}")
        
        low = df[df['Addiction_Level'] <= 4.0].shape[0]
        medium = df[(df['Addiction_Level'] > 4.0) & (df['Addiction_Level'] <= 7.0)].shape[0]  # 4.0 < x <= 7.0
        high = df[df['Addiction_Level'] > 7.0].shape[0]
        
        print(f"Low (1-4): {low}")
        print(f"Medium (4-7): {medium}")
        print(f"High (7-10): {high}")


        y = df['Addiction_Level'].apply(lambda level: 0 if level <= 4.0 else (1 if level <= 7.0 else 2))

        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
        '''
        x = metrikler, y = hedef değişken
            x: DataFrame veya numpy array → bağımsız değişkenler
            y: Series veya array → bağımlı değişken
        
        train_test_split fonksiyonu bu veriyi ikiye ayırıyor:
            Training set (X_train, y_train):  Modeli eğitmek için kullanılır (%80).
            Test set (X_test, y_test):  Modelin hiç görmediği veriler (%20).
        
        test_size=0.2 -> Toplam 1000 satır varsa → 800 train, 200 test.
        random_state=42 olduğu için hep aynı  200 satır test setine gider.
        '''
        
        smote = SMOTE(sampling_strategy={0: 200, 1: 400}, random_state=42, k_neighbors=5)
        smoteenn = SMOTEENN(sampling_strategy={0: 200, 1: 400}, random_state=42)
        adasyn = ADASYN(sampling_strategy='minority', random_state=42, n_neighbors=3) 
        
        model_smote, fi_smote, _ = random_forest_with_oversampling(
            X_train, X_test, y_train, y_test,
            oversampler=smote,
            method_name="SMOTE"
        )

        model_smoteenn, fi_smoteenn, _= random_forest_with_oversampling(
            X_train, X_test, y_train, y_test,
            oversampler=smoteenn,
            method_name="SMOTEENN"
        )

        model_adasyn, fi_adasyn, _, = random_forest_with_oversampling(
            X_train, X_test, y_train, y_test,
            oversampler=adasyn,
            method_name="ADASYN"
        )
        #model otomatik olarak print edilir ama feature_importance için ayrıca print edilmesi gerekiyor.

        print("\nFeature Importance:")
        print(fi_smote)
        print("\n")
        print(fi_smoteenn)
        print("\n")
        print(fi_adasyn)

        #cluster_plots(path)
        classification_plots(X_train, X_test, y_train, y_test, "SMOTE")
        classification_plots(X_train, X_test, y_train, y_test, "SMOTEENN")
        classification_plots(X_train, X_test, y_train, y_test, "ADASYN")

        
        
if __name__ == "__main__":
    main() 