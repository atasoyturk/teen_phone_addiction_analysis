import pandas as pd
from sklearn.model_selection import train_test_split
from ml.model import run_random_forest
from utils.data_loader import load_data 
from utils.plots import general_plotting, cluster_plots
from utils.stats import general_stats
from ml.analysis import standardization_info, f_test
import os

def main():
    
    path = r"C:\Users\User\Desktop\lectures\teen_phone_addiction\data\teen_phone_addiction_dataset.csv"
    print(f"File exists: {os.path.exists(path)}")
    df = load_data(path)
    
    if df is not None:
        #general_stats(df)
        #general_plotting(df)
        #standardization_info(path)
        #f_test(path)
        
        #cluster_plots(path)
        
        x = df[['Daily_Usage_Hours', 'Phone_Checks_Per_Day', 'Screen_Time_Before_Bed', 'Time_on_Social_Media']]
        y = df['Cluster']
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
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
        model, feature_importance = run_random_forest(X_train, X_test, y_train, y_test)
        
        print("\nFeature Importance:")
        print(feature_importance)


        
if __name__ == "__main__":
    main() 