import pandas as pd
from utils.data_loader import load_data 
from utils.plots import general_plotting
from utils.stats import general_stats
from ml.standardization import standardization_info, f_test

def main():
    
    path = r"C:\Users\User\Desktop\lectures\teen_phone_addiction\teen_phone_addiction_dataset.csv"
    df = load_data(path)
    df.columns = df.columns.str.strip()  # tum kolonlardaki boslukları temizler
    
    if df is not None:
        #general_stats(df)
        #general_plotting(df)
        standardization_info(path)
        f_test(path)
        
    
        
    
if __name__ == "__main__":
    main() 