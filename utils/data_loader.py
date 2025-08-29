import pandas as pd

def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    

df = load_data(r"C:\Users\User\Desktop\lectures\teen_phone_addiction\data\teen_phone_addiction_dataset.csv")
print(f"Data loaded successfully: {df is not None}")


