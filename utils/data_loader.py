import pandas as pd

path = r"C:\Users\User\Desktop\lectures\staj\machine_learning\teen_phone_addiction\data\teen_phone_addiction_dataset.csv"

def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    

df = load_data(path)
print(f"Data loaded successfully: {df is not None}")


