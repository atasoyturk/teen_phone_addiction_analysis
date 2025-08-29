from utils.data_loader import load_data

def addiction_df_create(path):
    
    df = load_data(path)
    features = ['Daily_Usage_Hours', 'Phone_Checks_Per_Day',
                'Screen_Time_Before_Bed', 'Time_on_Social_Media']
    addiction_df = df[features].copy()
    addiction_df.dropna(inplace=True)
    
    return addiction_df