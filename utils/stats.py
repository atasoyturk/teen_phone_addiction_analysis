import pandas as pd

def general_stats(df):
    if df is None or df.empty:
        print("Dataset is empty or is not exist")
        return
    
    print("General Stats: ")
    print(df.describe())
    print("\n")
    
    avg_usage = df['Daily_Usage_Hours'].mean()
    print(f" Average Daily Phone Usage: {avg_usage:.2f} hours")

    over_18 = df[df['Age'] > 18]
    under_18 = df[df['Age'] <= 18]
    total = len(df)

    print(f"\nAverage Usage over 18+ Teens: {over_18['Daily_Usage_Hours'].mean():.2f} hours")
    print(f"Average Usage under 18 Teens: {under_18['Daily_Usage_Hours'].mean():.2f} hours")
    
    if 'Gender' in df.columns:
        gender_counts = df['Gender'].value_counts()
        print(f"\nGender Distrubition:")
        for gender, count in gender_counts.items():
            print(f"   {gender}: {count} teens (%{count/total*100:.1f})")
        
    heavy_users = df[df['Daily_Usage_Hours'] > 5]
    print(f"\n6. Teens have phone usage over 5+ hours: {len(heavy_users)} teens (%{len(heavy_users)/total*100:.1f})")
    
    max_usage = df['Daily_Usage_Hours'].max()
    min_usage = df['Daily_Usage_Hours'].min()
    print(f"\nMax phone usage: {max_usage:.2f} hours")
    print(f"Min phone usage: {min_usage:.2f} hours")
    
    if 'Addiction_Level' in df.columns:
        avg_addiction = df['Addiction_Level'].mean()
        high_addiction = df[df['Addiction_Level'] >= 7]  
        print(f"\nAverage addiction level score: {avg_addiction:.2f}")
        print(f"High addiction level score (7+): {len(high_addiction)} teens (%{len(high_addiction)/total*100:.1f})")