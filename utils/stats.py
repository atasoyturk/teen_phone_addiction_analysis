import pandas as pd

def general_stats (df):
    if df is not None:
        daily_usage = df['Daily_Usage_Hours'].value_counts().mean()
        print(f"Average daily phone usage: {daily_usage:.2f} hours")

        over_18 = df[df['Age'] > 18].shape[0] #shape[0] satır sayısını verdiğinden, bu da 18 yaşından büyük kullanıcı sayısını verir
        print(f"\nNumber of users over 18: {over_18}")
        print(f"\nProportion of users over 18: {over_18 / df.shape[0] * 100:.2f}%")

        over_18_avg_usage = df[df['Age'] > 18]['Daily_Usage_Hours'].mean()
        print(f"\nAverage daily usage for users over 18: {over_18_avg_usage:.2f} hours")

        under_18 = df[df['Age'] < 18].shape[0]
        print(f"\nNumber of users under 18: {under_18}")
        print(f"\nProportion of users under 18: {under_18 / df.shape[0] * 100:.2f}%")

        under_18_avg_usage = df[df['Age'] < 18]['Daily_Usage_Hours'].mean()
        print(f"\nAverage daily usage for users under 18: {under_18_avg_usage:.2f} hours")

