import os
import pandas as pd

from utils.data_loader import load_data
from utils.plots import general_plotting
from utils.stats import general_stats

from ml.clustering import clustering_by_all
from ml.analysis import standardization_info, f_test, analyze_and_plot_results
from ml.training_workflow import train_models_and_get_importance

from sklearn.model_selection import train_test_split

import plotly.io as pio
pio.renderers.default = "browser"


def main():
    
    path = r"C:\Users\User\Desktop\lectures\teen_phone_addiction\data\teen_phone_addiction_dataset.csv"
    print(f"File exists: {os.path.exists(path)}")
    df = load_data(path)
    
    if df is None:
        print("Dataset could not be loaded")
        return

    print("\nGeneral Stats and Visualizations")
    print("=" * 60)
    general_stats(df)
    general_plotting(df)

    print("\nClustering Analysis")
    print("=" * 60)
    _, _, _, silhouette_scores, best_k = clustering_by_all(path, range(2, 10))
    best_k = best_k  
    print(f"Optimal k would be : {best_k}")

    standardization_info(path, best_k)
    f_test(path, best_k)

    feature_columns = [
        # Usage Metrics
        "Daily_Usage_Hours",
        "Phone_Checks_Per_Day",
        "Screen_Time_Before_Bed",
        "Time_on_Social_Media",
        "Sleep_Hours",
        "Exercise_Hours",
        "Time_on_Gaming",
        # Psicological metrics
        "Anxiety_Level",
        "Depression_Level",
        "Self_Esteem",
        "Family_Communication",
        "Social_Interactions",
    ]

    x = df[feature_columns]
    y = df['Addiction_Level'].apply(lambda level: 0 if level <= 4.0 else (1 if level <= 7.0 else 2))

    low = (y == 0).sum()
    medium = (y == 1).sum()
    high = (y == 2).sum()
    print(f"\nCluster Distribution:")
    print(f"  Low (1-4): {low}")
    print(f"  Medium (4-7): {medium}")
    print(f"  High (7-10): {high}")

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    results = train_models_and_get_importance(X_train, X_test, y_train, y_test)

    analyze_and_plot_results(results, X_train, X_test, y_train, y_test)

    print("\nTeens Phone Addiction Analysis is done.")


if __name__ == "__main__":
    main()