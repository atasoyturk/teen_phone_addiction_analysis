import os
import pandas as pd

from utils.data_loader import load_data
from utils.plots import general_plotting
from utils.stats import general_stats
from utils.plots import cluster_plots

from ml.clustering import clustering_by_all, standardization_process
from ml.analysis import standardization_info, f_test, analyze_and_plot_results
from ml.training_workflow import train_models_and_get_importance

from sklearn.model_selection import train_test_split

import plotly.io as pio
pio.renderers.default = "browser"


def main():
    
    path = r"C:\Users\User\Desktop\lectures\staj\machine_learning\teen_phone_addiction\data\teen_phone_addiction_dataset.csv"
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
    
    features = [
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
    
    _, k_values, wcss, silhouette_scores, best_k, df_pca = clustering_by_all(path, features, range(2, 10))
    best_k = best_k
    df_clustered, cluster_means = standardization_process(path, features, n_clusters=best_k)
    cluster_plots(path, features, k_values, wcss, silhouette_scores, best_k, df_clustered)
    print(f"Optimal k would be : {best_k}")

    standardization_info(df_clustered, cluster_means)
    f_test(path, features, best_k, df_clustered)

    x = df[features]
    y_str = df['Addiction_Level'].apply(
    lambda level: 'Low Risk' if level <= 4.0 else
                  ('Moderate Risk' if level <= 7.0 else 'High Addiction')
    )

    low = (y_str == 'Low Risk').sum()
    medium = (y_str == 'Moderate Risk').sum()
    high = (y_str == 'High Addiction').sum()

    print(f"\nOriginal Addiction Level Distribution (Manual Binning):")
    print(f"  Low Risk (1-4): {low}")
    print(f"  Moderate Risk (4-7): {medium}")
    print(f"  High Addiction (7-10): {high}")

    # Map string labels to numeric for model training
    label_mapping = {'Low Risk': 0, 'Moderate Risk': 1, 'High Addiction': 2}
    y_numeric = y_str.map(label_mapping)

    print(f"\nBehavioral Cluster Distribution (PCA + KMeans):")
    print(df_pca['Cluster'].value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        x, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
    )

    results = train_models_and_get_importance(X_train, X_test, y_train, y_test)

    analyze_and_plot_results(results, X_train, X_test, y_str.iloc[X_train.index], y_str.iloc[X_test.index], features)

    print("\nTeens Phone Addiction Analysis is done.")


if __name__ == "__main__":
    main()