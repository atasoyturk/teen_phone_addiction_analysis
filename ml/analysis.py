from scipy.stats import f_oneway
import pandas as pd
from ml.preprocessing import  apply_pca, addiction_df_create
from ml.clustering import standardization_process
from utils.plots import classification_plots, cluster_plots



def standardization_info(df_clustered, cluster_means):
    print("Cluster Means:\n", cluster_means)

    print("\nCluster Distribution:")
    distribution = df_clustered['Cluster'].value_counts().sort_index()
    print(f"\n{distribution}")

    if (distribution < 2).any():
        print("Warning: One or more clusters have fewer than 2 samples, which may cause issues in F-test.")

    
def f_test(path, features, best_k, df_clustered):

    addiction_df = addiction_df_create(path, features)
    #addiction_df = normalize_features(addiction_df)

    if addiction_df.isna().any().any():
        print("Warning: Missing values detected in the dataset. Filling with mean...")
        addiction_df = addiction_df.fillna(addiction_df.mean(numeric_only=True))

    addiction_df['Cluster'] = df_clustered['Cluster']

    features = [
        "Daily_Usage_Hours",
        "Phone_Checks_Per_Day",
        "Screen_Time_Before_Bed",
        "Time_on_Social_Media",
        "Sleep_Hours",
        "Exercise_Hours",
        "Time_on_Gaming",
        "Anxiety_Level",
        "Depression_Level",
        "Self_Esteem",
        "Family_Communication",
        "Social_Interactions",
    ]

    print("F-test results:\n")

    for feat in features:
        clusters_data = [addiction_df[addiction_df['Cluster'] == i][feat] for i in range(best_k)]

        f_stat, p_val = f_oneway(*clusters_data)
        print(f"{feat:25}: F={f_stat:.2f}, p={p_val:.2e}\n")
        

def analyze_and_plot_results(results, X_train, X_test, y_train, y_test, features):

    path = r"C:\Users\User\Desktop\lectures\staj\machine_learning\teen_phone_addiction\data\teen_phone_addiction_dataset.csv"

    for name, res in results.items():
        fi_df = res['feature_importance']
        fi_values = pd.Series(fi_df['Importance'].values, index=fi_df['Feature'])

        shap_magnitude = res['shap_magnitude']
        shap_series = pd.Series(shap_magnitude, index=fi_df['Feature'])

        correlation = fi_values.corr(shap_series)
        print(f"\n{name} - Correlation (FI (Feature Importance)-SHAP Magnitude): {correlation:.4f}")

        classification_plots(X_train, X_test, y_train, y_test, method_name=name, feature_importance=fi_df, shap_magnitude=shap_magnitude)
    
