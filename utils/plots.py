import pandas as pd
import plotly.express as px
import numpy as np


from plotly import graph_objects as go
from plotly.subplots import make_subplots
from ml.clustering import clustering_by_all, standardization_process
from ml.model import random_forest_with_oversampling
from utils.data_loader import load_data
from ml.preprocessing import normalize_features, addiction_df_create

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN
from shap import TreeExplainer




def general_plotting(df):
    
    if df is not None and 'Exercise_Hours' in df.columns:
        
        fig = px.scatter(
            df,
            x='Exercise_Hours',
            y='Time_on_Gaming',
            title='Exercise Hours vs Gaming Time',
            labels={
                'Exercise_Hours': 'Hours of Exercise',
                'Time_on_Gaming': 'Hours Spent on Gaming'
            },
            custom_data=['Gender', 'Age']
        )

        fig.update_traces(
            marker=dict(
                color=df['Gender'].map({'Male': 'blue', 'Female': 'pink', 'Other': 'green'}),
                size=10,
                sizemode='area',
                opacity=0.6,
                line=dict(width=1, color='black')
            ),
            hovertemplate=(
                '<b>Gender:</b> %{customdata[0]}<br>'
                '<b>Age:</b> %{customdata[1]}<br>'
                '<b>Exercise Hours:</b> %{x}<br>'
                '<b>Gaming Hours:</b> %{y}<br>'
            )
        )

        fig.update_layout(
            xaxis_title='Hours of Exercise',
            yaxis_title='Hours Spent on Gaming',
            xaxis=dict(showgrid=True, zeroline=True, showline=True, ticks='outside',
                       ticklen=5, tickwidth=2, tickcolor='black', gridcolor='lightgray'),
            yaxis=dict(showgrid=True, zeroline=True, showline=True, ticks='outside',
                       ticklen=5, tickwidth=2, tickcolor='black', gridcolor='lightgray'),
            margin=dict(l=40, r=40, t=80, b=40),
            legend={'title': 'Gender', 'orientation': 'h', 'x': 0.5, 'xanchor': 'center', 'y': 1.05, 'yanchor': 'top'}
        )
        fig.show()


def cluster_plots(path):
    
    #1. Tüm değişkenlerle Elbow & Silhouette
    _, k_values, wcss, silhouette_scores, best_k = clustering_by_all(path, range(2, 10))
    df1 = pd.DataFrame({'K': k_values, 'WCSS': wcss, 'Silhouette': silhouette_scores})
    
    fig_all = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Elbow Method (WCSS)", "Silhouette Analysis"),
        shared_xaxes=False,
        horizontal_spacing=0.1
    )

    fig_all.add_trace(go.Scatter(x=df1['K'], y=df1['WCSS'], mode='lines+markers', name='WCSS', line=dict(color='blue')), row=1, col=1)
    fig_all.add_trace(go.Scatter(x=df1['K'], y=df1['Silhouette'], mode='lines+markers', name='Silhouette', line=dict(color='red')), row=1, col=2)

    fig_all.update_yaxes(title_text="WCSS", row=1, col=1)
    fig_all.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    fig_all.update_xaxes(title_text="K", row=1, col=1)
    fig_all.update_xaxes(title_text="K", row=1, col=2)

    fig_all.update_layout(
        title_text="Optimal K Analysis: All Metrics",
        title_x=0.5,
        hovermode="x unified",
        height=500,
        template="plotly_white"
    )
    fig_all.show()


    # 3. Kutu grafikleri: Tüm özellikler
    addiction_df = addiction_df_create(path)    
    df_clustered, _ = standardization_process(path, best_k)
    addiction_df['Cluster'] = df_clustered['Cluster']
    
    addiction_df = normalize_features(addiction_df)
    
    cluster_names = {0: 'Low Risk', 1: 'Normal', 2: 'High Addiction'}
    cluster_colors = {
        'Low Risk': '#1f77b4',      # Mavi
        'Normal': '#2ca02c',        # Yeşil
        'High Addiction': '#d62728' # Kırmızı
    }
    
    addiction_df['Cluster_Label'] = addiction_df['Cluster'].map(cluster_names)

    melted = pd.melt(
        addiction_df,
        id_vars=['Cluster_Label'],
        
        value_vars=[
            # Usage features
            "Daily_Usage_Hours",
            "Phone_Checks_Per_Day",
            "Screen_Time_Before_Bed",
            "Time_on_Social_Media",
            "Sleep_Hours",
            "Exercise_Hours",
            "Time_on_Gaming",
            
            # Psychological / social features
            "Anxiety_Level",
            "Depression_Level",
            "Self_Esteem",
            "Family_Communication",
            "Social_Interactions",
            
        ],
        
        var_name='Feature',
        value_name='Value'
    )
    
    fig_box_all = px.box(
        melted,
        x='Feature',
        y='Value',
        color='Cluster_Label',
        title='Cluster Characters',
        hover_data=['Value']
    )
    fig_box_all.update_layout(
        xaxis_tickangle=45,
        legend_title_text='Cluster',
        title_x=0.5,
        legend=dict(
            title="User Group",
            itemsizing='constant',
            traceorder='normal'
        )
    )
    fig_box_all.show()
    
    

def classification_plots(X_train, X_test, y_train, y_test, method_name=""):
    if method_name == "SMOTE":
        oversampler = SMOTE(
            sampling_strategy={0: 200, 1: 400},
            random_state=42,
            k_neighbors=5
        )
    elif method_name == "SMOTEENN":
        oversampler = SMOTEENN(
            sampling_strategy={0: 200, 1: 400},
            random_state=42,
        )
    elif method_name == "ADASYN":
        oversampler = ADASYN(
            sampling_strategy='minority',
            random_state=42,
            n_neighbors=3
        )
    else:
        raise ValueError("method must be 'SMOTE', 'SMOTEENN', or 'ADASYN'")

    _, feature_importance, shap_magnitude = random_forest_with_oversampling(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        oversampler=oversampler,
        method_name=method_name,
        n_estimators=100,
        max_depth=10
    )

    # Feature importance grafiği
    fig_oversampling = px.bar(
        feature_importance.sort_values('Importance', ascending=True),
        x='Importance',
        y='Feature',
        title=f'Feature Importance ({method_name})',
        orientation='h',
        labels={'Importance': 'Importance', 'Feature': 'Feature'},
        height=600,
        color = 'Importance',
        color_continuous_scale='Blues'
    )
    fig_oversampling.show()
    
    
    shap_importance_df = pd.DataFrame({
        'Feature': shap_magnitude.index,
        'Average |SHAP|': shap_magnitude.values
    }).sort_values('Average |SHAP|', ascending= True)

    
    # SHAP grafiği
    fig_shap = px.bar(
        shap_importance_df,
        x='Average |SHAP|',
        y='Feature',
        orientation='h',
        title=f"SHAP Feature Importance (Average {method_name})",
        labels={'Average |SHAP|': 'Average |SHAP Values|', 'Feature': 'Feature'},
        color='Average |SHAP|',
        color_continuous_scale='Blues'
    )
    fig_shap.update_layout(yaxis={'categoryorder':'total ascending'})
    fig_shap.show()
    
