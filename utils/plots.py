import pandas as pd
import plotly.express as px
import numpy as np


from plotly import graph_objects as go
from plotly.subplots import make_subplots
from ml.clustering import clustering_by_all, standardization_process
from ml.model import random_forest_with_oversampling, random_forest_cost_sensitive
from utils.data_loader import load_data
from ml.preprocessing import  addiction_df_create

from sklearn.preprocessing import StandardScaler, MinMaxScaler
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


def cluster_plots(path, features, k_values, wcss, silhouette_scores, best_k, df_clustered):

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
    addiction_df = addiction_df_create(path, features)
    addiction_df['Cluster'] = df_clustered['Cluster']

    #addiction_df = normalize_features(addiction_df)

    cluster_names = {0: 'Low Risk', 1: 'Normal', 2: 'High Addiction'}
    cluster_colors = {
        'Low Risk': '#1f77b4',
        'Normal': '#2ca02c',
        'High Addiction': '#d62728'
    }

    addiction_df['Cluster_Label'] = addiction_df['Cluster'].map(cluster_names)

    scaler = MinMaxScaler()
    addiction_df_normalized = addiction_df.copy()
    addiction_df_normalized[features] = scaler.fit_transform(addiction_df[features])

    melted = pd.melt(
        addiction_df_normalized,
        id_vars=['Cluster_Label'],
        value_vars=features,
        var_name='Feature',
        value_name='Normalized Value'
    )

    fig_box_all = px.box(
        melted,
        x='Feature',
        y='Normalized Value',
        color='Cluster_Label',
        title='Cluster Characteristics (Normalized 0-1)',
        hover_data=['Normalized Value']
    )
    fig_box_all.update_layout(
        xaxis_tickangle=45,
        legend_title_text='User Group',
        title_x=0.5,
        yaxis_title='Normalized Value (0-1)',

        legend=dict(
            title="User Group",
            itemsizing='constant',
            traceorder='normal'
        )
    )
    fig_box_all.show()
    
    

def classification_plots(X_train, X_test, y_train, y_test, method_name="", feature_importance=None, shap_magnitude=None):

    # Feature importance graph
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

    #shap_magnitude may be form of pandas series or numpy array depending on python version
    #I have used shap_df.abs().mean(axis=0) in model.py to get shap_magnitude as pandas series but it may not work in all versions
    if hasattr(shap_magnitude, 'index'):

        shap_importance_df = pd.DataFrame({
            'Feature': shap_magnitude.index,
            'Average |SHAP|': shap_magnitude.values
        }).sort_values('Average |SHAP|', ascending=True)
    else:

        shap_importance_df = pd.DataFrame({
            'Feature': feature_importance['Feature'].values,
            'Average |SHAP|': shap_magnitude
        }).sort_values('Average |SHAP|', ascending=True)

    # SHAP graph
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
    
