# ml/plots.py

import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from ml.clustering import clustering_by_all, clustering_by_phone_checks_for_elbow, clustering_by_phone_checks, standardization_process
from utils.data_loader import load_data


def general_plotting(df):
    """
    Genel dağılım: Exercise vs Gaming
    """
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

        # Marker boyutu: yaşa göre
        size = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())
        size = size * 20 + 5  # 5-25 arası

        fig.update_traces(
            marker=dict(
                color=df['Gender'].map({'Male': 'blue', 'Female': 'pink', 'Other': 'green'}),
                size=size,
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
    
    # 1. Tüm değişkenlerle Elbow & Silhouette
    k_values, wcss, silhouette_scores = clustering_by_all(path, range(2, 10))
    df1 = pd.DataFrame({'K': k_values, 'WCSS': wcss, 'Silhouette': silhouette_scores})
    #verilerin ölçekleri birbirinden farklı olduğu için, WCSS ve Silhouette Score değerleri doğrudan karşılaştırılamaz.
    #bu yüzden mmelt yapmıyoruz.

    fig_all = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Elbow Method (WCSS)", "Silhouette Analysis"),
        shared_xaxes=False,
        horizontal_spacing=0.1
    )

    fig_all.add_trace(go.Scatter(x=df1['K'], y=df1['WCSS'], mode='lines+markers', name='WCSS', line=dict(color='blue')), row=1, col=1)
    fig_all.add_trace(go.Scatter(x=df1['K'], y=df1['Silhouette'], mode='lines+markers', name='Silhouette', line=dict(color='red')), row=1, col=2)

    fig_all.update_yaxes(title_text="WCSS", row=1, col=1)
    fig_all.update_yaxes(title_text="Silhouette Score", range=[0.10, 0.25], row=1, col=2)
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

    # 2. Sadece Phone_Checks_Per_Day ile Elbow & Silhouette
    print("2. Sadece Phone_Checks_Per_Day ile Elbow & Silhouette analizi...")
    k_vals_pc, wcss_pc, sil_pc = clustering_by_phone_checks_for_elbow(path, range(2, 10))
    df2 = pd.DataFrame({'K': k_vals_pc, 'WCSS': wcss_pc, 'Silhouette': sil_pc})

    fig_pc = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Elbow Method (WCSS)", "Silhouette Analysis"),
        shared_xaxes=False,
        horizontal_spacing=0.1
    )

    fig_pc.add_trace(go.Scatter(x=df2['K'], y=df2['WCSS'], mode='lines+markers', name='WCSS', line=dict(color='blue')), row=1, col=1)
    fig_pc.add_trace(go.Scatter(x=df2['K'], y=df2['Silhouette'], mode='lines+markers', name='Silhouette', line=dict(color='red')), row=1, col=2)

    fig_pc.update_yaxes(title_text="WCSS", row=1, col=1)
    fig_pc.update_yaxes(title_text="Silhouette Score", row=1, col=2)
    fig_pc.update_xaxes(title_text="K", row=1, col=1)
    fig_pc.update_xaxes(title_text="K", row=1, col=2)

    fig_pc.update_layout(
        title_text="Optimal K Analysis: Just Phone Checks per Day",
        title_x=0.5,
        hovermode="x unified",
        height=500,
        template="plotly_white"
    )
    fig_pc.show()

    # 3. Kutu grafikleri: Tüm özellikler
    print("3. Tüm özellikler için kutu grafiği...")
    addiction_df, _ = standardization_process(path)
    melted = pd.melt(
        addiction_df,
        id_vars=['Cluster'],
        value_vars=['Daily_Usage_Hours', 'Phone_Checks_Per_Day', 'Screen_Time_Before_Bed', 'Time_on_Social_Media'],
        var_name='Feature',
        value_name='Value'
    )
    
    fig_box_all = px.box(
        melted,
        x='Feature',
        y='Value',
        color='Cluster',
        facet_col='Feature',
        title='Kümelerin Özelliklere Göre Karşılaştırılması',
        hover_data=['Value']
    )
    fig_box_all.update_yaxes(matches=None)  # Her subplot kendi ölçeğinde
    fig_box_all.update_layout(
        xaxis_tickangle=45,
        legend_title_text='Küme',
        title_x=0.5
    )
    fig_box_all.show()

