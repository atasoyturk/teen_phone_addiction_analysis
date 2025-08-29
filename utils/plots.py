import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from ml.standardization import standardization_process


def general_plotting(df):
    
    if df is not None: 
        fig = px.scatter(
            df, 
            x = 'Exercise_Hours',
            y = 'Time_on_Gaming',
            title = 'Exercise Hours vs Gaming Time',
            labels = {
                'Exercise_Hours': 'Hours of Exercise',
                'Time_on_Gaming': 'Hours Spent on Gaming'
            },
            custom_data=['Gender', 'Age']  # hover için ekstra veri

        )

        #size icin normalizasyon
        size = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())  # 0-1 arası
        size = size * 20 + 5  # 5-25 piksel arası
        
        
        fig.update_traces(
            marker=dict(
                        color = df['Gender'].map({'Male': 'blue', 'Female': 'pink', 'Other': 'green'}),
                        #df[].map() ile otomatik sekilde istenilen özelliğe göre renk ataması yapılabilr
                        size= size,
                        sizemode = 'area',
                        opacity = 0.6,
                        line = dict(width=1, color='black')
                    ),

            hovertemplate=
                '<b>Gender:</b> %{customdata[0]}<br>' +
                '<b>Age: %{customdata[1]}<br>' +
                '<b>Exercise Hours:</b> %{x}<br>' +
                '<b>Gaming Hours:</b> %{y}<br>'
        )
        
        fig.update_layout(
            xaxis_title='Hours of Exercise',
            yaxis_title='Hours Spent on Gaming',
            xaxis = dict(
                showgrid=True,
                zeroline=True,
                showline=True,
                title=dict(standoff=10), # standoff = x ekseni ile x ekseni başlığı arasındaki mesafe
                ticks='outside', # bu, x eksenindeki tick'lerin dışarıda görünmesini sağlar
                ticklen=5,
                tickwidth=2,
                tickcolor='black',
                gridcolor='lightgray',
                gridwidth=1
            ),
            yaxis = dict(
                showgrid=True,
                zeroline=True,
                showline=True,
                title=dict(standoff=10), # standoff ile başlık arasındaki mesafe
                ticks='outside', # bu, y eksenindeki tick'lerin dışarıda görünmesini sağlar
                ticklen=5,
                tickwidth=2,
                tickcolor='black',
                gridcolor='lightgray',
                gridwidth=1
            ),
            margin=dict(l=40, r=40, t=80, b=40),
            legend = {
                'title': 'Gender',
                'orientation': 'h',   # yatay
                'xanchor': 'center', 'x': 0.5, #legendin ortası (center)  grafigin tam ortasına (0.5) gelecek
                'yanchor': 'bottom', 'y': 1.05, #legendin alt kısmı (bottom)  grafigin  biraz üst kısmına (1.05) gelecek
            }
                
        )
        
        fig.show()
    
        
    corr = df[['Anxiety_Level', 'Depression_Level', 'Sleep_Hours', 'Daily_Usage_Hours', 'Self_Esteem']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Psikolojik ve Davranışsal Değişkenler Arası Korelasyon")
    plt.show()
    plt.close()  # Mevcut figürü kapatır, belleği temizler
    
    sns.pairplot(
    df,
    vars=['Sleep_Hours', 'Anxiety_Level', 'Daily_Usage_Hours', 'Academic_Performance'],
    hue='Gender'
    )
    plt.show()
    plt.close()  # Mevcut figürü kapatır, belleği temizler
    
    
def cluster_plots (path):
    
    addiction_df, _ = standardization_process(path)  # Kümelenmiş veri lazım
    
    addiction_df_melted = pd.melt(
    addiction_df,
    id_vars=['Cluster'],
    value_vars=['Daily_Usage_Hours', 'Phone_Checks_Per_Day',
               'Screen_Time_Before_Bed', 'Time_on_Social_Media'],
    var_name='Feature',
    value_name='Value'
    )
    #  pd.melt(),  df olan veriyi uzun formata çevirir (px ile daha rahat gorslelestirme)
    
    fig_addiction = px.box(
        addiction_df_melted,
        x = 'Feature',
        y = 'Value', 
        color = 'Cluster', 
        title = 'Kümelerin Özelliklere Göre Karşılaştırılması',
        facet_col='Feature',  
        #facet_col ile hser özellik ayrı bir subplot'ta, Her birinin dağılımı net görülür.
        #Küme arasındaki farklar açıkça karşılaştırılabilir.
        hover_data=['Value'], #mouse ile uzerine gelince deger gozukur
        
    )
    fig_addiction.update_yaxes(matches=None)  # Her subplot kendi ölçeğini kullanır (facet kullanımında mantıklı)

    fig_addiction.update_layout(
    xaxis_tickangle=45,
    legend_title_text='Küme',
    title_x=0.5  # Başlığı ortala
    )
    
    fig_addiction.show()