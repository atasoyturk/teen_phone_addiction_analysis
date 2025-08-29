from sklearn.ensemble import RandomForestClassifier
#scikit-learn kütüphanesinden Random Forest sınıflandırıcısını çağırıyoruz.
#birden çok decision tree + majority vote” mantığını otomatik uyguluyor.
#Örneğin, model şu kuralı keşfedebilir: "Eğer Phone_Checks_Per_Day > 20 VE Screen_Time_Before_Bed > 2 saat ise, o kişi %98 ihtimalle Grup 1'e aittir."

from sklearn.metrics import classification_report
#Model tahmin yaptıktan sonra, performansı ölçmek için kullanıyoruz.
#accuracy, precision, recall, f1-score gibi metrikleri özetleyen bir tablo döndürüyor.

import pandas as pd


def run_random_forest(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None):
    '''
    X_train, X_test: bağımsız değişkenler (örneğin Daily_Usage_Hours, Phone_Checks_Per_Day …)
    y_train, y_test: hedef değişken (sınıf etiketleri)
    n_estimators=100: kaç tane decision tree kurulsun? (default: 100)
    max_depth=None: ağaçların maksimum derinliği → None olursa tamamen büyüyebiliyor, overfitting riski olabilir.
    
    '''
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    '''
    Yani X_train (özellikler) ile y_train (etiketler) arasındaki ilişkileri öğreniyor.
    Bu aşamada ağaçlar oluşturuluyor, her biri farklı veri subset’leri ile eğitiliyor.
    
    '''
    y_pred = model.predict(X_test)
    '''
    X_test verileri veriyoruz → model bu kişilerin hangi kümeye/sınıfa ait olduğunu tahmin ediyor.
    Çıktı y_pred → modelin tahmin ettiği label dizisi.
    '''
    print(classification_report(y_test, y_pred))
    # Burada gerçek etiketler (y_test) ile tahmin edilen etiketler (y_pred) karşılaştırılıyor.
    '''
    classification_report şu metrikleri basar:

        precision: modelin tahmin ettiği sınıflardan kaçı doğru?

        recall: gerçek sınıfların ne kadarını doğru buldu?

        f1-score: precision ve recall’un dengeli ortalaması

        support: her sınıftan kaç örnek var?
    '''
    
    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    return model, feature_importance
    #Özelliklerin (feature) modeldeki göreceli önemini gösterir.