from sklearn.ensemble import RandomForestClassifier
#scikit-learn kütüphanesinden Random Forest sınıflandırıcısını çağırıyoruz.
#birden çok decision tree + majority vote” mantığını otomatik uyguluyor.
#Örneğin, model şu kuralı keşfedebilir: "Eğer Phone_Checks_Per_Day > 20 VE Screen_Time_Before_Bed > 2 saat ise, o kişi %98 ihtimalle Grup 1'e aittir."

from sklearn.metrics import classification_report, roc_auc_score,confusion_matrix
#Model tahmin yaptıktan sonra, performansı ölçmek için kullanıyoruz.
#accuracy, precision, recall, f1-score gibi metrikleri özetleyen bir tablo döndürüyor.

#makine öğrenmesinde aşırı örnekleme (oversampling) teknikleri kullanarak dengesiz veri setlerini dengelemeye çalışıyoruz.
#sınıflar arası dengesizlik (class imbalance) sorununu çözmek için kullanılan bir yapay veri üretme (oversampling) yöntemidir.
'''
Özellikle makine öğrenimi modellerinde, bir sınıfın (örneğin pozitif sınıf) veri sayısı diğerine göre çok az olduğunda 
(örneğin 95 negatif, 5 pozitif), model çoğunluk sınıfına eğilimli olur ve azınlık sınıfını yeterince öğrenemez
'''
from sklearn.model_selection import cross_val_score, StratifiedKFold
'''
Model gerçekten iyi mi, yoksa sadece şanslı mı anlamamıza yarar.
# Veriyi böl: Train vs Test
X_train → Model eğit → X_test ile test et → 1 skor  -----> bu test yanıltıcı olabilir.

# Veriyi 5 parçaya böl:
Parça 1,2,3,4 → Eğit → Parça 5 test → Skor 1
Parça 1,2,3,5 → Eğit → Parça 4 test → Skor 2  
Parça 1,2,4,5 → Eğit → Parça 3 test → Skor 3
...
# 5 farklı skor → Ortalama ± std

(Tek test: "Bu model %97 başarılı"
 CV ile: "Bu model 5 farklı testte %97.5±0.5 başarılı" → Çok daha güvenilir!)

'''

from imblearn.over_sampling import SMOTE
'''
Azınlık sınıfından bir örnek seçilir.
Bu örneğin k-en yakın komşusu (k-nearest neighbors) bulunur.
Bu komşulardan rastgele biri seçilir.
Gerçek örnek ile seçilen komşu arasında rastgele bir nokta oluşturulur (interpolasyon).
Bu yeni nokta veri setine eklenir.
Bu işlem, azınlık sınıfının yapay olarak artırılmasını sağlar.
Rastgele kopyalama (duplication) yerine yeni örnekler üretir, bu da aşırı öğrenmeyi (overfitting) kısmen azaltır.

'''
from imblearn.combine import SMOTEENN
'''
SMOTE uygulanır → veri artırılır.
Sonra ENN uygulanır: Her örnek için en yakın komşularına bakılır. Eğer komşuların çoğu farklı sınıftan ise, örnek veri setinden çıkartılır.
Bu, sınırlara yakın, gürültülü veya yanlış sınıflandırılmış örnekleri temizler.
Sonuç: Daha temiz, daha dengeli bir veri seti.
'''
from imblearn.over_sampling import ADASYN
'''
SMOTE’a benzer ama ağırlıklı bir şekilde çalışır.
Azınlık sınıfında, çevresi çoğunluk sınıfıyla çevrili olan örnekler (yani sınıflandırması zor olanlar) daha çok sentetik örnek alır.
Yani: "Bu bölge zor, buraya daha çok sentetik örnek üretelim."
Kötü Yanı: Gürültülü verilerde aşırı sentetik örnek üretip overfittinge neden olabilir.
'''

import shap
#SHAP (SHapley Additive exPlanations),
#makine öğrenmesi modellerinin tahminlerini açıklamak için kullanılan güçlü bir açıklayıcı yapay zeka (XAI - eXplainable AI) yöntemidir.
#"Neden bu kişi 'yüksek riskli' olarak sınıflandırıldı?", Yanlış tahminler neden yapıldı? Hangi özellik yanlış yönlendirdi?
#TreeSHAP - Ağaç tabanlı modeller (Random Forest, XGBoost, LightGBM) için optimize edilmiştir. Hızlı ve doğru. 


import pandas as pd


def random_forest_with_oversampling(X_train, X_test, y_train, y_test, oversampler, method_name, n_estimators=100, max_depth=10):
    '''
    X_train, X_test: bağımsız değişkenler (örneğin Daily_Usage_Hours, Phone_Checks_Per_Day …)
    y_train, y_test: hedef değişken (sınıf etiketleri)
    n_estimators=100: kaç tane decision tree kurulsun? (default: 100)
    max_depth=None: ağaçların maksimum derinliği → None olursa tamamen büyüyebiliyor, overfitting riski olabilir.
    
    '''
    
    X_train_bal, y_train_bal = oversampler.fit_resample(X_train, y_train)


    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    # CROSS-VALIDATION EKLE
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Farklı metriklerde CV skorları
    cv_accuracy = cross_val_score(model, X_train_bal, y_train_bal, cv=cv, scoring='accuracy')
    cv_f1_macro = cross_val_score(model, X_train_bal, y_train_bal, cv=cv, scoring='f1_macro')
    cv_f1_weighted = cross_val_score(model, X_train_bal, y_train_bal, cv=cv, scoring='f1_weighted')
    cv_roc_auc = cross_val_score(model, X_train_bal, y_train_bal, cv=cv, scoring='roc_auc_ovr')
    
    print(f"\n Cross-Validation Results ({method_name}):")
    print(f"Accuracy:    {cv_accuracy.mean():.3f} ± {cv_accuracy.std():.3f}")
    print(f"F1-Macro:    {cv_f1_macro.mean():.3f} ± {cv_f1_macro.std():.3f}")
    print(f"F1-Weighted: {cv_f1_weighted.mean():.3f} ± {cv_f1_weighted.std():.3f}")
    print(f"ROC-AUC:     {cv_roc_auc.mean():.3f} ± {cv_roc_auc.std():.3f}")

    model.fit(X_train_bal, y_train_bal)
    '''
    Yani X_train (özellikler) ile y_train (etiketler) arasındaki ilişkileri öğreniyor.
    Bu aşamada ağaçlar oluşturuluyor, her biri farklı veri subset’leri ile eğitiliyor.
    
    '''
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)  # olasılıkları al

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
    print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba, multi_class='ovr'))
    '''
    AUC-ROC, modelin sınıflar arasındaki ayrımı ne kadar iyi yaptığını gösterir.
    1'e yakın değerler iyi, 0.5'e yakın değerler kötü ayrım anlamına gelir.
    
    AUC-ROC, modelin "karar verme kalitesini" ölçer.
    Classification Report, modelin "karar sonrası sonuçlarını" detaylı analiz eder. 
    '''
    
    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    
    # SHAP Explainer oluştur (TreeSHAP)
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_test)

    # SHAP çıktısı 3D mi (yeni SHAP) yoksa liste mi (eski SHAP)?
    class_of_interest = 2  # High bağımlılık sınıfı

    if isinstance(shap_values_raw, list):
        shap_values_selected = shap_values_raw[class_of_interest]  # (n_samples, n_features)
    else:
        shap_values_selected = shap_values_raw[:, :, class_of_interest]

    # Özellik önem sırası: SHAP değerlerinin mutlak ortalaması
    shap_df = pd.DataFrame(shap_values_selected, columns=X_test.columns)


    return model, feature_importance, shap_df
    #Özelliklerin (feature) modeldeki göreceli önemini gösterir.