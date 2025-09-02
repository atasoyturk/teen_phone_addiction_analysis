import pandas as pd


from sklearn.ensemble import RandomForestClassifier
#scikit-learn kütüphanesinden Random Forest sınıflandırıcısını çağırıyoruz.
#birden çok decision tree + majority vote” mantığını otomatik uyguluyor.
#Örneğin, model şu kuralı keşfedebilir: "Eğer Phone_Checks_Per_Day > 20 VE Screen_Time_Before_Bed > 2 saat ise, o kişi %98 ihtimalle Grup 1'e aittir."

from sklearn.metrics import classification_report, roc_auc_score
#Model tahmin yaptıktan sonra, performansı ölçmek için kullanıyoruz.
#accuracy, precision, recall, f1-score gibi metrikleri özetleyen bir tablo döndürüyor.

#makine öğrenmesinde aşırı örnekleme (oversampling) teknikleri kullanarak dengesiz veri setlerini dengelemeye çalışıyoruz.
#sınıflar arası dengesizlik (class imbalance) sorununu çözmek için kullanılan bir yapay veri üretme (oversampling) yöntemidir.

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
'''
#SHAP (SHapley Additive exPlanations),
makine öğrenmesi modellerinin tahminlerini açıklamak için kullanılan güçlü bir açıklayıcı yapay zeka (XAI - eXplainable AI) yöntemidir.
"Neden bu kişi 'yüksek riskli' olarak sınıflandırıldı?", Yanlış tahminler neden yapıldı? Hangi özellik yanlış yönlendirdi?
TreeSHAP - Ağaç tabanlı modeller (Random Forest, XGBoost, LightGBM) için optimize edilmiştir. Hızlı ve doğru. 

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


def random_forest_with_oversampling(X_train, X_test, y_train, y_test, oversampler, method_name, n_estimators=100, max_depth=10):
    '''
    X_train, X_test: bağımsız değişkenler (örneğin Daily_Usage_Hours, Phone_Checks_Per_Day …)
    y_train, y_test: hedef değişken (sınıf etiketleri)
    n_estimators=100: kaç tane decision tree kurulsun? (default: 100)
    max_depth=10: ağaçların maksimum derinliği → None olursa tamamen büyüyebiliyor, overfitting riski olabilir.
    
    '''
    X_train_bal, y_train_bal = oversampler.fit_resample(X_train, y_train)
    #Low ve Medium sınıflardan yapay örnekler üretir, Dengeli veri seti oluşturur
    #bu sayede model çoğunluğu değil, tüm sınıfları ogrenebilir.


    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    #cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    '''
    n_splits veriyi kaç parçaya bolecegini ifade eder.
    
    shuffle, veriyi parçalara rastgele ayırmayı sağlar, 
    shuffle olmadan -> 1. parça : row 0-199, 2.parça row 200-399 vs...
    shuffle olursa ->1. parça : row 45,103, 567, 685.. 2.parça row 12, 188, 400, 513..
    
    '''
    
    # Farklı metriklerde CV skorları
    
    cv_accuracy = cross_val_score(model, X_train_bal, y_train_bal, cv=cv, scoring='accuracy')
    #kaç tahmin dogru? yuzdelik verir. 
    
    #F1, doğruluk (precision) ve hatırlama (recall) metriklerinin ortalamasıdır. Sınıflandırma performansını tek bir sayıda özetler.(0 en kötü, 1 en iyi)

    cv_f1_macro = cross_val_score(model, X_train_bal, y_train_bal, cv=cv, scoring='f1_macro')
    #Her sınıfın F1 skorunu eşit ağırlıkla (ortalama) alır. Büyük sınıflar da, küçük sınıflar da aynı önemde.
    cv_f1_weighted = cross_val_score(model, X_train_bal, y_train_bal, cv=cv, scoring='f1_weighted')
    #Her sınıfın F1 skoru, o sınıfın verideki sıklığına göre ağırlıklandırılır. Büyük sınıflar daha çok etki eder.
    
    cv_roc_auc = cross_val_score(model, X_train_bal, y_train_bal, cv=cv, scoring='roc_auc_ovr')
    #roc-auc sınıfları ayırt etme kalitesi : 0.5 rastegele tahmin, 1.0 mükemmel ayırt etme
    
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
    X_test verileri (daily usage hours, phone check per day vs..) veriyoruz → model bu kişilerin hangi kümeye/sınıfa ait olduğunu tahmin ediyor.
    Çıktı y_pred → modelin tahmin ettiği label dizisi. (array örn : [0 1 1 0 1] )
    
    y_pred_proba, modelin tahmin ettigi olasılık sonuçlarıdır (2d array)
        Örn:[[0.2,, 0.7, 0.1],
            [0.6, 0.2, 0.2],   ----> örnegin 2. satırdaki öğrenci yuzde 60 ihtimalle (0.6) 1. kumeye ait
            [0.8, 0.2, 0.0]
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
    #cross validationdaki AUC-ROC ile classification reporttaki ROC-AUC  aynıdır.
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
   
    
    explainer = shap.TreeExplainer(model)
    # explainer, her feautre icin bir tahminde ne kadar etkili old. gosteren bir shap nesnesi 
    shap_values_raw = explainer.shap_values(X_test)
    #shap values_raw ise shap değerlerini gosterir.
    #XGBoost, Random Forest, LightGBM kullanılırsa genelde list şeklinde verir.


    class_of_interest = 2  # High bağımlılık sınıfı

    #isinstance(x, tür) x'in o tür olup olmadıgını kontrol eder.
    if isinstance(shap_values_raw, list):
        shap_values_selected = shap_values_raw[class_of_interest]  
        #eğer list ise sadece 3. sınfın shap değerlerini alır
    else:
        shap_values_selected = shap_values_raw[:, :, class_of_interest]
        #eğer list degil örn. 3d dizi ise [:, :, x] tüm satırlardaki(öğrenciler) tüm sutunların (features) sadece 3. sınıfı al

    shap_df = pd.DataFrame(shap_values_selected, columns=X_test.columns)
    #shap_values_raw'ı df haline getirdiks


    return model, feature_importance, shap_df
    #Özelliklerin (feature) modeldeki göreceli önemini gösterir.