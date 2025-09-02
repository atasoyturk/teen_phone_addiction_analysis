import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold


from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

from imblearn.over_sampling import ADASYN
import shap



def random_forest_with_oversampling(X_train, X_test, y_train, y_test, oversampler, method_name, n_estimators=100, max_depth=10):
    
    X_train_bal, y_train_bal = oversampler.fit_resample(X_train, y_train)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    
    #cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
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
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)  

    
    print(classification_report(y_test, y_pred))
    # Burada gerçek etiketler (y_test) ile tahmin edilen etiketler (y_pred) karşılaştırılıyor.
    
    print("AUC-ROC:", roc_auc_score(y_test, y_pred_proba, multi_class='ovr'))
    
    
    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
   
    
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_test)
    
    #XGBoost, Random Forest, LightGBM kullanılırsa genelde list şeklinde verir.

    class_of_interest = 2  # High bağımlılık sınıfı

    if isinstance(shap_values_raw, list):
        shap_values_selected = shap_values_raw[class_of_interest]  
    else:
        shap_values_selected = shap_values_raw[:, :, class_of_interest]
        #eğer list degil örn. 3d dizi ise [:, :, x] tüm satırlardaki(öğrenciler) tüm sutunların (features) sadece 3. sınıfı al
        
    if shap_values_selected.ndim == 1:
        raise ValueError(f"SHAP values 1D geldi! Shape: {shap_values_selected.shape}. X_test: {X_test.shape}")

    shap_df = pd.DataFrame(shap_values_selected, columns=X_test.columns)
    shap_magnitude = np.abs(shap_df).mean(axis=0)  

    return model, feature_importance, shap_magnitude
