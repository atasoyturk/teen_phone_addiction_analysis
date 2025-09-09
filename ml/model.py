import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold


from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN
import shap

def optimize_minority_class_threshold(model, x_val, y_val, minority_class = 0):

    y_proba = model.predict_proba(x_val)

    y_true_binary = (y_val == minority_class).astype(int)
    # Find the index of minority_class in model.classes_
    minority_index = list(model.classes_).index(minority_class)
    y_proba_minority = y_proba[:, minority_index]

    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_proba_minority)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    print(f"Optimal Class {minority_class} threshold: {optimal_threshold:.3f}")
    return optimal_threshold


def predict_with_threshold(model, X, threshold=0.1, target_class=0):

    y_proba = model.predict_proba(X)

    predictions = model.predict(X)  # This gives class labels

    # Find the index of target_class
    target_index = list(model.classes_).index(target_class)

    predictions[y_proba[:, target_index] >= threshold] = target_class

    return predictions



def random_forest_cost_sensitive(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=8):
    # firstly, we need to check model results without oversampling methods.
    
    #20x importance could be given to class 0, so model becomes overfitted when class_weight = 'balanced'
    #class weight form is like this for now
    class_weight = {0: 5, 1: 1, 2: 1}
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        min_samples_split=20,  # Increased to reduce overfitting
        min_samples_leaf=15,   # Increased to reduce overfitting
        max_features= 2,
        max_samples=0.5,       # Added to reduce overfitting by using subsample of data
        random_state=42
    )
    
    #cross valid.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_accuracy = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_f1_macro = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
    cv_f1_weighted = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
    cv_roc_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc_ovr')
    
    print(f"\n=== Cross-Validation Results (cost sensitive) ===")
    print(f"Accuracy:    {cv_accuracy.mean():.3f} ± {cv_accuracy.std():.3f}")
    print(f"F1-Macro:    {cv_f1_macro.mean():.3f} ± {cv_f1_macro.std():.3f}")
    print(f"F1-Weighted: {cv_f1_weighted.mean():.3f} ± {cv_f1_weighted.std():.3f}")
    print(f"ROC-AUC:     {cv_roc_auc.mean():.3f} ± {cv_roc_auc.std():.3f}")
    
    model.fit(X_train, y_train)
    
    X_val, X_test_new, y_val, y_test_new = train_test_split(
        X_test, y_test, test_size=0.5, stratify=y_test, random_state=42
    )
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    print("\n=== Test Set - Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    print(f"AUC-ROC: {auc:.5f}")
    
    
    threshold = optimize_minority_class_threshold(model, X_val, y_val)
    y_pred_threshold = predict_with_threshold(model, X_test_new, threshold)
    print(f"\n==== AFTER THRESHOLD OPTIMIZATION ===")
    print(classification_report(y_test_new, y_pred_threshold, zero_division=0))
    
    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    
    
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_test)
    #shap values are shown in list format, if XGBoost, Random Forest, LightGBM (tree based models) were used

    
    class_of_interest = 0  # high addiction class
    
    if isinstance(shap_values_raw, list):
        shap_values_selected = shap_values_raw[class_of_interest]
    else:
        shap_values_selected = shap_values_raw[:, :, class_of_interest]
    
    shap_magnitude = np.abs(shap_values_selected).mean(axis=0)
    
    
    return model, feature_importance, shap_magnitude
    
   


def random_forest_with_oversampling(X_train, X_test, y_train, y_test, oversampler, method_name, n_estimators=100, max_depth=10):
    
    X_train_bal, y_train_bal = oversampler.fit_resample(X_train, y_train)
    
    #sampling methods + light cost_weight
    class_weight = {0: 3, 1: 2, 2: 1}
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        min_samples_split=20,  # Increased to reduce overfitting
        min_samples_leaf=15,   # Increased to reduce overfitting
        max_features= 2,
        max_samples=0.5,       # Added to reduce overfitting by using subsample of data
        random_state=42
    )
    
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
    
    X_val, X_test_new, y_val, y_test_new = train_test_split(
        X_test, y_test, test_size=0.5, stratify=y_test, random_state=42
    )
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)  

    
    print(classification_report(y_test, y_pred, zero_division=0))
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    print(f"AUC-ROC: {auc:.5f}")
    
    
    threshold = optimize_minority_class_threshold(model, X_val, y_val)
    y_pred_threshold = predict_with_threshold(model, X_test_new, threshold)
    print(f"\n=== {method_name} - AFTER THRESHOLD OPTIMIZATION ===")
    print(classification_report(y_test_new, y_pred_threshold, zero_division=0))
    
    
    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
   
    
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_test)
    

    class_of_interest = 0

    if isinstance(shap_values_raw, list):
        shap_values_selected = shap_values_raw[class_of_interest]  
    else:
        shap_values_selected = shap_values_raw[:, :, class_of_interest]
        #if shap_values_raw is not list (could be 3d array),  [:, :, x] means all rows and all columns of x'th class
        
    if shap_values_selected.ndim != 2:
        raise ValueError(f"SHAP values are not 2D! Shape: {shap_values_selected.shape}")

    shap_df = pd.DataFrame(shap_values_selected, columns=X_test.columns)
    shap_magnitude = shap_df.abs().mean(axis=0)  
    #np.abs() may cause type changes, shap_df is dataframe now but np.abs() returns array, so we use pandas abs method.

    return model, feature_importance, shap_magnitude
