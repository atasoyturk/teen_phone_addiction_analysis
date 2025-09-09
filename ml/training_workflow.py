from .model import random_forest_cost_sensitive, random_forest_with_oversampling
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN

def train_models_and_get_importance(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=10):
    
    '''
    4 different scenario: 
        cost sensitive model (no oversampling)
        model with SMOTE
        model with SMOTEEN
        model with ADANYS
        
    '''
    results = {} 
    
    # simple randomforest model, class wights are balanced
    model, fi, shap_magnitude= random_forest_cost_sensitive(
        X_train, X_test, y_train, y_test,
        n_estimators=n_estimators,
        max_depth=max_depth
    )
    
    results['Cost Sensitive'] = {
        'model': model,
        'feature_importance': fi,
        'shap_magnitude': shap_magnitude    
    }

    #----------------
    # randomforest models with oversampling methods
    # all methods cause overfitted models when the strategy is 'auto'

    # For multi-class, sampling_strategy must be a dict specifying desired number of samples per class
    # Assuming classes are 0,1,2 and class 0 is minority, we oversample class 0 to 50% of majority class count
    # Dynamically calculate sampling_strategy based on class distribution to avoid excessive oversampling
    class_counts = y_train.value_counts()
    max_count = class_counts.max()
    sampling_strategy_dict = {cls: int(max_count * 0.5) for cls in class_counts.index if class_counts[cls] < max_count}
    
    oversamplers = {
        "SMOTE": SMOTE(sampling_strategy=sampling_strategy_dict, random_state=42, k_neighbors=3), #most basic method, no noice tolerance, overfitting risk 
        "SMOTEENN": SMOTEENN(sampling_strategy=sampling_strategy_dict, random_state=42), #most clever method, Less overfitting risk
        "ADASYN": ADASYN(sampling_strategy=sampling_strategy_dict, random_state=42, n_neighbors=3) #most adaptive method, more focus on difficult samples
    }
    
        
    for name, sampler in oversamplers.items():
        
        model, fi, shap_magnitude = random_forest_with_oversampling(
            X_train, X_test, y_train, y_test,
            oversampler=sampler,
            method_name=name
        )

        results[name] = {
            'model': model,
            'feature_importance': fi,
            'shap_magnitude': shap_magnitude
        }
        
        print(f"\n{name} - Feature Importance:")
        print(fi)

    return results