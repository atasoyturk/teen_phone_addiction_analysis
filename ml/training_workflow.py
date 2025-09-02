from .model import random_forest_with_oversampling
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN

def train_models_and_get_importance(X_train, X_test, y_train, y_test):

    oversamplers = {
        "SMOTE": SMOTE(sampling_strategy={0: 200, 1: 400}, random_state=42, k_neighbors=5),
        "SMOTEENN": SMOTEENN(sampling_strategy={0: 200, 1: 400}, random_state=42),
        "ADASYN": ADASYN(sampling_strategy='minority', random_state=42, n_neighbors=3)
    }
    
    results = {}
    
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