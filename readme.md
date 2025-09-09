# Teen Phone Addiction Analysis
A comprehensive machine learning analysis of smartphone addiction patterns among teenagers using clustering techniques and predictive modeling with emphasis on handling class imbalance and model reliability.

## Project Overview
This project analyzes smartphone usage patterns and addiction levels among 3,000 teenagers aged 13-19 (simulated data). The analysis employs various machine learning techniques including clustering, classification, and feature importance analysis to understand the factors contributing to phone addiction, with special attention to overfitting prevention and model interpretability.

## Key Findings
### General Statistics

* Average Daily Phone Usage: 5.02 hours

* High Addiction Rate: 85.1% of teens (2,554 out of 3,000) have addiction scores ≥7

* Heavy Usage: 48.5% of teens use their phones for 5+ hours daily

* Usage Range: 0 to 11.5 hours per day

* Average Addiction Score: 8.88/10


## Analysis Methods
### Clustering Analysis
Optimal Clusters: 3 clusters identified using K-means

* Cluster Distribution:
Cluster 0: 1027 teens (34.2%)

Cluster 1: 923 teens (30.8%)

Cluster 2: 1,050 teens (35.0%)
### Statistical Testing

Performed F-tests showing significant differences across all variables (p < 0.001):

Family Communication (F=484.59)

Depression Level (F=249.84)

Screen Time Before Bed (F=197.02)

Sleep Hours (F=184.23)

Daily Usage Hours (F=92.59)

### Machine Learning Models
Four different approaches were employed to handle severe class imbalance:

* 1- Cost-Sensitive Learning

Cross-Validation: Accuracy: 85.1%, F1-Macro: 30.7%, ROC-AUC: 94.9%

Feature Importance-SHAP Correlation: 0.88 (robust)

Best for production use - most reliable results

* 2- SMOTE (Synthetic Minority Oversampling)

Cross-Validation: Accuracy: 91.7%, F1-Macro: 91.1%, ROC-AUC: 98.3%

Feature Importance-SHAP Correlation: 0.996 (high due to synthetic data)

Good minority class recall improvement

* 3- SMOTEENN (SMOTE + Edited Nearest Neighbors)

Cross-Validation: Accuracy: 92.8%, F1-Macro: 92.1%, ROC-AUC: 98.8%

Feature Importance-SHAP Correlation: 0.996 (high due to synthetic data)

Best overall threshold optimization results

* 4- ADASYN (Adaptive Synthetic Sampling)

Cross-Validation: Accuracy: 91.1%, F1-Macro: 90.8%, ROC-AUC: 98.2%

Feature Importance-SHAP Correlation: 0.997 (high due to synthetic data)

Adaptive sampling strategy

### Feature Importance

* Top Predictive Features (consistent across all models):

Daily Usage Hours (34-44%) - Primary predictor

Time on Social Media (13-16%)

Phone Checks Per Day (12-17%)

Time on Gaming (7-10%)

Sleep Hours (6-7%)

### Model Performance & Limitations

## Overfitting Analysis

Cost-Sensitive Approach: FI-SHAP correlation = 0.88 (robust, recommended)

Sampling Methods: FI-SHAP correlation = 0.996+ (high correlation due to synthetic data generation)

## Why High Correlation in Sampling Methods?

1-Synthetic data generation creates similar feature patterns

2-SMOTE/ADASYN algorithms interpolate between existing samples

3-This leads to artificially consistent feature importance rankings

4-Known limitation of oversampling techniques, especially with small minority classes

## Model Selection

* Cost-sensitive approach provides most robust results for production deployment

* Sampling methods are valuable for improving minority class detection but should be interpreted with caution

* Threshold optimization significantly improves rare class prediction

* Consider ensemble approaches combining multiple methods for optimal results

### Technical Implementations

## Data Processing

* Principal Component Analysis (PCA) for clustering

* Stratified cross-validation to maintain class distribution

* Multiple resampling techniques with regularization to prevent overfitting

* Threshold optimization for minority class improvement

## Model Configuration

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=15,
    max_features=2,
    max_samples=0.5,
    class_weight={0: 5, 1: 1, 2: 1} # Cost-sensitive weights
    #class_weight = {0: 3, 1: 2, 2: 1} for sampling methods (light class weighting)
)
```

## Evaluation Metrics

* Accuracy, Precision, Recall, F1-Score (Macro & Weighted)

* ROC-AUC for multi-class classification

* SHAP values for model interpretability

* Feature Importance correlation analysis

### Key Risk Factors Identified

1-Daily Usage Hours: Primary addiction indicator

2-Social Media Time: Strong behavioral predictor

3-Phone Check Frequency: Compulsive behavior marker

4-Gaming Duration: Entertainment addiction component

5-Sleep Pattern Disruption: Health impact indicator


### Recommendations:

* Monitor daily usage patterns

* Implement healthy screen time limits

* Encourage physical activities

* Promote better sleep hygiene

Strengthen family communication

### Getting Started
Prerequisites
```bash
pip install -r requirements.txt
```

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap imbalanced-learn
```

python
```bash
python teen_addiction_analysis.py
```
### Data Features
* The analysis includes 12 core features covering:

Usage Patterns: Daily hours, phone checks, screen time before bed

Content Consumption: Social media, gaming time

Psychological Factors: Anxiety level, depression level, self esteem

Health Metrics: Sleep hours, exercise hours

Social Factors: Family communication, social interactions


### 🤝 Contributing
Contributions are welcome,  please feel free to submit a Pull Request.

### 📄 License
This project is licensed under the MIT License.



This analysis provides insights into teenage smartphone addiction patterns and can be used to develop targeted intervention strategies for healthier digital habits.

