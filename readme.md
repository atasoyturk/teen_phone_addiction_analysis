# Teen Phone Addiction Analysis
A comprehensive machine learning analysis of smartphone addiction patterns among teenagers using clustering techniques and predictive modeling.

## Project Overview
This project analyzes smartphone usage patterns and addiction levels among 3,000 teenagers aged 13-19 (simulated data). The analysis employs various machine learning techniques including clustering, classification, and feature importance analysis to understand the factors contributing to phone addiction.

## Key Findings
### General Statistics

* Average Daily Phone Usage: 5.02 hours

* High Addiction Rate: 86% of teens (2,581 out of 3,000) have addiction scores ≥7

* Heavy Usage: 48.5% of teens use their phones for 5+ hours daily

* Usage Range: 0 to 11.5 hours per day

### Demographics
* Gender Distribution: Relatively balanced across Male (33.9%), Female (33.6%), and Other (32.6%)

* Age Range: 13-19 years

* Sample Size: 3,000 teenagers (simulated teenagers)

* Average Addiction Score: 8.88/10

## Analysis Methods
### Clustering Analysis
Optimal Clusters: 3 clusters identified using K-means

* Cluster Distribution:
Cluster 0: 993 teens (33.1%)

Cluster 1: 936 teens (31.2%)

Cluster 2: 1,071 teens (35.7%)
### Statistical Testing

Performed F-tests on key variables showing significant differences across clusters:

Daily Usage Hours (F=114.33, p<0.001)

Sleep Hours (F=235.39, p<0.001)

Family Communication (F=460.49, p<0.001)

Time on Social Media (F=126.89, p<0.001)

### Machine Learning Models
Three different sampling techniques were employed to handle class imbalance:

* SMOTE (Synthetic Minority Oversampling Technique)

Accuracy: 91.2% ± 0.8%

F1-Macro: 85.2% ± 1.3%

ROC-AUC: 97.5% ± 0.5%

* SMOTEENN (SMOTE + Edited Nearest Neighbors)

Accuracy: 98.1% ± 0.3%

F1-Macro: 83.4% ± 4.1%

ROC-AUC: 99.0% ± 0.6%

* ADASYN (Adaptive Synthetic Sampling)

Accuracy: 93.9% ± 0.3%

F1-Macro: 75.6% ± 1.7%

ROC-AUC: 98.4% ± 0.1%

### Feature Importance

* Top Predictive Features (SMOTE Model):

Daily Usage Hours (32.8%) - Most important predictor

Time on Social Media (13.2%)

Phone Checks Per Day (12.1%)

Time on Gaming (10.4%)

Sleep Hours (8.5%)

Secondary Factors:

Exercise Hours (4.1%)

Screen Time Before Bed (3.9%)

Depression Level (3.3%)

Social Interactions (3.1%)

Family Communication (2.9%)

### Model Performance
All models show excellent performance with high correlation between Feature Importance and SHAP values:

* SMOTE: 99.11% correlation
* SMOTEENN: 98.68% correlation
* ADASYN: 99.17% correlation
### Technical Implementation
* Data Processing

* Principal Component Analysis (PCA) for dimensionality reduction

* Cross-validation with stratified sampling

* Multiple sampling techniques to handle class imbalance

* Evaluation Metrics

* Accuracy, Precision, Recall, F1-Score

* ROC-AUC for multi-class classification

* SHAP values for model interpretability

### Addiction Level Distribution

Low (1-4): 42 teens (1.4%)

Medium (4-7): 404 teens (13.5%)

High (7-10): 2,554 teens (85.1%)

### Implications
* Key Risk Factors:

High Daily Usage: Primary indicator of addiction

Social Media Consumption: Strong secondary predictor

Frequent Phone Checking: Behavioral indicator

Gaming Time: Significant contributor

Sleep Disruption: Important health-related factor

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

python
```bash
python teen_addiction_analysis.py
```
### Data Features
* The analysis includes 20+ features covering:

Usage Patterns: Daily hours, phone checks, screen time

Content Consumption: Social media, gaming, education time

Health Metrics: Sleep hours, exercise, anxiety levels

Social Factors: Family communication, social interactions

Demographics: Age, gender, location

### 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

### 📄 License
This project is licensed under the MIT License.



This analysis provides insights into teenage smartphone addiction patterns and can be used to develop targeted intervention strategies for healthier digital habits.

