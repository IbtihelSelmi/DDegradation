# ICU Deterioration Prediction Tool
## Overview
ICU Guardian is a machine learning-based clinical decision support system designed to predict acute patient deterioration in intensive care units. This Flask web application utilizes a K-Nearest Neighbors (KNN) classifier trained on critical physiological parameters to provide early warnings of potential health deterioration, enabling medical staff to intervene proactively.

# Clinical Significance
Early detection of patient deterioration in ICU settings is crucial for improving outcomes and reducing mortality rates. This tool addresses the critical need for predictive analytics in healthcare by leveraging machine learning to identify at-risk patients based on established clinical parameters.

# Technical Architecture
## Machine Learning Framework
### Algorithm: K-Nearest Neighbors (KNN) classifier with k=3

### Feature Set: 12 clinically significant parameters including:

* Vital signs (Heart Rate, SBP, DBP, SpO2)

* Laboratory values (Lactates)

* Scoring systems (SOFA, SAPS II, Glasgow Coma Scale)

* Patient demographics (Age, Gender)

* Comorbidity indicators (HTA)

* Respiratory parameters (FiO2)

### Data Preprocessing: Min-Max normalization and comprehensive missing value handling

### Model Performance: Accuracy metrics provided with each prediction

## Web Application Stack
### Backend: Flask web framework with Python 3.7+

### Frontend: Responsive Bootstrap 5.3 interface with custom medical styling

### Data Handling: Pandas for data manipulation and scikit-learn for ML operations

### Model Persistence: Joblib for serialization and loading of trained models

## Key Features
1- Real-time Risk Assessment: Instant prediction of deterioration risk based on clinical inputs

2- Clinical Parameter Integration: Supports standard ICU monitoring parameters and scoring systems

3- Visual Risk Stratification: Color-coded output (green/red) for immediate visual assessment

4- Responsive Design: Mobile-optimized interface for bedside use

5- Model Transparency: Display of accuracy metrics for clinical validation

6- Data Validation: Robust error handling for missing or invalid clinical inputs

# Installation and Deployment
## Prerequisites
bash
pip install flask pandas scikit-learn joblib numpy
## Local Execution
bash
python main.py
Access the application at: http://localhost:5000

## Production Deployment
For clinical environments, deploy using:

WSGI server (Gunicorn)

Reverse proxy (Nginx)

Process management (Systemd)

# Clinical Parameters
The model requires the following inputs for prediction:

## Demographic Information
Age (years)

Gender (1=Female, 0=Male)

## Vital Signs
Heart Rate (bpm)

Systolic Blood Pressure - SBP (mmHg)

Diastolic Blood Pressure - DBP (mmHg)

Oxygen Saturation - SpO2 (%)

## Laboratory Values
Lactates (mmol/L) - optional

## Scoring Systems
SOFA Score (0-24)

SAPS II Score (0-163)

Glasgow Coma Scale (3-15)

## Comorbidities and Support
Hypertension (HTA) (binary)

Fraction of Inspired Oxygen - FiO2 (0.21-1.0)

# Model Development
## Data Source
The model was trained on clinical data from ICU patients, featuring comprehensive electronic health records with meticulous preprocessing to ensure data quality and clinical relevance.

## Training Methodology
* Data Splitting: 80/20 train-test split with random state stabilization

* Feature Selection: Clinical expertise-driven parameter selection

* Normalization: Min-Max scaling for optimal KNN performance

* Validation: Stratified sampling to maintain class distribution

### Performance Metrics
The model achieves clinically relevant accuracy as displayed with each prediction, providing transparency about its predictive capabilities.

## Clinical Implementation Considerations
1- Intended Use: Decision support tool only - not a replacement for clinical judgment

2- Integration: Designed for integration with hospital EHR systems

3- Validation: Requires local validation with institution-specific data

4- Training: Healthcare professional orientation recommended before clinical use

## Ethical Considerations
* Patient data privacy protection implemented

* Algorithmic bias assessment recommended for diverse populations

* Clinical oversight required for all decision-making

* Regular model performance monitoring advised

## Future Enhancements
* Integration with real-time patient monitoring systems

* Additional predictive algorithms (Random Forest, XGBoost)

* Extended clinical parameter support

 * Advanced visualization of trend data

* API for EHR system integration

## License
© 2025 Medical Resuscitation Service. All rights reserved.

## Citation
Please acknowledge this tool in clinical publications or research derived from its use.

For technical support or clinical implementation guidance, please contact the development team at the Medical Resuscitation Service - Farhat Hached Hospital Sousse.
