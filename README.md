# Jaundice Prediction System

A machine learning-based diagnostic tool that predicts the likelihood of jaundice in patients based on clinical symptoms and bilirubin levels. This system uses a Random Forest Classifier to analyze patient data and provide risk assessments.

## 📋 Overview

This project implements a binary classification model to detect potential jaundice cases using patient symptoms and demographic information. The system is designed as an educational tool to demonstrate the application of machine learning in healthcare diagnostics.

## 🚀 Features

- **Patient Data Collection**: Interactive command-line interface for entering patient symptoms
- **Machine Learning Model**: Random Forest Classifier trained on synthetic patient data
- **Risk Assessment**: Provides probability-based risk scores
- **Symptom Analysis**: Identifies key symptoms contributing to the prediction
- **Medical Disclaimer**: Includes appropriate healthcare warnings

## 🏥 Symptoms Analyzed

The system evaluates the following clinical indicators:
- Elevated bilirubin levels
- Yellow eyes (icterus)
- Yellow skin (jaundice)
- Dark-colored urine
- Fatigue/weakness
- Abdominal pain
- Patient age

## 📊 Data Structure

Each patient record contains 8 features: [bilirubin, yellow_eyes, yellow_skin, age, dark_urine, fatigue, abdominal_pain, label]

- **Features (0-6)**: Binary indicators (0/1) for symptoms + age (integer)
- **Label (7)**: Binary diagnosis (0 = No Jaundice, 1 = Jaundice)

## 🔧 Installation

### Prerequisites
   
pip install pandas scikit-learn numpy

💻 Usage
Run the Application
bash
python jaundice_predictor.py
Input Process
The system will prompt you for:

Elevated bilirubin level? (0 = No, 1 = Yes)

Yellow eyes? (0 = No, 1 = Yes)

Yellow skin? (0 = No, 1 = Yes)

Age (years)

Dark-coloured urine? (0 = No, 1 = Yes)

Fatigue or weakness? (0 = No, 1 = Yes)

Abdominal pain? (0 = No, 1 = Yes)

Output Format
text
========================================
          PREDICTION RESULT
========================================
  Outcome  : JAUNDICE LIKELY DETECTED
  Risk     : 85%

  Symptoms reported:
    - Yellow eyes
    - Yellow skin
    - Dark-coloured urine
    - Fatigue

  *** Please consult a doctor immediately. ***
========================================

DISCLAIMER: This tool is for educational purposes
only. Always consult a qualified medical doctor
for any diagnosis or treatment.
🧠 Model Architecture
Algorithm: Random Forest Classifier

Number of Trees: 10

Random State: 42 (for reproducibility)

Features: 7 clinical parameters

Training Data: 15 synthetic patient cases

📈 Performance Considerations
The current model is trained on synthetic data and is intended for:

Educational purposes

Demonstrating ML concepts

Understanding healthcare ML applications

Not suitable for clinical use without proper validation and real patient data.

🔬 Technical Details
Data Processing
Features are extracted from patient records

Binary encoding for categorical symptoms

Age is treated as a continuous variable

Prediction Pipeline
Collect patient symptoms

Format data for model input

Generate prediction with confidence scores

Display results with symptom breakdown

📝 Code Structure
text
jaundice_predictor.py
├── create_training_data()    # Generates synthetic training data
├── train_model()             # Trains Random Forest classifier
├── get_patient_input()       # Collects patient symptoms
├── display_results()         # Shows prediction and recommendations
└── main()                    # Orchestrates the application flow
⚠️ Disclaimer
This is an educational tool only. The predictions made by this system are based on limited synthetic data and should not be used for actual medical diagnosis. Always consult with qualified healthcare professionals for any health concerns.

🎯 Learning Objectives
This project demonstrates:

Binary classification in healthcare contexts

Feature engineering with mixed data types

Model deployment in a practical application

Risk assessment and probability interpretation

Ethical considerations in medical AI

🔮 Future Improvements
Potential enhancements:

Integration with real patient datasets

Additional clinical features

Model performance metrics and validation

Web interface for easier access

Cross-validation and hyperparameter tuning

Explainable AI features (feature importance visualization)

📄 License
This project is for educational purposes. Please use responsibly and respect medical ethics guidelines.

👥 Contributors
Your Name - Initial work - YourGitHub

🙏 Acknowledgments
Course: Fundamentals of AI and ML

Random Forest implementation from scikit-learn

Synthetic data designed to demonstrate basic jaundice symptoms

                                                                                                                                                    BY ARYAN SHARMA 25BOE10059
