import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np



def create_training_data():
    data = [
        # [bilirubin, yellow_eyes, yellow_skin, age, dark_urine, fatigue, abdominal_pain, label]
        [0, 0, 0, 25, 0, 0, 0, 0],   
        [1, 1, 1, 30, 1, 1, 1, 1],   
        [0, 0, 0, 40, 0, 0, 0, 0],
        [1, 0, 0, 35, 0, 0, 0, 0],
        [1, 1, 0, 28, 1, 1, 0, 1],
        [0, 0, 0, 50, 0, 0, 0, 0],
        [1, 1, 1, 45, 1, 1, 1, 1],
        [0, 0, 0, 32, 0, 0, 0, 0],
        [1, 0, 1, 29, 1, 0, 1, 1],
        [0, 0, 0, 38, 0, 0, 0, 0],
        [1, 1, 1, 33, 1, 1, 1, 1],
        [0, 0, 0, 42, 0, 1, 0, 0],   
        [1, 0, 0, 27, 0, 0, 0, 0],
        [0, 0, 0, 55, 0, 0, 0, 0],
        [1, 1, 1, 36, 1, 1, 1, 1],
    ]
    return data




def train_model(data):
    X = [record[:7] for record in data]   # Features: first 7 columns
    y = [record[7]  for record in data]   # Label:    last column

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model



def get_patient_input():
    print("Enter patient details (answer each question with 0 for No or 1 for Yes):\n")

    def ask_binary(prompt):
        while True:
            try:
                val = int(input(prompt))
                if val in (0, 1):
                    return val
                print("  Please enter 0 or 1.")
            except ValueError:
                print("  Invalid input. Enter 0 or 1.")

    def ask_age():
        while True:
            try:
                age = int(input("Age (years): "))
                if 0 < age < 120:
                    return age
                print("  Please enter a realistic age.")
            except ValueError:
                print("  Invalid input. Enter a number.")

    bilirubin      = ask_binary("Elevated bilirubin level? (0=No, 1=Yes): ")
    yellow_eyes    = ask_binary("Yellow eyes?              (0=No, 1=Yes): ")
    yellow_skin    = ask_binary("Yellow skin?              (0=No, 1=Yes): ")
    age            = ask_age()
    dark_urine     = ask_binary("Dark-coloured urine?      (0=No, 1=Yes): ")
    fatigue        = ask_binary("Fatigue or weakness?      (0=No, 1=Yes): ")
    abdominal_pain = ask_binary("Abdominal pain?           (0=No, 1=Yes): ")

    return [bilirubin, yellow_eyes, yellow_skin, age, dark_urine, fatigue, abdominal_pain]




def display_results(patient_data, model):
    """Runs prediction and prints a formatted result for the patient."""
    features      = [patient_data]
    prediction    = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    risk_pct      = probabilities[1] * 100

    symptom_labels = {
        0: ("Elevated bilirubin", patient_data[0]),
        1: ("Yellow eyes",        patient_data[1]),
        2: ("Yellow skin",        patient_data[2]),
        4: ("Dark-coloured urine",patient_data[4]),
        5: ("Fatigue",            patient_data[5]),
        6: ("Abdominal pain",     patient_data[6]),
    }

    print("\n" + "=" * 40)
    print("          PREDICTION RESULT")
    print("=" * 40)

    if prediction == 1:
        print(f"  Outcome  : JAUNDICE LIKELY DETECTED")
        print(f"  Risk     : {risk_pct:.0f}%")
        print()
        print("  Symptoms reported:")
        for _, (label, present) in symptom_labels.items():
            if present:
                print(f"    - {label}")
        print()
        print("  *** Please consult a doctor immediately. ***")
    else:
        print(f"  Outcome  : NO JAUNDICE DETECTED")
        print(f"  Risk     : {risk_pct:.0f}%")
        print()
        print("  The patient appears healthy based on the")
        print("  information provided. Continue routine")
        print("  health checkups.")

    print("=" * 40)
    print()
    print("DISCLAIMER: This tool is for educational purposes")
    print("only. Always consult a qualified medical doctor")
    print("for any diagnosis or treatment.")
    print()




def main():
    print("=" * 40)
    print("      JAUNDICE PREDICTION SYSTEM")
    print("    Fundamentals of AI and ML")
    print("=" * 40)
    print()

    # Train model
    data  = create_training_data()
    model = train_model(data)

    # Collect input
    patient = get_patient_input()

    # Show prediction
    display_results(patient, model)


if __name__ == "__main__":
    main()
