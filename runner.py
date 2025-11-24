import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# ==========================================
# 1. PREPROCESSING & DISCRETIZATION [cite: 67, 68]
# ==========================================
# ==========================================
# 1. PREPROCESSING & DISCRETIZATION (UPDATED)
# ==========================================
def load_and_process_data(filepath):
    df = pd.read_csv(filepath)

    # ---------------------------------------------------------
    # NEW STEP: Calculate BMI
    # Formula: BMI = Weight (kg) / Height (m)^2
    # NOTE: Ensure your dataset's 'Height' is in meters (e.g., 1.75).
    # If it is in cm (e.g., 175), divide by 100 first.
    # ---------------------------------------------------------
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)

    # Example: Discretize BMI into BMI_bin (Intermediate Node)
    # Using standard WHO categories: Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obese (>30)
    # We use a large upper bound (100) to catch everything above 30
    df['BMI_bin'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100],
                           labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

    # Example: Discretize Age into Age_bin
    df['Age_bin'] = pd.qcut(df['Age'], q=3, labels=['Young', 'Middle', 'Senior'])

    # Example: Discretize Physical Activity (FAF)
    # Assuming FAF is 0-3 days/frequency
    df['FAF_bin'] = pd.cut(df['FAF'], bins=[-0.1, 1, 2, 4], labels=['Low', 'Medium', 'High'])

    # Ensure all columns used in the BN are strings/categories (Discrete)
    cols_to_use = [
        'Gender', 'Age_bin', 'family_history',  # Demographics
        'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'FAF_bin', 'TUE', 'CALC', 'MTRANS',  # Lifestyle
        'BMI_bin', 'Obesity'  # Intermediate & Target
    ]

    # Filter and convert to string to ensure discrete treatment by pgmpy
    data = df[cols_to_use].astype(str)


    return data


# ==========================================
# 2. DEFINE STRUCTURE & LEARN PARAMETERS [cite: 34, 45]
# ==========================================
def build_and_train_model(train_data):
    # Define the structure based on your Domain-Informed Structure section
    model = DiscreteBayesianNetwork([
        # 1. Demographics/History -> Lifestyle & Diet [cite: 37]
        ('Gender', 'FAF_bin'),
        ('Age_bin', 'FAVC'),  # Example connection
        ('family_history', 'BMI_bin'),  # History affects BMI [cite: 38]

        # 2. Lifestyle/Diet -> BMI_bin [cite: 39]
        ('FAVC', 'BMI_bin'),
        ('FAF_bin', 'BMI_bin'),
        ('SMOKE', 'BMI_bin'),

        # 3. BMI_bin -> Obesity (Target) [cite: 40, 81]
        ('BMI_bin', 'Obesity')

    ])

# Parameter Learning using MLE
# Since data is complete/fully observed, we do not need EM.
    print("Structure defined. Learning CPTs using MLE...")
    model.fit(train_data, estimator=MaximumLikelihoodEstimator)

    # # Optional: Check if the model is valid (no cycles, everything connected)
    # print(f"Model Check: {model.check_model()}")

    return model


# ==========================================
# 3. INFERENCE & EVALUATION [cite: 47, 49]
# ==========================================
def evaluate_model(model, test_data):
    inference = VariableElimination(model)

    y_true = test_data['Obesity'].tolist()
    y_pred = []

    print("Running Inference on Test Set...")
    # Iterating through test rows (Note: pgmpy also has predict() method for batch processing)
    for i, row in test_data.iterrows():
        # Remove target from evidence
        evidence = row.drop('Obesity').to_dict()

        # Compute P(Obesity | features) [cite: 48]
        # We query the 'Obesity' node given the evidence of other nodes
        try:
            result = inference.map_query(variables=['Obesity'], evidence=evidence, show_progress=False)
            y_pred.append(result['Obesity'])
        except:
            y_pred.append(None)  # Handle edge cases

    # Calculate Metrics [cite: 89]
    # Filter out Nones if any failed
    valid_indices = [i for i, x in enumerate(y_pred) if x is not None]
    clean_true = [y_true[i] for i in valid_indices]
    clean_pred = [y_pred[i] for i in valid_indices]

    acc = accuracy_score(clean_true, clean_pred)
    f1 = f1_score(clean_true, clean_pred, average='weighted')

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")


# ==========================================
# 4. WHAT-IF ANALYSIS [cite: 17, 49]
# ==========================================
def run_what_if_analysis(model):
    infer = VariableElimination(model)

    print("\n--- What-If Analysis ---")

    # Scenario: High Physical Activity vs Low Physical Activity
    print("Query: Effect of Physical Activity (FAF_bin) on Obesity risk")

    # Fix other variables (optional) or leave them marginal
    q_high_activity = infer.query(variables=['Obesity'], evidence={'FAF_bin': 'High'})
    q_low_activity = infer.query(variables=['Obesity'], evidence={'FAF_bin': 'Low'})

    print(f"P(Obesity | High Activity):\n{q_high_activity}")
    print(f"P(Obesity | Low Activity):\n{q_low_activity}")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    df = load_and_process_data('Obesity_prediction.csv')

    # 2. Split Data
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # 3. Train
    model = build_and_train_model(train)

    # 4. Evaluate
    evaluate_model(model, test)

    # 5. Interpret
    run_what_if_analysis(model)
    pass