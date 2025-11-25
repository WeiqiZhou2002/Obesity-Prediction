import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def load_and_process_data(filepath):
    df = pd.read_csv(filepath)

    df['BMI'] = df['Weight'] / (df['Height'] ** 2)

    # Discretize BMI into BMI_bin (Intermediate Node)
    # Using standard WHO categories: Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obese (>30)
    # We use a large upper bound (100) to catch everything above 30
    df['BMI_bin'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100],
                           labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

    # Discretize Age into Age_bin
    df['Age_bin'] = pd.qcut(df['Age'], q=3, labels=['Young', 'Middle', 'Senior'])

    # Assuming FAF is 0-3 days/frequency
    df['FAF_bin'] = pd.cut(df['FAF'], bins=[-0.1, 1, 2, 4], labels=['Low', 'Medium', 'High'])

    # FCVC (Vegetables) -> Round to integer (1, 2, 3)
    df['FCVC'] = df['FCVC'].round().astype(int).astype(str)

    # NCP (Meals per day) -> Round to integer (1, 2, 3, 4)
    df['NCP'] = df['NCP'].round().astype(int).astype(str)

    # TUE (Tech usage) -> Round to integer (0, 1, 2)
    df['TUE'] = df['TUE'].round().astype(int).astype(str)

    # CH2O (Water) -> Round to integer (1, 2, 3)
    df['CH2O_bin'] = df['CH2O'].round().astype(int).astype(str)

    cols_to_use = [
        'Gender', 'Age_bin', 'family_history',  # Demographics
        'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'FAF_bin', 'TUE', 'CALC', 'MTRANS',  # Lifestyle
        'BMI_bin', 'Obesity'  # Intermediate & Target
    ]

    data = df[cols_to_use].astype(str)


    return data


def build_and_train_model(train_data):
    model = DiscreteBayesianNetwork([
        # 1. Demographics/History -> Lifestyle & Diet
        ('Gender', 'FAF_bin'),
        ('Age_bin', 'FAVC'),
        ('family_history', 'BMI_bin'),

        # 2. Lifestyle/Diet -> BMI_bin
        ('FAVC', 'BMI_bin'),
        ('FCVC', 'BMI_bin'),
        ('FAF_bin', 'BMI_bin'),
        ('SMOKE', 'BMI_bin'),
        ('NCP', 'BMI_bin'),
        ('CAEC', 'BMI_bin'),
        # ('CH2O_bin', 'BMI_bin'),
        ('FAF_bin', 'BMI_bin'),
        ('TUE', 'BMI_bin'),
        ('CALC', 'BMI_bin'),
        ('MTRANS', 'BMI_bin'),

        # 3. BMI_bin -> Obesity (Target)
        ('BMI_bin', 'Obesity')

    ])

# Parameter Learning using MLE
# Since data is complete/fully observed, we do not need EM.
    print("Structure defined. Learning CPTs using MLE...")
    model.fit(train_data, estimator=MaximumLikelihoodEstimator)

    # Check if the model is valid (no cycles, everything connected)
    print(f"Model Check: {model.check_model()}")

    return model


def evaluate_model(model, test_data):
    inference = VariableElimination(model)

    y_true = test_data['Obesity'].tolist()
    y_pred = []

    print("Running Inference on Test Set...")
    for i, row in test_data.iterrows():
        evidence = row.drop('Obesity').to_dict()
        count=0
        total = len(test_data)

        # Compute P(Obesity | features)
        try:
            result = inference.map_query(variables=['Obesity'], evidence=evidence, show_progress=False)
            y_pred.append(result['Obesity'])
        except Exception as e:
            print(f"Error on row {i}: {e}")
            print(f"Evidence was: {evidence}")
            return

        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{total} records...")

    valid_indices = [i for i, x in enumerate(y_pred) if x is not None]
    clean_true = [y_true[i] for i in valid_indices]
    clean_pred = [y_pred[i] for i in valid_indices]

    acc = accuracy_score(clean_true, clean_pred)
    f1 = f1_score(clean_true, clean_pred, average='weighted')

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

def run_what_if_analysis(model):
    infer = VariableElimination(model)

    print("\n--- What-If Analysis ---")

    # Scenario: High Physical Activity vs Low Physical Activity
    print("Query: Effect of Physical Activity (FAF_bin) on Obesity risk")

    q_high_activity = infer.query(variables=['Obesity'], evidence={'FAF_bin': 'High'})
    q_low_activity = infer.query(variables=['Obesity'], evidence={'FAF_bin': 'Low'})

    print(f"P(Obesity | High Activity):\n{q_high_activity}")
    print(f"P(Obesity | Low Activity):\n{q_low_activity}")


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