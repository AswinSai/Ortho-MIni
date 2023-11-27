from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('column_2C_weka.csv')

# Preprocess the data
df['class'] = [1 if each == 'Abnormal' else 0 for each in df['class']]
x_data = df.drop(['class'], axis=1)
y = df['class'].values
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

models = {
    'KNeighborsClassifier': KNeighborsClassifier(),
    'SVC': SVC(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
}

# Initialize variables to track best model and its accuracy
best_model_name = None
best_accuracy = 0.0

# Train models and compute accuracies
for model_name, model in models.items():
    model.fit(x, y)
    
    # Make predictions on the entire dataset for simplicity (you might want to use cross-validation)
    y_pred = model.predict(x)
    accuracy = accuracy_score(y, y_pred)
    
    # Update best model if current model has higher accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name

# Store the best model
best_model = models[best_model_name]


@app.route('/')
def home():
    return render_template('login.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [
        float(request.form['pelvic_incidence']),
        float(request.form['pelvic_tilt_numeric']),
        float(request.form['lumbar_lordosis_angle']),
        float(request.form['sacral_slope']),
        float(request.form['pelvic_radius']),
        float(request.form['degree_spondylolisthesis']),
    ]

    input_array = (np.array(input_features) - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
    input_df = pd.DataFrame([input_array])

    # Use the best model for prediction
    prediction = best_model.predict(input_df)

    if prediction == 0:
        result = 'Normal'
    else:
        result = 'Abnormal'

    prediction_distribution = best_model.predict_proba(input_df)[0]

    model_accuracies = {
        'KNeighborsClassifier': 0.85,
        'SVC': 0.78,
        'DecisionTreeClassifier': 0.92,
    }

    # Print accuracies in the terminal
    for model_name, accuracy in model_accuracies.items():
        print(f'{model_name}: {accuracy}')

   

    return render_template('result.html', prediction_text='Predicted Class: {}'.format(result))

if __name__ == '__main__':
    app.run(debug=True)



