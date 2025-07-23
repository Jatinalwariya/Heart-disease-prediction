from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('Models/random_forest_model.pkl')
scaler = joblib.load('Models/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        #Get form data
        input_data = []

        # Add inputs in exact order matching your training data
        input_data.append(float(request.form['age']))
        input_data.append(int(request.form['sex']))
        input_data.append(float(request.form['trestbps']))
        input_data.append(float(request.form['chol']))
        input_data.append(int(request.form['fbs']))
        input_data.append(int(request.form['restecg']))
        input_data.append(float(request.form['thalach']))
        input_data.append(int(request.form['exang']))
        input_data.append(float(request.form['oldpeak']))
        input_data.append(int(request.form['ca']))

        # One-hot encoded inputs (example)
        input_data.append(int(request.form['cp_1']))
        input_data.append(int(request.form['cp_2']))
        input_data.append(int(request.form['cp_3']))
        input_data.append(int(request.form['thal_1']))
        input_data.append(int(request.form['thal_2']))
        input_data.append(int(request.form['thal_3']))
        input_data.append(int(request.form['slope_1']))
        input_data.append(int(request.form['slope_2']))

        final_input = np.array([input_data])

        final_scaled = scaler.transform(final_input)

        prediction = model.predict(final_scaled)

        result = "Heart Disease Detected" if prediction[0]==1 else "Healthy"

        return render_template('index.html', prediction=result)
    
if __name__ == '__main__':
    app.run(debug=True)