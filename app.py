#Import required libraries
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

#Intialize app
app = Flask(__name__)

#Load the model
with open('Regressor.pkl', 'rb') as f:
    model = pickle.load(f)
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = {
            'Gender': request.form['gender'],
            'Age': float(request.form['age']),
            'Height': float(request.form['height']),
            'Weight': float(request.form['weight']),
            'Duration': float(request.form['duration']),
            'Heart_Rate': float(request.form['heart_rate']),
            'Body_Temp': float(request.form['body_temp'])
            }
        
        input_df = pd.DataFrame([data])
        sqrt_pred = model.predict(input_df)[0]
        predicted_calories = round(sqrt_pred ** 2, 2)
        
        return render_template('index.html', prediction=predicted_calories, input_data=data)
    
    except Exception as e:
        return render_template('index.html', error=str(e))
    
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)

    