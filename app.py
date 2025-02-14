import pandas as pd
from joblib import load
from flask import Flask, render_template, request

app = Flask(__name__)

model = load('diabetes_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
   
    try:
        pregnancies = int(request.form['Pregnancies'])
        glucose = int(request.form['Glucose'])
        blood_pressure = int(request.form['BloodPressure'])
        skin_thickness = int(request.form['SkinThickness'])
        insulin = int(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        diabetes_pedigree_function = float(request.form['DiabetesPedigreeFunction'])
        age = int(request.form['Age'])
        
        input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
        
        prediction = model.predict(input_data)
       
        if prediction == 1:
            result = "Diabetic"
        else:
            result = "Not Diabetic"
        
        return render_template('index.html', result=result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
app.run(debug=True, port=8000)