# src/app.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_path = './models/best_model.joblib'
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json()
    # Convert JSON data to DataFrame
    df = pd.DataFrame(data)
    # Make prediction
    prediction = model.predict(df)
    # Return prediction as JSON response
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)
