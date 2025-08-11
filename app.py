#!/usr/bin/env python3
"""
Flask web application for House Price Prediction UI
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from house_price_prediction import HousePricePredictor
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variable to store the model
model = None

def load_model():
    """Load and train the model on startup"""
    global model
    print("Loading and training the model...")
    model = HousePricePredictor()
    model.load_data('train.csv')
    model.preprocess_data()
    model.train_model()
    print("Model training completed!")

def preprocess_input(data):
    """Preprocess user input data for prediction"""
    try:
        # Create a DataFrame with the input data
        input_df = pd.DataFrame([data])
        
        # Preprocess the data using the same pipeline
        processed_data = model.preprocess_data(input_df, is_training=False)
        
        return processed_data
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise e

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)[0]
        formatted_prediction = f"${prediction:,.2f}"
        return jsonify({'success': True, 'prediction': formatted_prediction, 'raw_prediction': prediction})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    load_model()
    print("Starting Flask application...")
    print("Open your browser and go to: http://localhost:8080")
    app.run(debug=True, host='0.0.0.0', port=8080)
