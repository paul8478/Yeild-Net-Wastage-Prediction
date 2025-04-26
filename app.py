from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
# import shap

app = Flask(__name__)

# Load models
with open('rice_model.pkl', 'rb') as f:
    rice_model = pickle.load(f)
with open('tomato_model.pkl', 'rb') as f:
    tomato_model = pickle.load(f)

# Ideal values for both crops
RICE_IDEAL = {
    "area (bigha)": (5.5, 6.5), "seeds_shown": (25, 35),
    "water (ml)": (700000, 900000), "PH level": (5.3, 6.5),
    "temperature": (25, 35)
}

TOMATO_IDEAL = {
    "area (bigha)": (5.5, 6.5), "seeds_shown": (17, 27),
    "water (ml)": (4000000, 6000000), "PH level": (6.0, 6.8),
    "temperature": (25, 35)
}

def predict_production(model, inputs):
    input_data = np.array([inputs])
    return model.predict(input_data)[0]

def calculate_net_production(predicted, impacts):
    total_impact = sum(impacts.values())
    net_produced = predicted * (1 - (total_impact / 100))
    net_wastage = predicted - net_produced
    return net_produced, net_wastage

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    crop_type = request.form['crop_type']
    area = float(request.form['area'])
    seeds = float(request.form['seeds'])
    ph = float(request.form['ph'])
    water = float(request.form['water'])
    investment = float(request.form['investment'])
    flood_impact = float(request.form['flood_impact'])
    disease_impact = float(request.form['disease_impact'])
    temperature = float(request.form['temperature'])
    other_damage = float(request.form['other_damage'])

    # Validation ranges
    if crop_type == "rice":
        model = rice_model
        ideal_values = RICE_IDEAL
        min_seeds, max_seeds = area * 8, area * 10
        min_water, max_water = area * 755000, area * 830000
        min_investment, max_investment = area * 8000, area * 10000
        ph_range = (5.0, 8.5)
    else:
        model = tomato_model
        ideal_values = TOMATO_IDEAL
        min_seeds, max_seeds = area * 17, area * 27
        min_water, max_water = area * 4000000, area * 6000000
        min_investment, max_investment = area * 50000, area * 65000
        ph_range = (5.0, 7.5)

    # Input validation
    if not (min_seeds <= seeds <= max_seeds and 
            min_water <= water <= max_water and 
            min_investment <= investment <= max_investment and 
            ph_range[0] <= ph <= ph_range[1] and 
            22 <= temperature < 40):
        return "Invalid input parameters", 400

    # Calculate impacts
    ph_damage = max(0, (ph - (ph_range[1] + ph_range[0])/2) * 8)
    temp_damage = max(0, (temperature - 35) * 8.1)
    
    impacts = {
        "flood": flood_impact,
        "disease": disease_impact,
        "temperature": temp_damage,
        "ph": ph_damage,
        "other": other_damage
    }

    # Prediction
    inputs = [area, seeds, ph, water, investment]
    predicted = predict_production(model, inputs)
    net_produced, net_wastage = calculate_net_production(predicted, impacts)

    # Feature ranking (simplified version)
    user_input = {"area (bigha)": area, "seeds_shown": seeds, "PH level": ph, 
                 "water (ml)": water, "temperature": temperature}
    
    deviation_scores = {
        f: max(0, abs(user_input[f] - ideal_values[f][0]) / ideal_values[f][0] * 100) 
        if user_input[f] < ideal_values[f][0] 
        else max(0, abs(user_input[f] - ideal_values[f][1]) / ideal_values[f][1] * 100) 
        for f in user_input
    }

    return render_template('result.html', 
                         crop_type=crop_type,
                         predicted=predicted,
                         net_produced=net_produced,
                         net_wastage=net_wastage,
                         deviations=deviation_scores)

if __name__ == '__main__':
    app.run(debug=True) 