from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# === Load all models and preprocessing objects ===
# Diabetes
diabetes_model = pickle.load(open("svm_model.pkl", "rb"))
diabetes_scaler = pickle.load(open("scaler.pkl", "rb"))
diabetes_features = [
    "Age", "Gender", "BMI", "SBP", "DBP", "FPG", "Chol",
    "HDL", "LDL", "BUN", "CCR", "FFPG", "smoking", "drinking", "family_histroy"
]

# CHD
chd_model = joblib.load("CHdxgboost_model.pkl")
chd_scaler = joblib.load("CHDscaler.pkl")
chd_features = [
    'male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',
    'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol',
    'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
]

# Stroke
stroke_model = joblib.load("stroke_predictor.pkl")
stroke_encoders = joblib.load("label_encoders.pkl")
stroke_features = [
    'gender', 'age', 'hypertension', 'heart_disease',
    'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
]

# === Routes ===
@app.route('/')
def home():
    return "Welcome to the Unified ML Prediction API! Endpoints: /predict/diabetes, /predict/chd, /predict/stroke"

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    data = request.get_json()
    try:
        input_data = [float(data[feature]) for feature in diabetes_features]
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = diabetes_scaler.transform(input_array)

        prediction = diabetes_model.predict(input_scaled)[0]
        confidence = diabetes_model.predict_proba(input_scaled)[0][1]

        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        return jsonify({
            "prediction": result,
            "confidence": f"{confidence:.2%}"
        })
    except KeyError as e:
        return jsonify({"error": f"Missing field in input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/chd', methods=['POST'])
def predict_chd():
    try:
        data = request.get_json()

        if not all(feature in data for feature in chd_features):
            return jsonify({"error": "Missing feature(s). Required: " + ", ".join(chd_features)}), 400

        input_values = [float(data[feature]) for feature in chd_features]
        input_array = np.array(input_values).reshape(1, -1)
        scaled_input = chd_scaler.transform(input_array)

        prediction = chd_model.predict(scaled_input)[0]
        probability = chd_model.predict_proba(scaled_input)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": round(float(probability), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/stroke', methods=['POST'])
def predict_stroke():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        print(data)
        if not all(col in input_df.columns for col in stroke_features):
            return jsonify({"error": "Missing required features"}), 400

        input_df = input_df[stroke_features]

        for col in stroke_encoders:
            if col in input_df.columns:
                le = stroke_encoders[col]
                input_df[col] = le.transform(input_df[col])

        prediction = stroke_model.predict(input_df)[0]
        result = "Stroke" if prediction == 1 else "No Stroke"

        return jsonify({
            "prediction": int(prediction),
            "result": result
        })
    except Exception as e:
        print("🔥 Internal Error:", e)
        return jsonify({"error": str(e)}), 500

# Run the unified app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
