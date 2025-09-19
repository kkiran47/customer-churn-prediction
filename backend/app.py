from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load models with joblib (safe across scikit-learn versions)
models = {
    "Logistic Regression": joblib.load("models/log_reg.pkl"),
    "Decision Tree": joblib.load("models/decision_tree.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "Support Vector Machine": joblib.load("models/svm.pkl")
}

# Load scaler & encoders
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")

def preprocess_input(data):
    df = pd.DataFrame([data])

    # Apply label encoding
    for col, le in encoders.items():
        if col in df:
            df[col] = le.transform(df[col])

    # Scale features
    df_scaled = scaler.transform(df)
    return df_scaled

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "Backend is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        X = preprocess_input(data)

        results = []
        for name, model in models.items():
            proba = model.predict_proba(X)[0][1]
            pred = int(proba >= 0.5)
            results.append({
                "model": name,
                "prediction": pred,
                "confidence": round(proba if pred == 1 else 1 - proba, 3),
                "churn_probability": round(proba, 3)
            })

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Render expects 0.0.0.0 for external access, and PORT from env
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
