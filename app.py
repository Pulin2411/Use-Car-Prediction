#!/usr/bin/env python3
import os
import pickle
from flask import Flask, request, jsonify, render_template_string
import pandas as pd

# ---------- Config ----------
DATA_DIR = os.environ.get(
    "DATA_DIR",
    "/content/drive/MyDrive/Project: Used Car Price Prediction using Vehicle Dataset/Car details",
)
# Auto-detect the latest best_model_*.pickle file
MODEL_PATH = os.environ.get("MODEL_PATH", "")
if not MODEL_PATH:
    candidates = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.startswith("best_model_") and f.endswith(".pickle")
    ]
    if not candidates:
        raise FileNotFoundError(
            "No best_model_*.pickle found in DATA_DIR. Train and save the model first."
        )
    # pick the most recently modified
    MODEL_PATH = max(candidates, key=os.path.getmtime)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)  # sklearn Pipeline

# Minimal HTML form
FORM_HTML = """
<!doctype html>
<html>
  <head>
    <title>Used Car Price Predictor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 20px; }
      .card { max-width: 720px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
      label { display: block; margin-top: 12px; font-weight: 600; }
      input, select { width: 100%; padding: 10px; margin-top: 6px; border: 1px solid #ccc; border-radius: 6px; }
      button { margin-top: 16px; padding: 10px 14px; border: none; background: #1363DF; color: #fff; border-radius: 6px; cursor: pointer; }
      .pred { margin-top: 16px; font-size: 1.1rem; font-weight: 600; color: #0b6; }
      .muted { color: #666; font-size: 0.9rem; }
    </style>
  </head>
  <body>
    <div class="card">
      <h2>Used Car Price Predictor</h2>
      <p class="muted">Model: {{ model_name }} (loaded from {{ model_path }})</p>
      <form method="post" action="/predict">
        <label>Year</label>
        <input type="number" name="year" placeholder="e.g., 2015" required />

        <label>Kilometers Driven</label>
        <input type="number" name="km_driven" placeholder="e.g., 60000" required />

        <label>Fuel</label>
        <select name="fuel">
          <option value="Petrol">Petrol</option>
          <option value="Diesel">Diesel</option>
          <option value="CNG">CNG</option>
          <option value="LPG">LPG</option>
          <option value="Electric">Electric</option>
          <option value="Unknown">Unknown</option>
        </select>

        <label>Seller Type</label>
        <select name="seller_type">
          <option value="Individual">Individual</option>
          <option value="Dealer">Dealer</option>
          <option value="Trustmark Dealer">Trustmark Dealer</option>
          <option value="Unknown">Unknown</option>
        </select>

        <label>Transmission</label>
        <select name="transmission">
          <option value="Manual">Manual</option>
          <option value="Automatic">Automatic</option>
          <option value="Unknown">Unknown</option>
        </select>

        <label>Owner</label>
        <select name="owner">
          <option value="First Owner">First Owner</option>
          <option value="Second Owner">Second Owner</option>
          <option value="Third Owner">Third Owner</option>
          <option value="Fourth & Above Owner">Fourth & Above Owner</option>
          <option value="Test Drive Car">Test Drive Car</option>
          <option value="Unknown">Unknown</option>
        </select>

        <button type="submit">Predict</button>
      </form>

      {% if prediction is not none %}
      <div class="pred">Estimated Price: ₹ {{ prediction | round(0) | int }}</div>
      <div class="muted">Note: The model was trained with price target in rupees after standardization heuristics.</div>
      {% endif %}
    </div>
  </body>
</html>
"""

app = Flask(__name__)


def build_features(payload):
    # Derive car_age to match training pipeline logic
    current_year = int(os.environ.get("CURRENT_YEAR", 2025))
    year = int(payload.get("year"))
    car_age = current_year - year

    row = {
        # numeric
        "km_driven": float(payload.get("km_driven")),
        "car_age": float(car_age),
        # categorical
        "fuel": str(payload.get("fuel", "Unknown")),
        "seller_type": str(payload.get("seller_type", "Unknown")),
        "transmission": str(payload.get("transmission", "Unknown")),
        "owner": str(payload.get("owner", "Unknown")),
    }
    return pd.DataFrame([row])


@app.route("/", methods=["GET"])
def index():
    model_name = type(model.named_steps.get("model")).__name__ if hasattr(model, "named_steps") else type(model).__name__
    return render_template_string(
        FORM_HTML,
        prediction=None,
        model_name=model_name,
        model_path=os.path.basename(MODEL_PATH),
    )


@app.route("/predict", methods=["POST"]) 
def predict_form():
    try:
        df = build_features(request.form)
        pred = float(model.predict(df)[0])
        return render_template_string(
            FORM_HTML,
            prediction=pred,
            model_name=type(model.named_steps.get("model")).__name__ if hasattr(model, "named_steps") else type(model).__name__,
            model_path=os.path.basename(MODEL_PATH),
        )
    except Exception as e:
        return render_template_string(
            FORM_HTML + f"<p style='color:#b00'>Error: {e}</p>",
            prediction=None,
            model_name=type(model.named_steps.get("model")).__name__ if hasattr(model, "named_steps") else type(model).__name__,
            model_path=os.path.basename(MODEL_PATH),
        ), 400


@app.route("/api/predict", methods=["POST"]) 
def predict_api():
    # JSON API: accept a single-item or batch list
    try:
        payload = request.get_json(force=True)
        rows = payload if isinstance(payload, list) else [payload]
        dfs = [build_features(row) for row in rows]
        X = pd.concat(dfs, ignore_index=True)
        preds = model.predict(X)
        out = [float(p) for p in preds]
        return jsonify({"predictions": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # For local testing: python app.py
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
