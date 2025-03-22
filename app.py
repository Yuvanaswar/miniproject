import os
import pandas as pd
import pickle
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH = "credit_card_fraud_model.pkl"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join("uploads", secure_filename(file.filename))
    file.save(filepath)

    # Load trained model
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not trained yet. Please upload a dataset to train the model first."})

    with open(MODEL_PATH, "rb") as f:
        model, scaler = pickle.load(f)

    df = pd.read_csv(filepath)

    required_features = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                         "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
                         "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]

    if not set(required_features).issubset(df.columns):
        return jsonify({"error": "CSV file missing required features"}), 400

    X = df[required_features]
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    df["Fraud"] = predictions
    total = len(df)
    fraud_count = df["Fraud"].sum()

    return render_template("result.html", total=total, fraud=fraud_count)

if __name__ == "__main__":
    app.run(debug=True)
