import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the folder to save uploaded CSV files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL_PATH = "credit_card_fraud_model.pkl"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
    file.save(filepath)

    # Train the model using the uploaded file
    message = train_model(filepath)

    return jsonify({"message": message})

def train_model(filepath):
    try:
        # Load dataset
        df = pd.read_csv(filepath)

        # Ensure proper feature selection
        required_features = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", 
                             "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", 
                             "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount", "Class"]
        
        if not set(required_features).issubset(df.columns):
            return "CSV file missing required features"

        X = df.drop(columns=["Class"])  # Features
        y = df["Class"]  # Labels (0 = Legitimate, 1 = Fraudulent)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Save trained model and scaler
        with open(MODEL_PATH, "wb") as f:
            pickle.dump((model, scaler), f)

        return "Model trained successfully!"

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
