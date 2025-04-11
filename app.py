from flask import Flask, request, jsonify
import librosa
import numpy as np
import pickle
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load trained models
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except:
        return None

@app.route('/predict', methods=['POST'])
def predict_emotion():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    features = extract_features(file_path)
    if features is None:
        return jsonify({"error": "Invalid audio file"}), 400

    features = scaler.transform([features])
    prediction = model.predict(features)
    emotion = encoder.inverse_transform(prediction)[0]

    return jsonify({"emotion": emotion})

if __name__ == '__main__':
    app.run(debug=True)

    

