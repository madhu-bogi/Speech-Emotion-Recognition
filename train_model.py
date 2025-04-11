import os
import numpy as np
import librosa
import librosa.display
import soundfile
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Define emotions and dataset path
DATASET_PATH = "dataset/"  # Change this to your dataset folder
EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

# Function to extract features
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except:
        return None

# Load dataset
X, y = [], []
for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".wav"):
            emotion = EMOTIONS.get(file.split("-")[2])  # Extract emotion from filename
            if emotion:
                features = extract_features(os.path.join(root, file))
                if features is not None:
                    X.append(features)
                    y.append(emotion)

# Convert to numpy arrays
X, y = np.array(X), np.array(y)

# Encode labels
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train classifier
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Save model and encoder
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

with open("encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)

print("Model training completed and saved!")
