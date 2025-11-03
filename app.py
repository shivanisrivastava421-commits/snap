# app.py
from pathlib import Path
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf

HERE = Path(__file__).resolve().parent

# load model and preprocessing artifacts
MODEL_PATH = HERE / "model.keras"
SCALER_PATH = HERE / "scaler.pkl"
TARGET_ENCODER_PATH = HERE / "target_encoder.pkl"

app = FastAPI()

# Load Keras model
try:
    model = tf.keras.models.load_model(str(MODEL_PATH))
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

# Load scaler and target encoder
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(TARGET_ENCODER_PATH, "rb") as f:
    target_encoder = pickle.load(f)

class InputData(BaseModel):
    features: list  # should match feature ordering used during training

@app.get("/")
def root():
    return {"status": "ok", "message": "Service running"}

@app.post("/predict")
def predict(data: InputData):
    try:
        x = np.array(data.features, dtype=float).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid features: {e}")

    # scale
    try:
        x_scaled = scaler.transform(x)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaler error: {e}")

    # predict
    probs = model.predict(x_scaled)  # shape (1, num_classes)
    pred_idx = int(np.argmax(probs, axis=1)[0])
    pred_label = target_encoder.inverse_transform([pred_idx])[0]
    prob = float(np.max(probs))

    return {"prediction": pred_label, "probability": prob}

