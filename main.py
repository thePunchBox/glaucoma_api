from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps
import numpy as np
import io
import os
import hashlib

# Initialize FastAPI app
app = FastAPI(title="Glaucoma Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] for React only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build full path to model file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_model.keras")

# Helper: md5 checksum of file
def md5_of_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# Load your saved model
print(f"🔄 Loading model from: {MODEL_PATH}")
print(f"🔢 Model checksum: {md5_of_file(MODEL_PATH)}")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

IMG_SIZE = (224, 224)

# Preprocessing function (same as in training)
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")  # handle EXIF rotation
    img = img.resize(IMG_SIZE)
    img_array = img_to_array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------- Normal Prediction Endpoint --------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(400, "Please upload an image file.")

    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        result = "Glaucoma"
        confidence = float(prediction) * 100
    else:
        result = "Normal"
        confidence = (1 - float(prediction)) * 100

    return {
        "prediction": result,
        "confidence": f"{confidence:.2f}%"
    }

# -------- Debugging Endpoint --------
@app.post("/predict-debug")
async def predict_debug(file: UploadFile = File(...)):
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(400, "Please upload an image file.")

    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    stats = {
        "shape": img_array.shape,
        "dtype": str(img_array.dtype),
        "min": float(np.min(img_array)),
        "max": float(np.max(img_array)),
        "mean": float(np.mean(img_array)),
        "sum": float(np.sum(img_array)),
    }

    preds = model.predict(img_array)
    prob = float(preds[0][0])
    predicted = "Glaucoma" if prob > 0.5 else "Normal"

    return {
        "model_md5": md5_of_file(MODEL_PATH),
        "stats": stats,
        "raw_prediction": preds.tolist(),
        "probability": prob,
        "predicted_label": predicted,
    }

# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
