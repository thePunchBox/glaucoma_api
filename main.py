from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os

# Initialize FastAPI app
app = FastAPI(title="Glaucoma Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://glaucoma-frontend.vercel.app/"],  # or ["http://localhost:3000"] for React only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build full path to model file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_model.keras")

# Load your saved model
print(f"🔄 Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Preprocessing function
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))  # same size used during training
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded file
    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Decide class
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

# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
