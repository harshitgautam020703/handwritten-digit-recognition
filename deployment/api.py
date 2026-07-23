from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Digit Recognition API", version="1.0")
model = tf.keras.models.load_model('../models/mnist_model.h5')

@app.get("/")
async def root():
    return {"message": "Handwritten Digit Recognition API", "status": "active"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = image.resize((28, 28))
        
        img_array = np.array(image).astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # Predict
        prediction = model.predict(img_array)
        digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100
        
        # Get all class probabilities
        probabilities = {str(i): float(prediction[0][i] * 100) for i in range(10)}
        
        return {
            "digit": digit,
            "confidence": round(confidence, 2),
            "all_probabilities": probabilities
        }
    
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile]):
    results = []
    for file in files:
        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('L')
            image = image.resize((28, 28))
            
            img_array = np.array(image).astype('float32') / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)
            
            prediction = model.predict(img_array)
            results.append({
                "filename": file.filename,
                "digit": int(np.argmax(prediction)),
                "confidence": float(np.max(prediction)) * 100
            })
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    
    return {"results": results}