# app.py
from fastapi import FastAPI, UploadFile, File
import shutil
import os
from inference import predict_image

# Classes (must match your training dataset)
CLASS_NAMES = ['Crazing', 'Inclusion', 'Patches', 'Pitted_Surface', 'Rolled_In_Scale', 'Scratches']

# Create uploads folder
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize FastAPI
app = FastAPI(title="Defect Detection API")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Predict
    prediction = predict_image(file_path, CLASS_NAMES)
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",  # module:variable
        host="0.0.0.0",
        port=8000,
        reload=True
    )