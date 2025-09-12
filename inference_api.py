#!/usr/bin/env python3
"""
Khet Guard - Unified ML Inference API
Serves disease/pest detection and cattle breed classification models
"""

import os
import json
import base64
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Khet Guard ML Inference API",
    description="Unified API for plant disease/pest detection and cattle breed classification",
    version="1.0.0"
)

# Global model variables
disease_model = None
cattle_model = None
disease_labels = []
cattle_labels = []
pesticide_map = {}

class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    class_id: int
    all_predictions: List[Dict[str, Union[str, float]]]

class DiseasePestResponse(PredictionResponse):
    pesticide_recommendation: Optional[Dict[str, Union[str, List[str]]]] = None
    grad_cam_url: Optional[str] = None

class CattleResponse(PredictionResponse):
    breed_info: Optional[Dict[str, str]] = None

def load_model(model_path: str) -> ort.InferenceSession:
    """Load ONNX model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    providers = ['CPUExecutionProvider']
    if ort.get_device() == 'GPU':
        providers.insert(0, 'CUDAExecutionProvider')
    
    session = ort.InferenceSession(model_path, providers=providers)
    logger.info(f"Loaded model: {model_path}")
    return session

def load_labels(labels_path: str) -> List[str]:
    """Load class labels from JSON file"""
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels not found: {labels_path}")
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    logger.info(f"Loaded {len(labels)} labels from {labels_path}")
    return labels

def load_pesticide_map(map_path: str) -> Dict:
    """Load pesticide recommendation map"""
    if not os.path.exists(map_path):
        logger.warning(f"Pesticide map not found: {map_path}")
        return {}
    
    with open(map_path, 'r', encoding='utf-8') as f:
        pesticide_map = json.load(f)
    logger.info(f"Loaded pesticide map with {len(pesticide_map)} entries")
    return pesticide_map

def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (380, 380)) -> np.ndarray:
    """Preprocess image for model inference"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Transpose to CHW format and add batch dimension
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def postprocess_predictions(logits: np.ndarray, labels: List[str]) -> Tuple[str, float, int, List[Dict[str, float]]]:
    """Postprocess model predictions"""
    # Apply softmax to get probabilities
    probabilities = softmax(logits[0])
    
    # Get top prediction
    class_id = np.argmax(probabilities)
    confidence = float(probabilities[class_id])
    class_name = labels[class_id]
    
    # Get all predictions
    all_predictions = [
        {"class": labels[i], "confidence": float(probabilities[i])}
        for i in range(len(labels))
    ]
    all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
    
    return class_name, confidence, class_id, all_predictions

def softmax(x: np.ndarray) -> np.ndarray:
    """Apply softmax function"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def generate_grad_cam(image: Image.Image, model: ort.InferenceSession, class_id: int) -> str:
    """Generate Grad-CAM visualization (simplified version)"""
    try:
        # For this demo, we'll create a simple heatmap overlay
        # In production, you'd implement proper Grad-CAM
        
        # Create a simple heatmap
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Create a simple gradient heatmap
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        heatmap = np.sin(X * np.pi) * np.cos(Y * np.pi)
        
        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        ax1.imshow(img_array)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Heatmap overlay
        ax2.imshow(img_array)
        ax2.imshow(heatmap, alpha=0.6, cmap='jet')
        ax2.set_title("Grad-CAM Visualization")
        ax2.axis('off')
        
        # Save to temporary file
        output_path = "temp_grad_cam.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating Grad-CAM: {e}")
        return None

def get_pesticide_recommendation(class_name: str) -> Optional[Dict[str, str]]:
    """Get pesticide recommendation for detected pest/disease"""
    # Simple keyword matching for demo
    class_lower = class_name.lower()
    
    if "aphid" in class_lower:
        return {
            "recommended": ["Imidacloprid 17.8% SL"],
            "dosage": "0.3ml per L",
            "safety": "Spray in evening, wear gloves"
        }
    elif "bollworm" in class_lower or "worm" in class_lower:
        return {
            "recommended": ["Spinosad 45% SC"],
            "dosage": "0.5ml per L",
            "safety": "Avoid during flowering stage"
        }
    elif "blight" in class_lower:
        return {
            "recommended": ["Copper Oxychloride 50% WP"],
            "dosage": "2g per L",
            "safety": "Apply during early morning"
        }
    elif "mildew" in class_lower:
        return {
            "recommended": ["Sulfur 80% WP"],
            "dosage": "3g per L",
            "safety": "Do not apply in hot weather"
        }
    else:
        return {
            "recommended": ["Consult local agricultural expert"],
            "dosage": "N/A",
            "safety": "Get proper diagnosis before treatment"
        }

@app.on_event("startup")
async def startup_event():
    """Initialize models and labels on startup"""
    global disease_model, cattle_model, disease_labels, cattle_labels, pesticide_map
    
    try:
        # Load models
        disease_model_path = os.getenv("DISEASE_MODEL", "ml/artifacts/disease_pest/exports/model.onnx")
        cattle_model_path = os.getenv("CATTLE_MODEL", "ml/artifacts/cattle/exports/model.onnx")
        
        disease_model = load_model(disease_model_path)
        cattle_model = load_model(cattle_model_path)
        
        # Load labels
        disease_labels_path = os.getenv("DISEASE_LABELS", "ml/artifacts/disease_pest/labels.json")
        cattle_labels_path = os.getenv("CATTLE_LABELS", "ml/artifacts/cattle/labels.json")
        
        disease_labels = load_labels(disease_labels_path)
        cattle_labels = load_labels(cattle_labels_path)
        
        # Load pesticide map
        pesticide_map_path = os.getenv("PESTICIDE_MAP", "ml/recommender/pesticide_map.json")
        pesticide_map = load_pesticide_map(pesticide_map_path)
        
        logger.info("All models and labels loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "disease_model": disease_model is not None,
            "cattle_model": cattle_model is not None
        },
        "labels_loaded": {
            "disease_labels": len(disease_labels),
            "cattle_labels": len(cattle_labels)
        }
    }

@app.post("/predict/disease_pest", response_model=DiseasePestResponse)
async def predict_disease_pest(
    file: UploadFile = File(...),
    include_grad_cam: bool = False
):
    """Predict plant disease or pest from image"""
    if disease_model is None:
        raise HTTPException(status_code=500, detail="Disease model not loaded")
    
    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        processed_image = preprocess_image(image)
        
        # Run inference
        input_name = disease_model.get_inputs()[0].name
        output_name = disease_model.get_outputs()[0].name
        logits = disease_model.run([output_name], {input_name: processed_image})[0]
        
        # Postprocess predictions
        class_name, confidence, class_id, all_predictions = postprocess_predictions(logits, disease_labels)
        
        # Get pesticide recommendation
        pesticide_recommendation = get_pesticide_recommendation(class_name)
        
        # Generate Grad-CAM if requested
        grad_cam_url = None
        if include_grad_cam:
            grad_cam_path = generate_grad_cam(image, disease_model, class_id)
            if grad_cam_path:
                grad_cam_url = f"/grad_cam/{os.path.basename(grad_cam_path)}"
        
        return DiseasePestResponse(
            class_name=class_name,
            confidence=confidence,
            class_id=class_id,
            all_predictions=all_predictions,
            pesticide_recommendation=pesticide_recommendation,
            grad_cam_url=grad_cam_url
        )
        
    except Exception as e:
        logger.error(f"Error in disease/pest prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/cattle", response_model=CattleResponse)
async def predict_cattle(file: UploadFile = File(...)):
    """Predict cattle breed from image"""
    if cattle_model is None:
        raise HTTPException(status_code=500, detail="Cattle model not loaded")
    
    try:
        # Read and preprocess image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        processed_image = preprocess_image(image)
        
        # Run inference
        input_name = cattle_model.get_inputs()[0].name
        output_name = cattle_model.get_outputs()[0].name
        logits = cattle_model.run([output_name], {input_name: processed_image})[0]
        
        # Postprocess predictions
        class_name, confidence, class_id, all_predictions = postprocess_predictions(logits, cattle_labels)
        
        # Get breed info (placeholder)
        breed_info = {
            "origin": "Indian subcontinent",
            "milk_yield": "Medium to High",
            "characteristics": "Hardy, disease resistant"
        }
        
        return CattleResponse(
            class_name=class_name,
            confidence=confidence,
            class_id=class_id,
            all_predictions=all_predictions,
            breed_info=breed_info
        )
        
    except Exception as e:
        logger.error(f"Error in cattle prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/grad_cam/{filename}")
async def get_grad_cam(filename: str):
    """Serve Grad-CAM visualization"""
    file_path = f"temp_{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Grad-CAM not found")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return RedirectResponse(url="/ui")

@app.get("/ui", response_class=HTMLResponse)
async def ui_page():
    """Simple UI for uploading images and viewing predictions"""
    return """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Khet Guard - Demo UI</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; background: #0f172a; color: #e2e8f0; }
    h1 { margin-bottom: 4px; }
    .card { background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 16px; margin-bottom: 16px; }
    .row { display: flex; flex-wrap: wrap; gap: 16px; }
    .col { flex: 1 1 360px; }
    button { background: #2563eb; color: white; border: 0; padding: 10px 14px; border-radius: 8px; cursor: pointer; }
    button:disabled { background: #475569; cursor: default; }
    input[type=file] { padding: 8px; background: #0b1220; border: 1px solid #1f2937; border-radius: 8px; color: #e2e8f0; }
    img { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #1f2937; }
    pre { white-space: pre-wrap; word-break: break-word; background: #0b1220; padding: 12px; border-radius: 8px; border: 1px solid #1f2937; }
    a { color: #93c5fd; }
  </style>
  <script>
    async function predict(endpointId) {
      const fileInput = document.getElementById(endpointId + '-file');
      const resultEl = document.getElementById(endpointId + '-result');
      const previewEl = document.getElementById(endpointId + '-preview');
      if (!fileInput.files.length) { alert('Choose an image first'); return; }
      const file = fileInput.files[0];
      previewEl.src = URL.createObjectURL(file);
      const form = new FormData();
      form.append('file', file);
      try {
        resultEl.textContent = 'Running inference...';
        const res = await fetch('/predict/' + (endpointId === 'disease' ? 'disease_pest' : 'cattle'), { method: 'POST', body: form });
        if (!res.ok) { throw new Error('HTTP ' + res.status); }
        const data = await res.json();
        resultEl.textContent = JSON.stringify(data, null, 2);
      } catch (e) {
        resultEl.textContent = 'Error: ' + e.message;
      }
    }
  </script>
  </head>
<body>
  <h1>Khet Guard ML Inference API</h1>
  <div style=\"margin-bottom:8px\">Try predictions below or use <a href=\"/docs\">/docs</a>.</div>

  <div class=\"row\">
    <div class=\"col\">
      <div class=\"card\">
        <h3>Plant Disease/Pest</h3>
        <input id=\"disease-file\" type=\"file\" accept=\"image/*\">\n
        <div style=\"margin-top:10px\">\n<button onclick=\"predict('disease')\">Predict Disease/Pest</button></div>
        <div style=\"margin-top:12px\"><img id=\"disease-preview\" alt=\"preview\"></div>
        <h4>Result</h4>
        <pre id=\"disease-result\"></pre>
      </div>
    </div>

    <div class=\"col\">
      <div class=\"card\">
        <h3>Cattle Breed</h3>
        <input id=\"cattle-file\" type=\"file\" accept=\"image/*\">\n
        <div style=\"margin-top:10px\">\n<button onclick=\"predict('cattle')\">Predict Cattle</button></div>
        <div style=\"margin-top:12px\"><img id=\"cattle-preview\" alt=\"preview\"></div>
        <h4>Result</h4>
        <pre id=\"cattle-result\"></pre>
      </div>
    </div>
  </div>

  <div class=\"card\">
    <h3>Status</h3>
    <pre id=\"status\">Loading...</pre>
  </div>

  <script>
    (async () => {
      try { const r = await fetch('/health'); document.getElementById('status').textContent = await r.text(); }
      catch (e) { document.getElementById('status').textContent = 'Error: ' + e.message; }
    })();
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
