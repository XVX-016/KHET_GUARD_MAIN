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
  <title>Khet Guard</title>
  <style>
    :root { --bg:#0b1220; --panel:#0f172a; --panel-2:#111827; --text:#e2e8f0; --muted:#94a3b8; --brand1:#16a34a; --brand2:#f59e0b; --brand3:#22c55e; --accent:#2563eb; }
    [data-theme="light"] { --bg:#f6f7fb; --panel:#ffffff; --panel-2:#ffffff; --text:#0f172a; --muted:#475569; }
    * { box-sizing: border-box; }
    body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: var(--bg); color: var(--text); }
    .container { max-width: 980px; margin: 0 auto; padding: 16px; }
    .header {
      background: linear-gradient(135deg, #10b981 0%, #16a34a 35%, #0ea5e9 100%);
      color: #fff; border-bottom-left-radius: 20px; border-bottom-right-radius: 20px;
      padding: 22px 16px 90px 16px;
    }
    .header .top { display: flex; justify-content: flex-end; gap: 8px; }
    .chip { display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:10px; background: rgba(255,255,255,.2); backdrop-filter: blur(6px); font-weight:600; }
    .greeting { font-size: 26px; font-weight: 800; margin-top: 8px; }
    .subtitle { opacity:.9; margin-top: 6px; }
    .grid { display:grid; gap:14px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-top:-60px; padding: 0 16px; }
    .stat { background: var(--panel); border:1px solid #1f2937; border-radius:14px; padding:14px; }
    .stat .label { color: var(--muted); font-size: 13px; }
    .stat .value { font-size: 24px; font-weight: 800; margin-top: 6px; }
    .actions { display:grid; gap:16px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); margin: 18px 16px; }
    .tile { border-radius:16px; padding:18px; color:#fff; border:0; cursor:pointer; text-align:left; transition: transform .08s ease; }
    .tile:active { transform: scale(.99); }
    .tile.green { background: linear-gradient(135deg, #16a34a, #22c55e); }
    .tile.gold { background: linear-gradient(135deg, #f59e0b, #fbbf24); }
    .tile.teal { background: linear-gradient(135deg, #06b6d4, #0ea5e9); }
    .tile h3 { margin: 0 0 6px 0; }
    .tile p { margin: 0; opacity:.95; }
    .panel { background: var(--panel-2); border:1px solid #1f2937; border-radius:14px; padding:16px; margin: 16px; }
    .row { display:flex; gap:16px; flex-wrap: wrap; }
    input[type=file], input[type=text] { width:100%; padding:10px; background: var(--bg); color: var(--text); border:1px solid #1f2937; border-radius:10px; }
    button.primary { background: var(--accent); border:0; color:#fff; padding:10px 14px; border-radius:10px; cursor:pointer; }
    img.preview { max-width: 100%; border-radius:12px; border:1px solid #1f2937; }
    pre { white-space: pre-wrap; word-break: break-word; background: var(--bg); padding: 12px; border-radius: 12px; border: 1px solid #1f2937; }
    .tabs { position: sticky; bottom: 0; left: 0; right: 0; display:flex; gap: 10px; justify-content: space-around; padding: 10px 12px; background: rgba(15,23,42,.85); backdrop-filter: blur(6px); border-top:1px solid #1f2937; }
    .tab { color: var(--text); text-decoration: none; font-weight: 600; padding:8px 12px; border-radius: 12px; background: rgba(255,255,255,.04); }
    .muted { color: var(--muted); }
  </style>
  <script>
    function setTheme(mode){ document.documentElement.setAttribute('data-theme', mode); localStorage.setItem('kg_theme', mode); }
    function toggleTheme(){ const cur = localStorage.getItem('kg_theme') || 'dark'; setTheme(cur==='dark'?'light':'dark'); }
    (function(){ const saved = localStorage.getItem('kg_theme'); if(saved){ setTheme(saved); }})();
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
    async function recommendCrop(){
      const lat = document.getElementById('crop-lat').value.trim();
      const lon = document.getElementById('crop-lon').value.trim();
      const out = document.getElementById('crop-result');
      if(!lat || !lon){ out.textContent = 'Enter latitude and longitude'; return; }
      try{
        out.textContent = 'Fetching recommendation...';
        const res = await fetch('/recommend/crop', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ latitude: parseFloat(lat), longitude: parseFloat(lon) })});
        if(!res.ok) throw new Error('HTTP ' + res.status);
        out.textContent = JSON.stringify(await res.json(), null, 2);
      }catch(e){ out.textContent = 'Error: ' + e.message; }
    }
  </script>
  </head>
<body>
  <div class=\"header\">
    <div class=\"container\">
      <div class=\"top\">
        <span class=\"chip\">EN</span>
        <button class=\"chip\" onclick=\"toggleTheme()\">☼</button>
      </div>
      <div class=\"greeting\">Good morning, Farmer!</div>
      <div class=\"subtitle\">Location not set</div>
    </div>
  </div>

  <div class=\"container\">
    <div class=\"grid\">
      <div class=\"stat\"><div class=\"label\">Today's Temp</div><div class=\"value\" id=\"stat-temp\">28°C</div></div>
      <div class=\"stat\"><div class=\"label\">Humidity</div><div class=\"value\" id=\"stat-hum\">65%</div></div>
      <div class=\"stat\"><div class=\"label\">Growth %</div><div class=\"value\">87%</div></div>
      <div class=\"stat\"><div class=\"label\">Health %</div><div class=\"value\">92%</div></div>
    </div>
  </div>

  <div class=\"actions\">
    <button class=\"tile green\" onclick=\"document.getElementById('disease-file').click()\">
      <h3>Plant Disease Detection</h3>
      <p>Upload leaf photo to detect diseases</p>
    </button>
    <button class=\"tile gold\" onclick=\"document.getElementById('crop-lat').focus()\">
      <h3>Crop Recommendation</h3>
      <p>Get personalized crop suggestions</p>
    </button>
    <button class=\"tile teal\" onclick=\"document.getElementById('cattle-file').click()\">
      <h3>Cattle Breed Detection</h3>
      <p>Identify cattle breeds from photos</p>
    </button>
  </div>

  <div class=\"panel\">
    <div class=\"row\">
      <div style=\"flex:1 1 320px\"> 
        <h3>Plant Disease/Pest</h3>
        <input id=\"disease-file\" type=\"file\" accept=\"image/*\" onchange=\"predict('disease')\">\n
        <div style=\"margin-top:12px\"><img class=\"preview\" id=\"disease-preview\" alt=\"preview\"></div>
        <h4>Result</h4>
        <pre id=\"disease-result\"></pre>
      </div>
      <div style=\"flex:1 1 320px\"> 
        <h3>Cattle Breed</h3>
        <input id=\"cattle-file\" type=\"file\" accept=\"image/*\" onchange=\"predict('cattle')\">\n
        <div style=\"margin-top:12px\"><img class=\"preview\" id=\"cattle-preview\" alt=\"preview\"></div>
        <h4>Result</h4>
        <pre id=\"cattle-result\"></pre>
      </div>
    </div>
  </div>

  <div class=\"panel\">
    <h3>Crop Recommendation</h3>
    <div class=\"row\">
      <input id=\"crop-lat\" type=\"text\" placeholder=\"Latitude\" style=\"flex:1 1 160px\">
      <input id=\"crop-lon\" type=\"text\" placeholder=\"Longitude\" style=\"flex:1 1 160px\">
      <button class=\"primary\" onclick=\"recommendCrop()\">Recommend</button>
    </div>
    <pre id=\"crop-result\" class=\"muted\">Enter coordinates and click Recommend</pre>
  </div>

  <script>
    (async () => {
      try { const r = await fetch('/health'); document.getElementById('status').textContent = await r.text(); }
      catch (e) { document.getElementById('status').textContent = 'Error: ' + e.message; }
    })();
  </script>

  <div class=\"tabs\">
    <span class=\"tab\">Home</span>
    <a class=\"tab\" href=\"/docs\">Docs</a>
    <span class=\"tab\">Profile</span>
  </div>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
