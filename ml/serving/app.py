"""
Production FastAPI serving microservice for Khet Guard ML models.
Supports plant disease detection, cattle breed classification, and crop recommendation.
"""

import os
import json
import logging
import time
from pathlib import Path
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any, Union
import asyncio
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import httpx
import redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.redis import RedisIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
MODEL_INFERENCE_TIME = Histogram('model_inference_seconds', 'Model inference time', ['model_type'])
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')
MODEL_LOAD_TIME = Gauge('model_load_time_seconds', 'Time to load model', ['model_type'])

# Security
security = HTTPBearer()

# Redis client for caching
redis_client = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, db=0)

# Global model cache
models_cache = {}

class ModelMetadata(BaseModel):
    """Model metadata schema."""
    model_type: str
    version: str
    num_classes: int
    class_mapping: Dict[str, int]
    input_shape: List[int]
    preprocessing: Dict[str, Any]
    postprocessing: Dict[str, Any]

class PredictionRequest(BaseModel):
    """Prediction request schema."""
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    return_uncertainty: bool = False
    return_gradcam: bool = False

class PredictionResponse(BaseModel):
    """Prediction response schema."""
    predictions: List[Dict[str, Any]]
    model_info: Dict[str, Any]
    processing_time: float
    uncertainty: Optional[Dict[str, Any]] = None
    gradcam_url: Optional[str] = None

class CropRecommendationRequest(BaseModel):
    """Crop recommendation request schema."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    return_uncertainty: bool = False

class CropRecommendationResponse(BaseModel):
    """Crop recommendation response schema."""
    recommendations: List[Dict[str, Any]]
    features: Dict[str, Any]
    processing_time: float
    uncertainty: Optional[Dict[str, Any]] = None

class ModelManager:
    """Manages ML models and inference."""
    
    def __init__(self, models_dir: str = "/app/models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.metadata = {}
        
    async def load_model(self, model_type: str, model_path: str) -> bool:
        """Load a model for inference."""
        try:
            start_time = time.time()
            
            if model_type == "plant_disease":
                # Load ONNX model
                session = ort.InferenceSession(model_path)
                self.models[model_type] = session
                
                # Load metadata
                metadata_path = self.models_dir / f"{model_type}_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        self.metadata[model_type] = json.load(f)
                
            elif model_type == "cattle_breed":
                # Load ONNX model
                session = ort.InferenceSession(model_path)
                self.models[model_type] = session
                
                # Load metadata
                metadata_path = self.models_dir / f"{model_type}_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        self.metadata[model_type] = json.load(f)
            
            load_time = time.time() - start_time
            MODEL_LOAD_TIME.labels(model_type=model_type).set(load_time)
            
            logger.info(f"Loaded {model_type} model in {load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            return False
    
    async def predict_plant_disease(self, image: np.ndarray, return_uncertainty: bool = False) -> Dict[str, Any]:
        """Predict plant disease from image."""
        if "plant_disease" not in self.models:
            raise HTTPException(status_code=503, detail="Plant disease model not loaded")
        
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image, self.metadata["plant_disease"]["preprocessing"])
            
            # Run inference
            session = self.models["plant_disease"]
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: processed_image})
            
            # Postprocess results
            logits = outputs[0]
            probabilities = F.softmax(torch.tensor(logits), dim=1).numpy()[0]
            
            # Get top predictions
            top_indices = np.argsort(probabilities)[::-1][:5]
            predictions = []
            
            for idx in top_indices:
                class_name = self.metadata["plant_disease"]["id_to_class"][str(idx)]
                predictions.append({
                    "class": class_name,
                    "confidence": float(probabilities[idx]),
                    "class_id": int(idx)
                })
            
            result = {
                "predictions": predictions,
                "model_info": {
                    "model_type": "plant_disease",
                    "version": self.metadata["plant_disease"].get("version", "1.0"),
                    "num_classes": len(self.metadata["plant_disease"]["class_mapping"])
                },
                "processing_time": time.time() - start_time
            }
            
            # Add uncertainty if requested
            if return_uncertainty:
                result["uncertainty"] = self._calculate_uncertainty(probabilities)
            
            MODEL_INFERENCE_TIME.labels(model_type="plant_disease").observe(result["processing_time"])
            return result
            
        except Exception as e:
            logger.error(f"Plant disease prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    async def predict_cattle_breed(self, image: np.ndarray, return_uncertainty: bool = False) -> Dict[str, Any]:
        """Predict cattle breed from image."""
        if "cattle_breed" not in self.models:
            raise HTTPException(status_code=503, detail="Cattle breed model not loaded")
        
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image, self.metadata["cattle_breed"]["preprocessing"])
            
            # Run inference
            session = self.models["cattle_breed"]
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: processed_image})
            
            # Postprocess results
            logits = outputs[0]
            probabilities = F.softmax(torch.tensor(logits), dim=1).numpy()[0]
            
            # Get top predictions
            top_indices = np.argsort(probabilities)[::-1][:5]
            predictions = []
            
            for idx in top_indices:
                class_name = self.metadata["cattle_breed"]["id_to_class"][str(idx)]
                predictions.append({
                    "breed": class_name,
                    "confidence": float(probabilities[idx]),
                    "breed_id": int(idx)
                })
            
            result = {
                "predictions": predictions,
                "model_info": {
                    "model_type": "cattle_breed",
                    "version": self.metadata["cattle_breed"].get("version", "1.0"),
                    "num_classes": len(self.metadata["cattle_breed"]["class_mapping"])
                },
                "processing_time": time.time() - start_time
            }
            
            # Add uncertainty if requested
            if return_uncertainty:
                result["uncertainty"] = self._calculate_uncertainty(probabilities)
            
            MODEL_INFERENCE_TIME.labels(model_type="cattle_breed").observe(result["processing_time"])
            return result
            
        except Exception as e:
            logger.error(f"Cattle breed prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def _preprocess_image(self, image: np.ndarray, preprocessing_config: Dict[str, Any]) -> np.ndarray:
        """Preprocess image for model inference."""
        # Resize image
        target_size = preprocessing_config["input_size"]
        image = cv2.resize(image, (target_size, target_size))
        
        # Normalize
        mean = np.array(preprocessing_config["mean"])
        std = np.array(preprocessing_config["std"])
        image = (image.astype(np.float32) / 255.0 - mean) / std
        
        # Add batch dimension and transpose for ONNX
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        return image.astype(np.float32)
    
    def _calculate_uncertainty(self, probabilities: np.ndarray) -> Dict[str, float]:
        """Calculate prediction uncertainty."""
        # Entropy-based uncertainty
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
        
        # Max probability uncertainty
        max_prob = np.max(probabilities)
        max_uncertainty = 1.0 - max_prob
        
        return {
            "entropy": float(entropy),
            "max_uncertainty": float(max_uncertainty),
            "confidence": float(max_prob)
        }

class ExternalServiceClient:
    """Client for external services (SoilGrids, OpenWeather, GEE)."""
    
    def __init__(self):
        self.soilgrids_base_url = "https://rest.soilgrids.org"
        self.openweather_api_key = os.getenv("OPENWEATHER_API_KEY")
        self.gee_service_url = os.getenv("GEE_SERVICE_URL", "http://localhost:8000")
    
    async def get_soil_features(self, lat: float, lon: float) -> Dict[str, Any]:
        """Get soil features from SoilGrids."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.soilgrids_base_url}/query",
                    params={"lon": lon, "lat": lat}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"SoilGrids API error: {e}")
            return {}
    
    async def get_weather_features(self, lat: float, lon: float) -> Dict[str, Any]:
        """Get weather features from OpenWeather."""
        if not self.openweather_api_key:
            logger.warning("OpenWeather API key not configured")
            return {}
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    "https://api.openweathermap.org/data/2.5/onecall",
                    params={
                        "lat": lat,
                        "lon": lon,
                        "exclude": "minutely,hourly,current,alerts",
                        "units": "metric",
                        "appid": self.openweather_api_key
                    }
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"OpenWeather API error: {e}")
            return {}
    
    async def get_ndvi_features(self, lat: float, lon: float) -> Dict[str, Any]:
        """Get NDVI features from GEE service."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.gee_service_url}/ndvi",
                    json={"lat": lat, "lon": lon, "buffer_m": 500}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"GEE service error: {e}")
            return {}

class CropRecommender:
    """Rule-based crop recommendation system."""
    
    def __init__(self):
        self.crop_rules = {
            "Wheat": {"ph_range": (6.0, 7.5), "rainfall_range": (5, 40), "temp_range": (15, 25)},
            "Rice": {"ph_range": (5.5, 7.0), "rainfall_range": (30, 150), "temp_range": (20, 30)},
            "Maize": {"ph_range": (5.5, 7.5), "rainfall_range": (10, 80), "temp_range": (18, 27)},
            "Sorghum": {"ph_range": (5.0, 7.5), "rainfall_range": (5, 50), "temp_range": (20, 30)},
            "Millet": {"ph_range": (5.0, 7.5), "rainfall_range": (5, 40), "temp_range": (20, 30)},
            "Pulses": {"ph_range": (6.0, 7.5), "rainfall_range": (5, 40), "temp_range": (15, 25)},
            "Cotton": {"ph_range": (5.8, 8.0), "rainfall_range": (10, 70), "temp_range": (20, 30)},
            "Sugarcane": {"ph_range": (6.0, 8.0), "rainfall_range": (40, 200), "temp_range": (20, 30)}
        }
    
    def recommend_crops(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend crops based on environmental features."""
        recommendations = []
        
        for crop, rules in self.crop_rules.items():
            score = 0.0
            reasons = []
            
            # Check pH
            if "ph" in features:
                ph = features["ph"]
                if rules["ph_range"][0] <= ph <= rules["ph_range"][1]:
                    score += 0.25
                    reasons.append(f"pH suitable ({ph:.1f})")
                else:
                    reasons.append(f"pH {'low' if ph < rules['ph_range'][0] else 'high'} ({ph:.1f})")
            
            # Check rainfall
            if "rainfall" in features:
                rainfall = features["rainfall"]
                if rules["rainfall_range"][0] <= rainfall <= rules["rainfall_range"][1]:
                    score += 0.25
                    reasons.append(f"Rainfall suitable ({rainfall:.1f}mm)")
                else:
                    reasons.append(f"Rainfall {'low' if rainfall < rules['rainfall_range'][0] else 'high'} ({rainfall:.1f}mm)")
            
            # Check temperature
            if "temperature" in features:
                temp = features["temperature"]
                if rules["temp_range"][0] <= temp <= rules["temp_range"][1]:
                    score += 0.25
                    reasons.append(f"Temperature suitable ({temp:.1f}°C)")
                else:
                    reasons.append(f"Temperature {'low' if temp < rules['temp_range'][0] else 'high'} ({temp:.1f}°C)")
            
            # Check NDVI trend
            if "ndvi_trend" in features:
                ndvi_trend = features["ndvi_trend"]
                if ndvi_trend > 0:
                    score += 0.25
                    reasons.append("NDVI trend positive")
                else:
                    reasons.append("NDVI trend negative")
            
            recommendations.append({
                "crop": crop,
                "score": min(score, 1.0),
                "reasons": reasons
            })
        
        # Sort by score and return top 3
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:3]

# Initialize services
model_manager = ModelManager()
external_client = ExternalServiceClient()
crop_recommender = CropRecommender()

# Initialize Sentry
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    integrations=[
        FastApiIntegration(auto_enabling_instrumentations=False),
        RedisIntegration(),
    ],
    traces_sample_rate=0.1,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Khet Guard ML service...")
    
    # Load models
    await model_manager.load_model("plant_disease", "/app/models/plant_disease.onnx")
    await model_manager.load_model("cattle_breed", "/app/models/cattle_breed.onnx")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Khet Guard ML service...")

# Create FastAPI app
app = FastAPI(
    title="Khet Guard ML Service",
    description="Production ML service for plant disease detection, cattle breed classification, and crop recommendation",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate JWT token."""
    # Implement JWT validation here
    # For now, just return a mock user
    return {"user_id": "test_user", "role": "farmer"}

# Utility functions
def decode_base64_image(image_base64: str) -> np.ndarray:
    """Decode base64 image to numpy array."""
    import base64
    from io import BytesIO
    
    try:
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

async def download_image_from_url(image_url: str) -> np.ndarray:
    """Download image from URL."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(image_url)
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict/plant_disease", response_model=PredictionResponse)
async def predict_plant_disease(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Predict plant disease from image."""
    start_time = time.time()
    
    try:
        # Get image
        if request.image_base64:
            image = decode_base64_image(request.image_base64)
        elif request.image_url:
            image = await download_image_from_url(request.image_url)
        else:
            raise HTTPException(status_code=400, detail="Either image_base64 or image_url must be provided")
        
        # Run prediction
        result = await model_manager.predict_plant_disease(
            image, 
            return_uncertainty=request.return_uncertainty
        )
        
        # Log request
        REQUEST_COUNT.labels(method="POST", endpoint="/predict/plant_disease", status="200").inc()
        
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plant disease prediction error: {e}")
        REQUEST_COUNT.labels(method="POST", endpoint="/predict/plant_disease", status="500").inc()
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        REQUEST_DURATION.labels(method="POST", endpoint="/predict/plant_disease").observe(time.time() - start_time)

@app.post("/predict/cattle_breed", response_model=PredictionResponse)
async def predict_cattle_breed(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Predict cattle breed from image."""
    start_time = time.time()
    
    try:
        # Get image
        if request.image_base64:
            image = decode_base64_image(request.image_base64)
        elif request.image_url:
            image = await download_image_from_url(request.image_url)
        else:
            raise HTTPException(status_code=400, detail="Either image_base64 or image_url must be provided")
        
        # Run prediction
        result = await model_manager.predict_cattle_breed(
            image, 
            return_uncertainty=request.return_uncertainty
        )
        
        # Log request
        REQUEST_COUNT.labels(method="POST", endpoint="/predict/cattle_breed", status="200").inc()
        
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cattle breed prediction error: {e}")
        REQUEST_COUNT.labels(method="POST", endpoint="/predict/cattle_breed", status="500").inc()
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        REQUEST_DURATION.labels(method="POST", endpoint="/predict/cattle_breed").observe(time.time() - start_time)

@app.post("/recommend/crop", response_model=CropRecommendationResponse)
async def recommend_crop(
    request: CropRecommendationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Recommend crops based on location and environmental data."""
    start_time = time.time()
    
    try:
        # Get environmental features
        soil_task = external_client.get_soil_features(request.latitude, request.longitude)
        weather_task = external_client.get_weather_features(request.latitude, request.longitude)
        ndvi_task = external_client.get_ndvi_features(request.latitude, request.longitude)
        
        # Wait for all external services
        soil_data, weather_data, ndvi_data = await asyncio.gather(
            soil_task, weather_task, ndvi_task, return_exceptions=True
        )
        
        # Process features
        features = {
            "latitude": request.latitude,
            "longitude": request.longitude
        }
        
        # Process soil data
        if isinstance(soil_data, dict) and not isinstance(soil_data, Exception):
            properties = soil_data.get("properties", {})
            features["ph"] = properties.get("phh2o", {}).get("mean", 6.5)
            features["clay"] = properties.get("clay", {}).get("mean", 20)
            features["sand"] = properties.get("sand", {}).get("mean", 40)
        
        # Process weather data
        if isinstance(weather_data, dict) and not isinstance(weather_data, Exception):
            daily = weather_data.get("daily", [])
            if daily:
                # Calculate 7-day averages
                rainfall = sum(day.get("rain", {}).get("1h", 0) for day in daily[:7])
                temperature = sum(day.get("temp", {}).get("day", 20) for day in daily[:7]) / len(daily[:7])
                features["rainfall"] = rainfall
                features["temperature"] = temperature
        
        # Process NDVI data
        if isinstance(ndvi_data, dict) and not isinstance(ndvi_data, Exception):
            ndvi_values = ndvi_data.get("medianMonthlyNdvi", [])
            if len(ndvi_values) >= 2:
                features["ndvi_trend"] = ndvi_values[-1] - ndvi_values[0]
        
        # Get recommendations
        recommendations = crop_recommender.recommend_crops(features)
        
        result = {
            "recommendations": recommendations,
            "features": features,
            "processing_time": time.time() - start_time
        }
        
        # Add uncertainty if requested
        if request.return_uncertainty:
            result["uncertainty"] = {
                "data_quality": "high" if all(k in features for k in ["ph", "rainfall", "temperature"]) else "medium",
                "confidence": 0.8 if len(features) > 3 else 0.6
            }
        
        # Log request
        REQUEST_COUNT.labels(method="POST", endpoint="/recommend/crop", status="200").inc()
        
        return CropRecommendationResponse(**result)
        
    except Exception as e:
        logger.error(f"Crop recommendation error: {e}")
        REQUEST_COUNT.labels(method="POST", endpoint="/recommend/crop", status="500").inc()
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        REQUEST_DURATION.labels(method="POST", endpoint="/recommend/crop").observe(time.time() - start_time)

@app.get("/models/status")
async def get_model_status(current_user: dict = Depends(get_current_user)):
    """Get status of loaded models."""
    status = {}
    for model_type in ["plant_disease", "cattle_breed"]:
        status[model_type] = {
            "loaded": model_type in model_manager.models,
            "metadata": model_manager.metadata.get(model_type, {})
        }
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
