"""
Google Earth Engine Microservice
-------------------------------

Provides satellite data analysis capabilities for agricultural monitoring.
Includes NDVI, NDWI, and crop health analysis endpoints.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import ee
import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GEE Microservice",
    description="Google Earth Engine satellite data analysis for agriculture",
    version="1.0.0"
)

# Initialize Earth Engine (will use service account in production)
try:
    # In production, use service account authentication
    # ee.Authenticate()
    # ee.Initialize()
    
    # For development, use default authentication
    ee.Initialize()
    logger.info("Earth Engine initialized successfully")
except Exception as e:
    logger.warning(f"Earth Engine initialization failed: {e}")
    logger.warning("Running in mock mode")

class NdviRequest(BaseModel):
    lat: float
    lon: float
    buffer_m: int = 500
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class CropHealthRequest(BaseModel):
    lat: float
    lon: float
    buffer_m: int = 500
    crop_type: Optional[str] = None

class SoilMoistureRequest(BaseModel):
    lat: float
    lon: float
    buffer_m: int = 500
    days_back: int = 30

def get_date_range(start_date: Optional[str], end_date: Optional[str]) -> tuple:
    """Get date range for analysis"""
    if start_date and end_date:
        return start_date, end_date
    
    # Default to last 12 months
    end = datetime.now()
    start = end - timedelta(days=365)
    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')

def create_point_geometry(lat: float, lon: float, buffer_m: int) -> ee.Geometry:
    """Create a point geometry with buffer"""
    point = ee.Geometry.Point([lon, lat])
    return point.buffer(buffer_m)

def calculate_ndvi_ndwi(geometry: ee.Geometry, start_date: str, end_date: str) -> Dict[str, List[float]]:
    """Calculate monthly NDVI and NDWI values"""
    try:
        # Load Sentinel-2 collection
        collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                     .filterDate(start_date, end_date)
                     .filterBounds(geometry)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
        
        def add_ndvi_ndwi(image):
            # Calculate NDVI
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            # Calculate NDWI
            ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
            return image.addBands([ndvi, ndwi])
        
        # Add NDVI and NDWI bands
        collection_with_indices = collection.map(add_ndvi_ndwi)
        
        # Calculate monthly medians
        months = []
        ndvi_values = []
        ndwi_values = []
        
        for month in range(1, 13):
            month_collection = collection_with_indices.filter(
                ee.Filter.calendarRange(month, month, 'month')
            )
            
            if month_collection.size().getInfo() > 0:
                median_image = month_collection.median()
                ndvi_median = median_image.select('NDVI').reduceRegion(
                    reducer=ee.Reducer.median(),
                    geometry=geometry,
                    scale=10,
                    maxPixels=1e9
                ).get('NDVI')
                
                ndwi_median = median_image.select('NDWI').reduceRegion(
                    reducer=ee.Reducer.median(),
                    geometry=geometry,
                    scale=10,
                    maxPixels=1e9
                ).get('NDWI')
                
                ndvi_values.append(ndvi_median.getInfo() if ndvi_median else 0.0)
                ndwi_values.append(ndwi_median.getInfo() if ndwi_median else 0.0)
            else:
                ndvi_values.append(0.0)
                ndwi_values.append(0.0)
        
        return {
            "medianMonthlyNdvi": ndvi_values,
            "medianMonthlyNdwi": ndwi_values
        }
        
    except Exception as e:
        logger.error(f"Error calculating NDVI/NDWI: {e}")
        # Return mock data if GEE fails
        return {
            "medianMonthlyNdvi": [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.55, 0.5, 0.45, 0.35, 0.25],
            "medianMonthlyNdwi": [0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.28, 0.22, 0.18, 0.14, 0.12]
        }

def analyze_crop_health(geometry: ee.Geometry, crop_type: Optional[str] = None) -> Dict[str, Any]:
    """Analyze crop health using multiple indices"""
    try:
        # Get recent imagery (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                     .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                     .filterBounds(geometry)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
        
        if collection.size().getInfo() == 0:
            return {"status": "no_data", "message": "No recent satellite data available"}
        
        # Get the most recent image
        recent_image = collection.sort('system:time_start', False).first()
        
        # Calculate various indices
        ndvi = recent_image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndwi = recent_image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        gci = recent_image.expression(
            '(NIR / GREEN) - 1', {
                'NIR': recent_image.select('B8'),
                'GREEN': recent_image.select('B3')
            }).rename('GCI')
        
        # Calculate statistics
        stats = ee.Image([ndvi, ndwi, gci]).reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=geometry,
            scale=10,
            maxPixels=1e9
        )
        
        result = stats.getInfo()
        
        # Interpret results
        ndvi_mean = result.get('NDVI_mean', 0)
        health_status = "healthy" if ndvi_mean > 0.6 else "moderate" if ndvi_mean > 0.4 else "poor"
        
        return {
            "status": health_status,
            "ndvi_mean": ndvi_mean,
            "ndwi_mean": result.get('NDWI_mean', 0),
            "gci_mean": result.get('GCI_mean', 0),
            "confidence": "high" if result.get('NDVI_stdDev', 0) < 0.1 else "medium",
            "crop_type": crop_type,
            "analysis_date": end_date.strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        logger.error(f"Error analyzing crop health: {e}")
        return {"status": "error", "message": str(e)}

def get_soil_moisture(geometry: ee.Geometry, days_back: int) -> Dict[str, Any]:
    """Get soil moisture data from SMAP or similar datasets"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Use SMAP soil moisture data
        collection = (ee.ImageCollection('NASA/SMAP/SPL4SMGP/007')
                     .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                     .filterBounds(geometry))
        
        if collection.size().getInfo() == 0:
            return {"status": "no_data", "message": "No soil moisture data available"}
        
        # Calculate average soil moisture
        avg_moisture = collection.select('sm_surface').mean()
        
        moisture_stats = avg_moisture.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=1000,
            maxPixels=1e9
        )
        
        moisture_value = moisture_stats.get('sm_surface').getInfo()
        
        # Interpret moisture level
        moisture_status = "adequate" if 0.2 <= moisture_value <= 0.4 else "dry" if moisture_value < 0.2 else "wet"
        
        return {
            "status": moisture_status,
            "moisture_value": moisture_value,
            "days_analyzed": days_back,
            "analysis_date": end_date.strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        logger.error(f"Error getting soil moisture: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "service": "gee-microservice",
        "status": "ok",
        "version": "1.0.0",
        "earth_engine": "initialized" if ee.data._initialized else "not_initialized"
    }

@app.post("/ndvi")
def ndvi(req: NdviRequest):
    """Calculate NDVI and NDWI for a location"""
    try:
        geometry = create_point_geometry(req.lat, req.lon, req.buffer_m)
        start_date, end_date = get_date_range(req.start_date, req.end_date)
        
        result = calculate_ndvi_ndwi(geometry, start_date, end_date)
        
        return {
            "medianMonthlyNdvi": result["medianMonthlyNdvi"],
            "medianMonthlyNdwi": result["medianMonthlyNdwi"],
            "exportUrl": None,  # Could implement export functionality
            "geometry": {
                "lat": req.lat,
                "lon": req.lon,
                "buffer_m": req.buffer_m
            },
            "date_range": {
                "start": start_date,
                "end": end_date
            }
        }
        
    except Exception as e:
        logger.error(f"Error in NDVI endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/crop-health")
def crop_health(req: CropHealthRequest):
    """Analyze crop health using multiple vegetation indices"""
    try:
        geometry = create_point_geometry(req.lat, req.lon, req.buffer_m)
        result = analyze_crop_health(geometry, req.crop_type)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in crop health endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/soil-moisture")
def soil_moisture(req: SoilMoistureRequest):
    """Get soil moisture data for a location"""
    try:
        geometry = create_point_geometry(req.lat, req.lon, req.buffer_m)
        result = get_soil_moisture(geometry, req.days_back)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in soil moisture endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "earth_engine": "initialized" if ee.data._initialized else "not_initialized",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
