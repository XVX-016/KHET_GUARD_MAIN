"""
Model drift detection service for Khet Guard ML models.
Monitors input data distribution changes and triggers alerts.
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import kl_divergence
import redis
import httpx
from prometheus_client import Gauge, Counter, start_http_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
DRIFT_SCORE = Gauge('model_drift_score', 'Model drift score', ['model_type', 'feature'])
DRIFT_ALERTS = Counter('model_drift_alerts_total', 'Total drift alerts', ['model_type', 'severity'])

class DriftDetector:
    """Detects model drift by monitoring input data distribution changes."""
    
    def __init__(self, 
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 ml_service_url: str = 'http://localhost:8000'):
        
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        self.ml_service_url = ml_service_url
        self.baseline_distributions = {}
        self.drift_thresholds = {
            'kl_divergence': 0.5,
            'wasserstein_distance': 0.3,
            'ks_test_pvalue': 0.05
        }
    
    async def load_baseline_distributions(self, model_type: str) -> Dict[str, Any]:
        """Load baseline distributions for drift detection."""
        try:
            baseline_key = f"baseline_distributions:{model_type}"
            baseline_data = self.redis_client.get(baseline_key)
            
            if baseline_data:
                return json.loads(baseline_data)
            else:
                logger.warning(f"No baseline distributions found for {model_type}")
                return {}
        except Exception as e:
            logger.error(f"Error loading baseline distributions: {e}")
            return {}
    
    async def save_baseline_distributions(self, model_type: str, distributions: Dict[str, Any]):
        """Save baseline distributions for drift detection."""
        try:
            baseline_key = f"baseline_distributions:{model_type}"
            self.redis_client.setex(
                baseline_key, 
                86400 * 30,  # 30 days
                json.dumps(distributions)
            )
            logger.info(f"Saved baseline distributions for {model_type}")
        except Exception as e:
            logger.error(f"Error saving baseline distributions: {e}")
    
    async def collect_recent_data(self, model_type: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Collect recent data for drift detection."""
        try:
            # Get recent predictions from Redis
            recent_key = f"recent_predictions:{model_type}"
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Get all recent predictions
            recent_data = []
            for key in self.redis_client.scan_iter(match=f"{recent_key}:*"):
                data = self.redis_client.get(key)
                if data:
                    prediction = json.loads(data)
                    prediction_time = datetime.fromisoformat(prediction.get('timestamp', ''))
                    if prediction_time >= cutoff_time:
                        recent_data.append(prediction)
            
            logger.info(f"Collected {len(recent_data)} recent predictions for {model_type}")
            return recent_data
        except Exception as e:
            logger.error(f"Error collecting recent data: {e}")
            return []
    
    def calculate_feature_distributions(self, data: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Calculate feature distributions from data."""
        distributions = {}
        
        if not data:
            return distributions
        
        # Extract features
        features = {
            'confidence': [pred.get('confidence', 0) for pred in data],
            'uncertainty': [pred.get('uncertainty', {}).get('entropy', 0) for pred in data],
            'processing_time': [pred.get('processing_time', 0) for pred in data],
        }
        
        # Calculate distributions
        for feature_name, values in features.items():
            if values:
                distributions[feature_name] = np.array(values)
        
        return distributions
    
    def calculate_drift_metrics(self, 
                               baseline_dist: np.ndarray, 
                               current_dist: np.ndarray) -> Dict[str, float]:
        """Calculate drift metrics between baseline and current distributions."""
        metrics = {}
        
        try:
            # KL Divergence
            if len(baseline_dist) > 0 and len(current_dist) > 0:
                # Create histograms for KL divergence
                bins = np.linspace(
                    min(np.min(baseline_dist), np.min(current_dist)),
                    max(np.max(baseline_dist), np.max(current_dist)),
                    50
                )
                
                baseline_hist, _ = np.histogram(baseline_dist, bins=bins, density=True)
                current_hist, _ = np.histogram(current_dist, bins=bins, density=True)
                
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                baseline_hist = baseline_hist + epsilon
                current_hist = current_hist + epsilon
                
                # Normalize
                baseline_hist = baseline_hist / np.sum(baseline_hist)
                current_hist = current_hist / np.sum(current_hist)
                
                kl_div = kl_divergence(baseline_hist, current_hist)
                metrics['kl_divergence'] = float(kl_div)
            
            # Wasserstein Distance
            if len(baseline_dist) > 0 and len(current_dist) > 0:
                from scipy.stats import wasserstein_distance
                wasserstein_dist = wasserstein_distance(baseline_dist, current_dist)
                metrics['wasserstein_distance'] = float(wasserstein_dist)
            
            # Kolmogorov-Smirnov Test
            if len(baseline_dist) > 0 and len(current_dist) > 0:
                ks_statistic, ks_pvalue = stats.ks_2samp(baseline_dist, current_dist)
                metrics['ks_test_statistic'] = float(ks_statistic)
                metrics['ks_test_pvalue'] = float(ks_pvalue)
            
            # Mean shift
            if len(baseline_dist) > 0 and len(current_dist) > 0:
                mean_shift = abs(np.mean(current_dist) - np.mean(baseline_dist))
                metrics['mean_shift'] = float(mean_shift)
            
            # Variance change
            if len(baseline_dist) > 0 and len(current_dist) > 0:
                var_ratio = np.var(current_dist) / (np.var(baseline_dist) + 1e-10)
                metrics['variance_ratio'] = float(var_ratio)
            
        except Exception as e:
            logger.error(f"Error calculating drift metrics: {e}")
        
        return metrics
    
    async def detect_drift(self, model_type: str) -> Dict[str, Any]:
        """Detect drift for a specific model."""
        try:
            # Load baseline distributions
            baseline_distributions = await self.load_baseline_distributions(model_type)
            if not baseline_distributions:
                logger.warning(f"No baseline distributions for {model_type}")
                return {'drift_detected': False, 'reason': 'no_baseline'}
            
            # Collect recent data
            recent_data = await self.collect_recent_data(model_type)
            if len(recent_data) < 100:  # Need minimum data points
                logger.warning(f"Insufficient recent data for {model_type}: {len(recent_data)} samples")
                return {'drift_detected': False, 'reason': 'insufficient_data'}
            
            # Calculate current distributions
            current_distributions = self.calculate_feature_distributions(recent_data)
            
            # Compare with baseline
            drift_results = {}
            max_drift_score = 0
            
            for feature_name, current_dist in current_distributions.items():
                if feature_name in baseline_distributions:
                    baseline_dist = np.array(baseline_distributions[feature_name])
                    
                    # Calculate drift metrics
                    metrics = self.calculate_drift_metrics(baseline_dist, current_dist)
                    
                    # Calculate overall drift score
                    drift_score = self.calculate_drift_score(metrics)
                    drift_results[feature_name] = {
                        'metrics': metrics,
                        'drift_score': drift_score
                    }
                    
                    # Update Prometheus metrics
                    DRIFT_SCORE.labels(model_type=model_type, feature=feature_name).set(drift_score)
                    
                    max_drift_score = max(max_drift_score, drift_score)
            
            # Determine if drift is detected
            drift_detected = max_drift_score > 0.5  # Threshold
            
            if drift_detected:
                DRIFT_ALERTS.labels(model_type=model_type, severity='high').inc()
                logger.warning(f"Drift detected for {model_type}: max score = {max_drift_score:.3f}")
            
            return {
                'drift_detected': drift_detected,
                'max_drift_score': max_drift_score,
                'feature_results': drift_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error detecting drift for {model_type}: {e}")
            return {'drift_detected': False, 'error': str(e)}
    
    def calculate_drift_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall drift score from individual metrics."""
        score = 0.0
        
        # KL Divergence weight
        if 'kl_divergence' in metrics:
            kl_score = min(metrics['kl_divergence'] / self.drift_thresholds['kl_divergence'], 1.0)
            score += kl_score * 0.4
        
        # Wasserstein Distance weight
        if 'wasserstein_distance' in metrics:
            wasserstein_score = min(metrics['wasserstein_distance'] / self.drift_thresholds['wasserstein_distance'], 1.0)
            score += wasserstein_score * 0.3
        
        # KS Test weight
        if 'ks_test_pvalue' in metrics:
            ks_score = 1.0 - min(metrics['ks_test_pvalue'] / self.drift_thresholds['ks_test_pvalue'], 1.0)
            score += ks_score * 0.2
        
        # Mean shift weight
        if 'mean_shift' in metrics:
            mean_shift_score = min(metrics['mean_shift'] / 0.1, 1.0)  # Normalize by 0.1
            score += mean_shift_score * 0.1
        
        return min(score, 1.0)
    
    async def store_prediction(self, model_type: str, prediction: Dict[str, Any]):
        """Store prediction data for drift detection."""
        try:
            prediction_key = f"recent_predictions:{model_type}:{datetime.now().isoformat()}"
            self.redis_client.setex(prediction_key, 86400 * 7, json.dumps(prediction))  # Keep for 7 days
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
    
    async def create_baseline(self, model_type: str, days: int = 30):
        """Create baseline distributions from historical data."""
        try:
            logger.info(f"Creating baseline for {model_type} from last {days} days")
            
            # Collect historical data
            historical_data = await self.collect_recent_data(model_type, hours=days * 24)
            
            if len(historical_data) < 1000:
                logger.warning(f"Insufficient historical data for baseline: {len(historical_data)} samples")
                return False
            
            # Calculate distributions
            distributions = self.calculate_feature_distributions(historical_data)
            
            # Save baseline
            await self.save_baseline_distributions(model_type, {
                feature: dist.tolist() for feature, dist in distributions.items()
            })
            
            logger.info(f"Created baseline for {model_type} with {len(historical_data)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error creating baseline for {model_type}: {e}")
            return False

async def main():
    """Main drift detection loop."""
    detector = DriftDetector()
    
    # Start Prometheus metrics server
    start_http_server(8001)
    logger.info("Started drift detection service on port 8001")
    
    # Create baselines for all models
    models = ['plant_disease', 'cattle_breed']
    for model_type in models:
        await detector.create_baseline(model_type)
    
    # Run drift detection loop
    while True:
        try:
            for model_type in models:
                result = await detector.detect_drift(model_type)
                logger.info(f"Drift detection result for {model_type}: {result}")
            
            # Wait before next check
            await asyncio.sleep(3600)  # Check every hour
            
        except Exception as e:
            logger.error(f"Error in drift detection loop: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying

if __name__ == "__main__":
    asyncio.run(main())
