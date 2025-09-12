"""
Pesticide mapping utilities
---------------------------

Provides:
- load_pesticide_map() function
- Pesticide recommendation logic
- Integration with trained models
"""

import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def load_pesticide_map(map_path: str) -> Dict[str, Any]:
    """
    Load pesticide mapping from JSON file.
    
    Args:
        map_path: Path to pesticide mapping JSON file
    
    Returns:
        Dictionary containing pesticide recommendations
    """
    try:
        if not Path(map_path).exists():
            logger.warning(f"Pesticide map not found at {map_path}, creating default")
            return create_default_pesticide_map(map_path)
        
        with open(map_path, 'r', encoding='utf-8') as f:
            pesticide_map = json.load(f)
        
        logger.info(f"Loaded pesticide map with {len(pesticide_map)} entries")
        return pesticide_map
        
    except Exception as e:
        logger.error(f"Error loading pesticide map: {e}")
        return create_default_pesticide_map(map_path)


def create_default_pesticide_map(map_path: str) -> Dict[str, Any]:
    """
    Create a default pesticide mapping if none exists.
    
    Args:
        map_path: Path where to save the default mapping
    
    Returns:
        Default pesticide mapping dictionary
    """
    default_map = {
        "Tomato___Early_blight": {
            "recommended": ["Mancozeb 75% WP", "Chlorothalonil 75% WP"],
            "dosage": "2g per litre of water, spray weekly",
            "safety": "Wear gloves and mask while spraying. Avoid contact with skin and eyes.",
            "organic_alternatives": ["Neem oil", "Copper fungicide", "Baking soda solution"]
        },
        "Tomato___Late_blight": {
            "recommended": ["Metalaxyl 8% + Mancozeb 64% WP", "Copper oxychloride 50% WP"],
            "dosage": "2.5g per litre, repeat every 10 days",
            "safety": "Avoid spraying close to harvest. Do not exceed 3 applications per season.",
            "organic_alternatives": ["Copper fungicide", "Bordeaux mixture", "Garlic extract"]
        },
        "Tomato___Leaf_Mold": {
            "recommended": ["Azoxystrobin 23% SC", "Chlorothalonil 75% WP"],
            "dosage": "1.5g per litre, apply at first sign of disease",
            "safety": "Do not exceed 3 sprays per season. Maintain proper ventilation.",
            "organic_alternatives": ["Neem oil", "Sulfur dust", "Baking soda spray"]
        },
        "Tomato___Bacterial_spot": {
            "recommended": ["Copper oxychloride 50% WP", "Streptomycin sulfate"],
            "dosage": "2g per litre, spray every 7-10 days",
            "safety": "Use protective equipment. Avoid spraying during flowering.",
            "organic_alternatives": ["Copper fungicide", "Garlic extract", "Horsetail tea"]
        },
        "Tomato___healthy": {
            "recommended": ["Consult local agricultural expert"],
            "dosage": "N/A",
            "safety": "Get proper diagnosis before treatment",
            "organic_alternatives": ["Preventive measures", "Good cultural practices", "Regular monitoring"]
        },
        "Corn_(maize)___Common_rust_": {
            "recommended": ["Propiconazole 25% EC", "Tebuconazole 25% EC"],
            "dosage": "1ml per litre, spray at first sign",
            "safety": "Apply during early morning or evening. Avoid during flowering.",
            "organic_alternatives": ["Neem oil", "Sulfur dust", "Copper fungicide"]
        },
        "Potato___Late_blight": {
            "recommended": ["Metalaxyl 8% + Mancozeb 64% WP"],
            "dosage": "2.5g per litre, repeat every 10 days",
            "safety": "Critical disease - apply preventively. Avoid overhead irrigation.",
            "organic_alternatives": ["Copper fungicide", "Bordeaux mixture", "Garlic extract"]
        },
        "Aphid": {
            "recommended": ["Imidacloprid 17.8% SL", "Thiamethoxam 25% WG"],
            "dosage": "0.3ml per litre, spray in evening",
            "safety": "Wear protective clothing. Avoid contact with beneficial insects.",
            "organic_alternatives": ["Neem oil", "Soap solution", "Ladybird beetles"]
        },
        "Bollworm": {
            "recommended": ["Spinosad 45% SC", "Emamectin benzoate 5% SG"],
            "dosage": "0.5ml per litre, avoid during flowering",
            "safety": "Apply during early morning. Do not exceed 3 applications per season.",
            "organic_alternatives": ["Bacillus thuringiensis", "Neem oil", "Trichogramma wasps"]
        },
        "Whitefly": {
            "recommended": ["Acetamiprid 20% SP", "Pyriproxyfen 10% EC"],
            "dosage": "0.5g per litre, spray under leaves",
            "safety": "Apply in early morning. Monitor for resistance development.",
            "organic_alternatives": ["Yellow sticky traps", "Neem oil", "Encarsia wasps"]
        }
    }
    
    # Save default map
    try:
        Path(map_path).parent.mkdir(parents=True, exist_ok=True)
        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump(default_map, f, indent=2, ensure_ascii=False)
        logger.info(f"Default pesticide map saved to {map_path}")
    except Exception as e:
        logger.error(f"Error saving default pesticide map: {e}")
    
    return default_map


def get_pesticide_recommendation(disease_name: str, pesticide_map: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get pesticide recommendation for a specific disease.
    
    Args:
        disease_name: Name of the disease/pest
        pesticide_map: Pesticide mapping dictionary
    
    Returns:
        Pesticide recommendation dictionary
    """
    # Try exact match first
    if disease_name in pesticide_map:
        return pesticide_map[disease_name]
    
    # Try partial matches for common variations
    disease_lower = disease_name.lower()
    
    # Check for common disease patterns
    if "blight" in disease_lower:
        if "early" in disease_lower:
            return pesticide_map.get("Tomato___Early_blight", get_default_recommendation())
        elif "late" in disease_lower:
            return pesticide_map.get("Tomato___Late_blight", get_default_recommendation())
    
    if "mold" in disease_lower or "mildew" in disease_lower:
        return pesticide_map.get("Tomato___Leaf_Mold", get_default_recommendation())
    
    if "bacterial" in disease_lower:
        return pesticide_map.get("Tomato___Bacterial_spot", get_default_recommendation())
    
    if "aphid" in disease_lower:
        return pesticide_map.get("Aphid", get_default_recommendation())
    
    if "bollworm" in disease_lower or "worm" in disease_lower:
        return pesticide_map.get("Bollworm", get_default_recommendation())
    
    if "whitefly" in disease_lower:
        return pesticide_map.get("Whitefly", get_default_recommendation())
    
    # Default recommendation
    return get_default_recommendation()


def get_default_recommendation() -> Dict[str, Any]:
    """
    Get default pesticide recommendation for unknown diseases.
    
    Returns:
        Default recommendation dictionary
    """
    return {
        "recommended": ["Consult local agricultural expert"],
        "dosage": "N/A",
        "safety": "Get proper diagnosis before treatment",
        "organic_alternatives": ["Preventive measures", "Good cultural practices", "Regular monitoring"]
    }


def validate_pesticide_map(pesticide_map: Dict[str, Any]) -> bool:
    """
    Validate pesticide mapping structure.
    
    Args:
        pesticide_map: Pesticide mapping dictionary
    
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["recommended", "dosage", "safety"]
    
    for disease, recommendation in pesticide_map.items():
        if not isinstance(recommendation, dict):
            logger.error(f"Invalid recommendation format for {disease}")
            return False
        
        for field in required_fields:
            if field not in recommendation:
                logger.error(f"Missing field '{field}' for {disease}")
                return False
    
    return True


if __name__ == "__main__":
    # Test pesticide mapping
    map_path = "ml/recommender/pesticide_map.json"
    pesticide_map = load_pesticide_map(map_path)
    
    # Test recommendations
    test_diseases = [
        "Tomato___Early_blight",
        "Aphid",
        "Unknown_disease"
    ]
    
    for disease in test_diseases:
        recommendation = get_pesticide_recommendation(disease, pesticide_map)
        print(f"\n{disease}:")
        print(f"  Recommended: {recommendation['recommended']}")
        print(f"  Dosage: {recommendation['dosage']}")
        print(f"  Safety: {recommendation['safety']}")
    
    # Validate map
    is_valid = validate_pesticide_map(pesticide_map)
    print(f"\nPesticide map validation: {'PASS' if is_valid else 'FAIL'}")
