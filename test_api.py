#!/usr/bin/env python3
"""
Test script for Khet Guard ML Inference API
"""

import requests
import json
from pathlib import Path

API_BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_disease_prediction():
    """Test disease/pest prediction endpoint"""
    print("Testing disease/pest prediction...")
    
    # Create a dummy image file for testing
    from PIL import Image
    import io
    
    # Create a simple test image
    test_image = Image.new('RGB', (224, 224), color='green')
    img_buffer = io.BytesIO()
    test_image.save(img_buffer, format='JPEG')
    img_buffer.seek(0)
    
    files = {'file': ('test_leaf.jpg', img_buffer, 'image/jpeg')}
    response = requests.post(f"{API_BASE_URL}/predict/disease_pest", files=files)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Predicted class: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.3f}")
        if result.get('pesticide_recommendation'):
            print(f"Pesticide: {result['pesticide_recommendation']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_cattle_prediction():
    """Test cattle prediction endpoint"""
    print("Testing cattle prediction...")
    
    # Create a dummy image file for testing
    from PIL import Image
    import io
    
    # Create a simple test image
    test_image = Image.new('RGB', (224, 224), color='brown')
    img_buffer = io.BytesIO()
    test_image.save(img_buffer, format='JPEG')
    img_buffer.seek(0)
    
    files = {'file': ('test_cow.jpg', img_buffer, 'image/jpeg')}
    response = requests.post(f"{API_BASE_URL}/predict/cattle", files=files)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Predicted breed: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.3f}")
        if result.get('breed_info'):
            print(f"Breed info: {result['breed_info']}")
    else:
        print(f"Error: {response.text}")
    print()

def main():
    """Run all tests"""
    print("🧪 Testing Khet Guard ML Inference API")
    print("=" * 50)
    
    try:
        test_health()
        test_disease_prediction()
        test_cattle_prediction()
        print("✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running:")
        print("   uvicorn inference_api:app --reload --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()
