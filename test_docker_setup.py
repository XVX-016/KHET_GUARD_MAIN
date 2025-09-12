#!/usr/bin/env python3
"""
Test script to verify Docker Compose setup for Khet Guard
"""

import requests
import time
import psycopg2
import redis
import json
from typing import Dict, Any

def test_ml_api() -> Dict[str, Any]:
    """Test ML API health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            return {"status": "✅ PASS", "data": response.json()}
        else:
            return {"status": "❌ FAIL", "error": f"Status {response.status_code}"}
    except Exception as e:
        return {"status": "❌ FAIL", "error": str(e)}

def test_database() -> Dict[str, Any]:
    """Test PostgreSQL connection"""
    try:
        conn = psycopg2.connect(
            host="localhost",
            port="5432",
            database="khet_guard",
            user="postgres",
            password="khet_guard_password"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users;")
        user_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM scans;")
        scan_count = cursor.fetchone()[0]
        conn.close()
        return {
            "status": "✅ PASS", 
            "data": {"users": user_count, "scans": scan_count}
        }
    except Exception as e:
        return {"status": "❌ FAIL", "error": str(e)}

def test_redis() -> Dict[str, Any]:
    """Test Redis connection"""
    try:
        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()
        r.set("test_key", "test_value", ex=10)
        value = r.get("test_key")
        r.delete("test_key")
        return {"status": "✅ PASS", "data": {"test_value": value}}
    except Exception as e:
        return {"status": "❌ FAIL", "error": str(e)}

def test_prometheus() -> Dict[str, Any]:
    """Test Prometheus metrics endpoint"""
    try:
        response = requests.get("http://localhost:9090/api/v1/query?query=up", timeout=10)
        if response.status_code == 200:
            return {"status": "✅ PASS", "data": "Prometheus is running"}
        else:
            return {"status": "❌ FAIL", "error": f"Status {response.status_code}"}
    except Exception as e:
        return {"status": "❌ FAIL", "error": str(e)}

def test_grafana() -> Dict[str, Any]:
    """Test Grafana health endpoint"""
    try:
        response = requests.get("http://localhost:3000/api/health", timeout=10)
        if response.status_code == 200:
            return {"status": "✅ PASS", "data": "Grafana is running"}
        else:
            return {"status": "❌ FAIL", "error": f"Status {response.status_code}"}
    except Exception as e:
        return {"status": "❌ FAIL", "error": str(e)}

def main():
    """Run all tests"""
    print("🧪 Testing Khet Guard Docker Compose Setup")
    print("=" * 50)
    
    tests = [
        ("ML API", test_ml_api),
        ("PostgreSQL", test_database),
        ("Redis", test_redis),
        ("Prometheus", test_prometheus),
        ("Grafana", test_grafana),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        result = test_func()
        results[test_name] = result
        print(f"   {result['status']}")
        if "data" in result:
            print(f"   Data: {result['data']}")
        if "error" in result:
            print(f"   Error: {result['error']}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    passed = sum(1 for r in results.values() if r["status"] == "✅ PASS")
    total = len(results)
    
    for test_name, result in results.items():
        print(f"   {test_name}: {result['status']}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All services are running correctly!")
        print("\n🌐 Access URLs:")
        print("   ML API: http://localhost:8000")
        print("   Grafana: http://localhost:3000 (admin/admin)")
        print("   Prometheus: http://localhost:9090")
    else:
        print("⚠️  Some services are not running. Check Docker Compose logs:")
        print("   docker-compose logs")

if __name__ == "__main__":
    main()
