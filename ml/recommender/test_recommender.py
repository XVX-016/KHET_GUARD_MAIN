#!/usr/bin/env python3
"""Test script for pesticide recommender"""

from engine import recommend

def test_recommender():
    print("Testing pesticide recommender...")
    
    # Test known pests
    test_cases = ["aphid", "bollworm", "unknown_pest"]
    
    for pest in test_cases:
        result = recommend(pest)
        print(f"\nPest: {pest}")
        print(f"Recommended: {result['recommended']}")
        print(f"Dosage: {result['dosage']}")
        print(f"Safety: {result['safety']}")

if __name__ == "__main__":
    test_recommender()
