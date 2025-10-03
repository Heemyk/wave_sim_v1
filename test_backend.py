#!/usr/bin/env python3
"""Simple test script to verify the backend is working."""

import requests
import json
import time

def test_backend():
    """Test if the backend is running and responding."""
    base_url = "http://localhost:8000"
    
    print("Testing backend endpoints...")
    
    # Wait a moment for the server to start
    time.sleep(2)
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("[OK] Health check passed")
            health_data = response.json()
            print(f"  Active jobs: {health_data.get('active_jobs', 0)}")
            print(f"  Working directory: {health_data.get('working_directory', 'unknown')}")
        else:
            print(f"[FAILED] Health check failed: {response.status_code}")
            return False
        
        # Test root endpoint
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("[OK] Root endpoint working")
            root_data = response.json()
            print(f"  API version: {root_data.get('version', 'unknown')}")
        else:
            print(f"[FAILED] Root endpoint failed: {response.status_code}")
        
        # Test examples endpoint
        response = requests.get(f"{base_url}/api/examples", timeout=5)
        if response.status_code == 200:
            print("[OK] Examples endpoint working")
            examples_data = response.json()
            print(f"  Available examples: {len(examples_data.get('examples', []))}")
        else:
            print(f"[FAILED] Examples endpoint failed: {response.status_code}")
        
        print("\n[SUCCESS] Backend is running successfully!")
        print(f"Open http://localhost:8000/docs for API documentation")
        return True
        
    except requests.exceptions.ConnectionError:
        print("[FAILED] Backend is not running or not accessible")
        print("Make sure you started the backend with: python run_backend.py")
        return False
    except Exception as e:
        print(f"[FAILED] Error testing backend: {e}")
        return False

if __name__ == "__main__":
    success = test_backend()
    exit(0 if success else 1)
