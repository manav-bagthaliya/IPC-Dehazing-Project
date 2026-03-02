#!/usr/bin/env python3
"""
Error handling wrapper for IPC-Dehaze
"""

import sys
import traceback
import os

def safe_run(func, *args, **kwargs):
    """Safely run a function with comprehensive error handling"""
    try:
        return func(*args, **kwargs)
    except IndexError as e:
        print(f"IndexError: {e}")
        print("This usually means:")
        print("1. Empty list accessed with index [0]")
        print("2. Encoder features list is empty")
        print("3. Missing model components")
        print("\nStack trace:")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("\nStack trace:")
        traceback.print_exc()
        return None

def test_web_app():
    """Test the web application"""
    print("Testing web application...")
    
    try:
        from app import app
        print("✓ Flask app imported successfully")
        
        # Test basic Flask functionality
        with app.test_client() as client:
            response = client.get('/')
            if response.status_code == 200:
                print("✓ Web interface accessible")
                return True
            else:
                print(f"✗ Web interface error: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"✗ Web app error: {e}")
        traceback.print_exc()
        return False

def test_inference():
    """Test inference functionality"""
    print("Testing inference...")
    
    try:
        import cv2
        import numpy as np
        from inference import detect_license_plates_advanced
        
        # Create test image
        test_img = np.ones((300, 400, 3), dtype=np.uint8) * 128
        
        # Test detection
        plates = safe_run(detect_license_plates_advanced, test_img)
        if plates is not None:
            print(f"✓ Detection test passed: {len(plates)} plates found")
            return True
        else:
            print("✗ Detection test failed")
            return False
            
    except Exception as e:
        print(f"✗ Inference error: {e}")
        traceback.print_exc()
        return False

def test_file_upload():
    """Test file upload functionality"""
    print("Testing file upload...")
    
    try:
        from app import app
        
        with app.test_client() as client:
            # Test upload endpoint
            response = client.post('/upload', data={})
            if response.status_code == 400:  # Expected for empty upload
                print("✓ Upload endpoint working")
                return True
            else:
                print(f"✗ Upload endpoint error: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"✗ File upload error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("IPC-Dehaze Error Diagnosis")
    print("=" * 30)
    
    tests = [
        ("Web Application", test_web_app),
        ("Inference", test_inference),
        ("File Upload", test_file_upload),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = safe_run(test_func)
        results.append((test_name, result))
    
    print("\n" + "=" * 30)
    print("Test Results:")
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    if all(result for _, result in results):
        print("\n✓ All tests passed! No errors found.")
    else:
        print("\n✗ Some tests failed. Check the errors above.")
    
    print("\nIf you're still getting 'list index out of range' errors:")
    print("1. Check if you have the required model files")
    print("2. Ensure all dependencies are installed")
    print("3. Try running with a simple test image first")
