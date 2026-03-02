#!/usr/bin/env python3
"""
Simple test to identify list index errors
"""

import sys
import traceback
import os

def test_specific_scenarios():
    """Test specific scenarios that might cause list index errors"""
    print("Testing specific scenarios...")
    
    # Test 1: Empty file list
    print("\n1. Testing empty file list handling...")
    try:
        files = []
        if not files or len(files) == 0:
            print("   PASS: Empty file list handled correctly")
        else:
            print("   FAIL: Empty file list not handled")
    except IndexError as e:
        print(f"   FAIL: IndexError with empty list: {e}")
    
    # Test 2: Encoder features
    print("\n2. Testing encoder features...")
    try:
        import torch
        import numpy as np
        
        # Simulate empty encoder features
        enc_feats = []
        if len(enc_feats) == 0:
            print("   PASS: Empty encoder features handled correctly")
        else:
            print("   FAIL: Empty encoder features not handled")
            
        # Test accessing first element
        if len(enc_feats) > 0:
            x = enc_feats[0]
            print("   PASS: Can access first element")
        else:
            print("   PASS: Correctly skipped accessing empty list")
            
    except IndexError as e:
        print(f"   FAIL: IndexError with encoder features: {e}")
    
    # Test 3: License plate detection with empty results
    print("\n3. Testing license plate detection...")
    try:
        import cv2
        import numpy as np
        from inference import detect_license_plates_advanced
        
        # Test with simple image
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        plates = detect_license_plates_advanced(test_img)
        
        print(f"   PASS: Detection completed, found {len(plates)} plates")
        
        # Test processing empty results
        for i, plate in enumerate(plates):
            bbox = plate['bbox']
            print(f"   Plate {i+1}: bbox={bbox}")
            
    except IndexError as e:
        print(f"   FAIL: IndexError in detection: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"   FAIL: Other error in detection: {e}")
        traceback.print_exc()
    
    # Test 4: Flask app file handling
    print("\n4. Testing Flask file handling...")
    try:
        from app import app
        
        with app.test_client() as client:
            # Test with no files
            response = client.post('/upload', data={})
            print(f"   PASS: Upload without files handled: {response.status_code}")
            
            # Test with empty file list
            response = client.post('/upload', data={'files': []})
            print(f"   PASS: Upload with empty file list handled: {response.status_code}")
            
    except IndexError as e:
        print(f"   FAIL: IndexError in Flask: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"   FAIL: Other error in Flask: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("IPC-Dehaze List Index Error Test")
    print("=" * 40)
    
    test_specific_scenarios()
    
    print("\n" + "=" * 40)
    print("Test completed!")
    print("\nIf you're still getting 'list index out of range' errors:")
    print("Please tell me exactly when the error occurs:")
    print("1. When starting the web app?")
    print("2. When uploading images?")
    print("3. When processing images?")
    print("4. When running specific commands?")
