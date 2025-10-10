#!/usr/bin/env python3
"""
Test script for license plate detection functionality
"""

import cv2
import numpy as np
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the license plate detection functions
from inference import (
    detect_license_plates_opencv,
    detect_license_plates_yolo,
    enhance_plate_region,
    bbox_overlap
)

def test_license_plate_detection():
    """Test the license plate detection functions"""
    print("Testing License Plate Detection...")
    
    # Create a test image (you can replace this with a real image path)
    test_image_path = "examples/OIP.jpeg"  # Use one of your example images
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        print("Please ensure you have example images in the examples/ folder")
        return
    
    # Load test image
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"Failed to load image: {test_image_path}")
        return
    
    print(f"Loaded test image: {image.shape}")
    
    # Test OpenCV detection
    print("\n1. Testing OpenCV-based license plate detection...")
    plates_opencv = detect_license_plates_opencv(image)
    print(f"   Found {len(plates_opencv)} plates using OpenCV")
    
    for i, plate in enumerate(plates_opencv):
        print(f"   Plate {i+1}: bbox={plate['bbox']}, confidence={plate['confidence']}")
    
    # Test YOLO detection (if available)
    print("\n2. Testing YOLO-based license plate detection...")
    plates_yolo = detect_license_plates_yolo(image)
    print(f"   Found {len(plates_yolo)} plates using YOLO")
    
    for i, plate in enumerate(plates_yolo):
        print(f"   Plate {i+1}: bbox={plate['bbox']}, confidence={plate['confidence']}")
    
    # Test plate enhancement
    print("\n3. Testing license plate enhancement...")
    if plates_opencv or plates_yolo:
        # Use the first detected plate
        test_plate = (plates_opencv + plates_yolo)[0]
        bbox = test_plate['bbox']
        
        print(f"   Enhancing plate at bbox: {bbox}")
        enhanced_plate = enhance_plate_region(image, bbox)
        
        # Save enhanced plate for inspection
        output_path = "test_enhanced_plate.jpg"
        cv2.imwrite(output_path, enhanced_plate)
        print(f"   Enhanced plate saved to: {output_path}")
    else:
        print("   No plates detected to test enhancement")
    
    # Test bbox overlap function
    print("\n4. Testing bbox overlap calculation...")
    bbox1 = [100, 100, 200, 150]
    bbox2 = [150, 120, 250, 170]
    overlap = bbox_overlap(bbox1, bbox2)
    print(f"   Overlap between {bbox1} and {bbox2}: {overlap:.3f}")
    
    print("\nLicense plate detection test completed!")

if __name__ == "__main__":
    test_license_plate_detection()
