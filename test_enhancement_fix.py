#!/usr/bin/env python3
"""
Test script to verify the balanced enhancement parameters
"""

import cv2
import numpy as np
import os

def test_balanced_enhancement():
    """Test the balanced enhancement parameters"""
    print("Testing balanced enhancement parameters...")
    
    # Create a realistic test image with license plate
    test_image = np.zeros((300, 500, 3), dtype=np.uint8)
    
    # Add realistic background (gray road)
    test_image[:, :] = [80, 80, 80]
    
    # Add a license plate region (darker rectangle)
    cv2.rectangle(test_image, (150, 100), (350, 180), (40, 40, 40), -1)
    
    # Add license plate text (white text on dark background)
    cv2.putText(test_image, "ABC123", (170, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    # Add some haze effect (realistic)
    haze = np.ones_like(test_image) * 50
    test_image = cv2.addWeighted(test_image, 0.8, haze, 0.2, 0)
    
    # Test the balanced enhancement
    from inference import enhance_plate_region
    
    # Define license plate region
    plate_bbox = [150, 100, 350, 180]
    
    # Apply balanced enhancement
    enhanced = enhance_plate_region(test_image, plate_bbox)
    
    # Save results
    os.makedirs('test_results', exist_ok=True)
    cv2.imwrite('test_results/original_hazy.png', test_image)
    cv2.imwrite('test_results/balanced_enhanced.png', enhanced)
    
    # Check if enhancement is reasonable (not over-processed)
    enhanced_mean = np.mean(enhanced)
    original_mean = np.mean(test_image)
    
    print(f"Original image mean brightness: {original_mean:.1f}")
    print(f"Enhanced image mean brightness: {enhanced_mean:.1f}")
    print(f"Brightness change: {enhanced_mean - original_mean:.1f}")
    
    # Verify the enhancement is reasonable
    if abs(enhanced_mean - original_mean) < 50:  # Not too much change
        print("✓ Enhancement is balanced and reasonable")
    else:
        print("⚠ Enhancement might be too aggressive")
    
    print("✓ Test completed - check test_results/balanced_enhanced.png")

if __name__ == '__main__':
    test_balanced_enhancement()
