#!/usr/bin/env python3
"""
Test script to verify license plate number visibility improvements
"""

import cv2
import numpy as np
import os

def test_license_plate_visibility():
    """Test the enhanced license plate visibility"""
    print("Testing license plate number visibility...")
    
    # Create a realistic hazy license plate image
    test_image = np.zeros((200, 400, 3), dtype=np.uint8)
    
    # Add realistic background (gray road)
    test_image[:, :] = [70, 70, 70]
    
    # Add a license plate region (darker rectangle)
    cv2.rectangle(test_image, (100, 80), (300, 140), (30, 30, 30), -1)
    
    # Add license plate text (white text on dark background)
    cv2.putText(test_image, "ABC123", (120, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)
    
    # Add realistic haze effect (making text hard to read)
    haze = np.ones_like(test_image) * 60
    test_image = cv2.addWeighted(test_image, 0.7, haze, 0.3, 0)
    
    # Add some noise to simulate real conditions
    noise = np.random.normal(0, 8, test_image.shape).astype(np.uint8)
    test_image = cv2.add(test_image, noise)
    
    # Test the enhanced license plate processing
    from app import enhance_selected_region
    
    # Apply enhanced processing
    enhanced = enhance_selected_region(test_image)
    
    # Save results
    os.makedirs('test_results', exist_ok=True)
    cv2.imwrite('test_results/original_hazy_plate.png', test_image)
    cv2.imwrite('test_results/enhanced_plate.png', enhanced)
    
    # Calculate visibility metrics
    original_mean = np.mean(test_image)
    enhanced_mean = np.mean(enhanced)
    
    # Calculate contrast in the license plate region
    plate_region_orig = test_image[80:140, 100:300]
    plate_region_enhanced = enhanced[80:140, 100:300]
    
    orig_contrast = np.std(plate_region_orig)
    enhanced_contrast = np.std(plate_region_enhanced)
    
    print(f"Original image mean brightness: {original_mean:.1f}")
    print(f"Enhanced image mean brightness: {enhanced_mean:.1f}")
    print(f"Brightness improvement: {enhanced_mean - original_mean:.1f}")
    print(f"Original plate contrast: {orig_contrast:.1f}")
    print(f"Enhanced plate contrast: {enhanced_contrast:.1f}")
    print(f"Contrast improvement: {enhanced_contrast - orig_contrast:.1f}")
    
    # Verify the enhancement is effective
    if enhanced_contrast > orig_contrast * 1.5:  # Significant contrast improvement
        print("✓ License plate numbers should be much more visible!")
    else:
        print("⚠ Enhancement might need adjustment")
    
    print("✓ Test completed - check test_results/enhanced_plate.png")
    print("  The license plate numbers should be clearly readable now!")

if __name__ == '__main__':
    test_license_plate_visibility()
