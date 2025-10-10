#!/usr/bin/env python3
"""
Test script to verify balanced license plate enhancement
"""

import cv2
import numpy as np
import os

def test_balanced_enhancement():
    """Test the balanced enhancement approach"""
    print("Testing balanced license plate enhancement...")
    
    # Create a realistic hazy license plate image
    test_image = np.zeros((150, 300, 3), dtype=np.uint8)
    
    # Add realistic background
    test_image[:, :] = [80, 80, 80]
    
    # Add a license plate region
    cv2.rectangle(test_image, (50, 50), (250, 100), (40, 40, 40), -1)
    
    # Add license plate text
    cv2.putText(test_image, "FA600CH", (70, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    
    # Add haze effect
    haze = np.ones_like(test_image) * 50
    test_image = cv2.addWeighted(test_image, 0.8, haze, 0.2, 0)
    
    # Test the balanced enhancement
    from app import enhance_selected_region
    
    # Apply balanced processing
    enhanced = enhance_selected_region(test_image)
    
    # Save results
    os.makedirs('test_results', exist_ok=True)
    cv2.imwrite('test_results/original_hazy.png', test_image)
    cv2.imwrite('test_results/balanced_enhanced.png', enhanced)
    
    # Check if enhancement is reasonable
    original_mean = np.mean(test_image)
    enhanced_mean = np.mean(enhanced)
    
    print(f"Original brightness: {original_mean:.1f}")
    print(f"Enhanced brightness: {enhanced_mean:.1f}")
    print(f"Brightness change: {enhanced_mean - original_mean:.1f}")
    
    # Verify the enhancement is balanced
    if abs(enhanced_mean - original_mean) < 30:  # Not too much change
        print("✓ Enhancement is balanced and should preserve details!")
    else:
        print("⚠ Enhancement might be too aggressive")
    
    print("✓ Test completed - check test_results/balanced_enhanced.png")
    print("  The license plate should be clearly readable without over-processing!")

if __name__ == '__main__':
    test_balanced_enhancement()
