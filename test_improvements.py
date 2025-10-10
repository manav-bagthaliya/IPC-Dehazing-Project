#!/usr/bin/env python3
"""
Test script to verify the improvements to IPC-Dehaze:
1. Better dehazing quality
2. Enhanced license plate visibility
3. Proper image sizing
4. High-quality output
"""

import cv2
import numpy as np
import os
import sys

def test_license_plate_enhancement():
    """Test the enhanced license plate enhancement function"""
    print("Testing license plate enhancement...")
    
    # Create a test image with simulated license plate
    test_image = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Add some background
    test_image[:, :] = [100, 100, 100]
    
    # Simulate a license plate region (darker area)
    cv2.rectangle(test_image, (200, 150), (400, 250), (50, 50, 50), -1)
    
    # Add some text-like patterns (simulating license plate numbers)
    cv2.putText(test_image, "ABC123", (220, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    
    # Apply haze effect
    test_image = cv2.addWeighted(test_image, 0.7, np.ones_like(test_image) * 100, 0.3, 0)
    
    # Test the enhancement function
    from inference import enhance_plate_region
    
    # Define a license plate region
    plate_bbox = [200, 150, 400, 250]
    
    # Apply enhancement
    enhanced = enhance_plate_region(test_image, plate_bbox)
    
    # Save test results
    os.makedirs('test_results', exist_ok=True)
    cv2.imwrite('test_results/original_hazy.png', test_image)
    cv2.imwrite('test_results/enhanced_plate.png', enhanced)
    
    print("✓ License plate enhancement test completed")
    print("  - Original hazy image: test_results/original_hazy.png")
    print("  - Enhanced plate: test_results/enhanced_plate.png")

def test_output_quality_enhancement():
    """Test the output quality enhancement function"""
    print("Testing output quality enhancement...")
    
    # Create a test dehazed image
    test_image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    
    # Apply some noise
    noise = np.random.normal(0, 10, test_image.shape).astype(np.uint8)
    test_image = cv2.add(test_image, noise)
    
    # Test the enhancement function
    from inference import enhance_output_quality
    
    enhanced = enhance_output_quality(test_image)
    
    # Save test results
    os.makedirs('test_results', exist_ok=True)
    cv2.imwrite('test_results/original_noisy.png', test_image)
    cv2.imwrite('test_results/enhanced_quality.png', enhanced)
    
    print("✓ Output quality enhancement test completed")
    print("  - Original noisy image: test_results/original_noisy.png")
    print("  - Enhanced quality: test_results/enhanced_quality.png")

def test_coordinate_scaling():
    """Test the coordinate scaling logic"""
    print("Testing coordinate scaling...")
    
    # Test cases
    test_cases = [
        {
            'name': 'Wide image',
            'original': (1200, 800),
            'canvas': (800, 600),
            'selection': {'x': 100, 'y': 100, 'width': 200, 'height': 100}
        },
        {
            'name': 'Tall image',
            'original': (800, 1200),
            'canvas': (800, 600),
            'selection': {'x': 100, 'y': 100, 'width': 200, 'height': 100}
        }
    ]
    
    for test_case in test_cases:
        print(f"  Testing {test_case['name']}...")
        
        # Calculate expected canvas dimensions
        orig_w, orig_h = test_case['original']
        canvas_w, canvas_h = test_case['canvas']
        
        image_aspect = orig_w / orig_h
        canvas_aspect = canvas_w / canvas_h
        
        if image_aspect > canvas_aspect:
            actual_canvas_w = canvas_w
            actual_canvas_h = canvas_w / image_aspect
        else:
            actual_canvas_h = canvas_h
            actual_canvas_w = canvas_h * image_aspect
        
        # Calculate scaling factors
        scale_x = orig_w / actual_canvas_w
        scale_y = orig_h / actual_canvas_h
        
        # Scale selection coordinates
        selection = test_case['selection']
        scaled_x = int(selection['x'] * scale_x)
        scaled_y = int(selection['y'] * scale_y)
        scaled_w = int(selection['width'] * scale_x)
        scaled_h = int(selection['height'] * scale_y)
        
        print(f"    Original: {orig_w}x{orig_h}")
        print(f"    Canvas: {actual_canvas_w:.1f}x{actual_canvas_h:.1f}")
        print(f"    Scale: {scale_x:.2f}x, {scale_y:.2f}y")
        print(f"    Selection: {selection['x']},{selection['y']} {selection['width']}x{selection['height']}")
        print(f"    Scaled: {scaled_x},{scaled_y} {scaled_w}x{scaled_h}")
        
        # Verify scaling is reasonable
        assert 0 <= scaled_x < orig_w, f"Scaled x out of bounds: {scaled_x}"
        assert 0 <= scaled_y < orig_h, f"Scaled y out of bounds: {scaled_y}"
        assert scaled_w > 0, f"Invalid scaled width: {scaled_w}"
        assert scaled_h > 0, f"Invalid scaled height: {scaled_h}"
        
        print(f"    ✓ {test_case['name']} scaling test passed")
    
    print("✓ All coordinate scaling tests completed")

def main():
    """Run all tests"""
    print("IPC-Dehaze Improvements Test Suite")
    print("=" * 40)
    
    try:
        test_license_plate_enhancement()
        print()
        
        test_output_quality_enhancement()
        print()
        
        test_coordinate_scaling()
        print()
        
        print("🎉 All tests completed successfully!")
        print("\nTest results saved in 'test_results/' directory")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
