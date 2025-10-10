#!/usr/bin/env python3
"""
Demo script for the new interactive license plate selection feature
"""

import cv2
import numpy as np
import os

def create_demo_image():
    """Create a demo image with simulated license plates"""
    # Create a 800x600 image
    img = np.ones((600, 800, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Add some simulated license plates
    # Plate 1 (top left)
    cv2.rectangle(img, (100, 100), (300, 150), (50, 50, 50), -1)
    cv2.putText(img, "ABC-123", (120, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Plate 2 (top right)
    cv2.rectangle(img, (500, 120), (700, 170), (50, 50, 50), -1)
    cv2.putText(img, "XYZ-789", (520, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Plate 3 (bottom center)
    cv2.rectangle(img, (300, 400), (500, 450), (50, 50, 50), -1)
    cv2.putText(img, "DEF-456", (320, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add some haze effect
    haze = np.ones_like(img) * 100
    img = cv2.addWeighted(img, 0.7, haze, 0.3, 0)
    
    return img

def test_selection_processing():
    """Test the selection processing functionality"""
    print("Creating demo image...")
    demo_img = create_demo_image()
    
    # Save demo image
    demo_path = "demo_hazy_image.jpg"
    cv2.imwrite(demo_path, demo_img)
    print(f"Demo image saved to: {demo_path}")
    
    # Simulate selections (these would come from the web interface)
    selections = [
        {"x": 100, "y": 100, "width": 200, "height": 50},  # Plate 1
        {"x": 500, "y": 120, "width": 200, "height": 50},  # Plate 2
        {"x": 300, "y": 400, "width": 200, "height": 50},  # Plate 3
    ]
    
    print(f"\nSimulated selections: {len(selections)} regions")
    for i, sel in enumerate(selections):
        print(f"  Region {i+1}: x={sel['x']}, y={sel['y']}, w={sel['width']}, h={sel['height']}")
    
    # Process each selection
    print("\nProcessing selected regions...")
    for i, selection in enumerate(selections):
        # Extract region
        x, y, w, h = selection['x'], selection['y'], selection['width'], selection['height']
        region = demo_img[y:y+h, x:x+w]
        
        # Apply enhancement
        enhanced = enhance_selected_region(region)
        
        # Save enhanced region
        output_filename = f"demo_enhanced_area_{i+1}.png"
        cv2.imwrite(output_filename, enhanced)
        print(f"  Enhanced region {i+1} saved to: {output_filename}")
    
    print("\nDemo completed! Check the generated files:")
    print("  - demo_hazy_image.jpg (original hazy image)")
    print("  - demo_enhanced_area_1.png (enhanced license plate 1)")
    print("  - demo_enhanced_area_2.png (enhanced license plate 2)")
    print("  - demo_enhanced_area_3.png (enhanced license plate 3)")

def enhance_selected_region(region):
    """Apply enhancement to selected region (same as in Flask app)"""
    try:
        # Convert to float for processing
        region_float = region.astype(np.float32) / 255.0
        
        # Apply contrast enhancement
        enhanced = cv2.convertScaleAbs(region_float * 255, alpha=1.8, beta=40)
        
        # Apply sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Apply bilateral filter for noise reduction
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
        
    except Exception as e:
        print(f"Error enhancing region: {e}")
        return region

if __name__ == "__main__":
    print("IPC-Dehaze Interactive Selection Demo")
    print("=" * 40)
    
    test_selection_processing()
    
    print("\nTo test the full web interface:")
    print("1. Run: python app.py")
    print("2. Open: http://localhost:5000")
    print("3. Upload the demo_hazy_image.jpg")
    print("4. Use the selection tool to draw boxes around license plates")
    print("5. Process and see dual results!")
