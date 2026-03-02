#!/usr/bin/env python3
"""
Simple test script to verify the processing works
"""
import os
import sys
import subprocess

def test_processing():
    """Test the processing with a simple image"""
    print("Testing image processing...")
    
    # Check if we have test images
    test_dir = "examples"
    if not os.path.exists(test_dir):
        print(f"Error: {test_dir} directory not found")
        return False
    
    # Find a test image
    test_images = []
    for file in os.listdir(test_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            test_images.append(os.path.join(test_dir, file))
    
    if not test_images:
        print("Error: No test images found in examples directory")
        return False
    
    test_image = test_images[0]
    print(f"Using test image: {test_image}")
    
    # Create output directory
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run inference
    cmd = [
        sys.executable, "inference.py",
        "-i", test_image,
        "-o", output_dir,
        "-n", "8",
        "--max_size", "1500",
        "--enhance_plates"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Processing completed successfully!")
            
            # Check output files
            output_files = os.listdir(output_dir)
            print(f"Output files: {output_files}")
            
            # Check for specific files
            has_dehazed = any('_dehazed.png' in f for f in output_files)
            has_processed = any('processed_' in f for f in output_files)
            
            print(f"Dehazed image: {'✅' if has_dehazed else '❌'}")
            print(f"Processed image: {'✅' if has_processed else '❌'}")
            
            return True
        else:
            print(f"❌ Processing failed with return code: {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Processing timed out")
        return False
    except Exception as e:
        print(f"❌ Error running processing: {e}")
        return False

if __name__ == "__main__":
    success = test_processing()
    if success:
        print("\n🎉 Test passed! The processing is working correctly.")
    else:
        print("\n💥 Test failed! There are issues with the processing.")
        sys.exit(1)
