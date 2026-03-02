#!/usr/bin/env python3
"""
Debug script to identify list index out of range errors
"""

import os
import sys
import traceback

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic functionality to identify where the error occurs"""
    print("🔍 Debugging List Index Out of Range Error")
    print("=" * 50)
    
    try:
        # Test 1: Import inference module
        print("1. Testing imports...")
        from inference import detect_license_plates_advanced
        print("   ✓ Inference module imported successfully")
        
        # Test 2: Test with a simple image
        print("\n2. Testing with simple image...")
        import cv2
        import numpy as np
        
        # Create a simple test image
        test_img = np.ones((300, 400, 3), dtype=np.uint8) * 128
        
        # Test detection
        plates = detect_license_plates_advanced(test_img)
        print(f"   ✓ Detection completed: {len(plates)} plates found")
        
        # Test 3: Test Flask app imports
        print("\n3. Testing Flask app imports...")
        from app import app
        print("   ✓ Flask app imported successfully")
        
        print("\n✅ All basic tests passed!")
        return True
        
    except IndexError as e:
        print(f"\n❌ IndexError found: {e}")
        print("Stack trace:")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ Other error found: {e}")
        print("Stack trace:")
        traceback.print_exc()
        return False

def test_with_real_image():
    """Test with a real image if available"""
    print("\n4. Testing with real image...")
    
    try:
        import cv2
        import numpy as np
        from inference import detect_license_plates_advanced
        
        # Look for any image in the examples folder
        examples_dir = "examples"
        if os.path.exists(examples_dir):
            image_files = [f for f in os.listdir(examples_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if image_files:
                test_image_path = os.path.join(examples_dir, image_files[0])
                print(f"   Testing with: {test_image_path}")
                
                # Load image
                img = cv2.imread(test_image_path)
                if img is not None:
                    print(f"   ✓ Image loaded: {img.shape}")
                    
                    # Test detection
                    plates = detect_license_plates_advanced(img)
                    print(f"   ✓ Detection completed: {len(plates)} plates found")
                    
                    for i, plate in enumerate(plates):
                        print(f"   Plate {i+1}: {plate.get('color', 'unknown')} plate, confidence: {plate.get('confidence', 0):.2f}")
                    
                    return True
                else:
                    print(f"   ⚠️ Could not load image: {test_image_path}")
            else:
                print("   ⚠️ No image files found in examples folder")
        else:
            print("   ⚠️ Examples folder not found")
            
    except IndexError as e:
        print(f"\n❌ IndexError with real image: {e}")
        print("Stack trace:")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ Other error with real image: {e}")
        print("Stack trace:")
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("🚗 IPC-Dehaze Debug Script")
    print("=" * 30)
    
    # Run basic tests
    basic_success = test_basic_functionality()
    
    if basic_success:
        # Run real image tests
        real_success = test_with_real_image()
        
        if real_success:
            print("\n🎉 All tests passed! No list index errors found.")
        else:
            print("\n⚠️ Some issues found with real images.")
    else:
        print("\n❌ Basic functionality has issues.")
    
    print("\n" + "=" * 50)
    print("Debug complete!")
