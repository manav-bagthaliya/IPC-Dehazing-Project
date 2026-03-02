#!/usr/bin/env python3
"""
Phase 2: License Plate Detection on Dehazed Images
This script performs license plate detection and enhancement on already-dehazed images.
It's designed to work with the clear images from Phase 1 for better detection accuracy.
"""

import os
import argparse
import glob
import cv2
import numpy as np
from tqdm import tqdm

# License plate detection imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available. Install with: pip install ultralytics")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: EasyOCR not available. Install with: pip install easyocr")

# Lazy-initialized OCR reader
_easyocr_reader = None

def recognize_plate_text(plate_bgr: np.ndarray) -> str:
    """Recognize license plate text using EasyOCR if available."""
    global _easyocr_reader
    if not EASYOCR_AVAILABLE or plate_bgr is None or plate_bgr.size == 0:
        return ''
    try:
        if _easyocr_reader is None:
            _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        plate_rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
        results = _easyocr_reader.readtext(plate_rgb, detail=0)
        if not results:
            return ''
        text = ' '.join(results)
        cleaned = ''.join(ch for ch in text if ch.isalnum() or ch in [' ', '-', '_'])
        return cleaned.strip()
    except Exception as _:
        return ''

def detect_license_plates_advanced(image):
    """Advanced license plate detection using multiple methods including color detection"""
    plates = []
    
    print(f"Starting license plate detection on dehazed image shape: {image.shape}")
    
    # Method 1: Color-based detection for yellow license plates (commercial vehicles)
    yellow_plates = detect_yellow_license_plates(image)
    print(f"Yellow plates found: {len(yellow_plates)}")
    plates.extend(yellow_plates)
    
    # Method 2: Color-based detection for white license plates (private vehicles) - HIGH PRIORITY
    white_plates = detect_white_license_plates(image)
    print(f"White plates found: {len(white_plates)}")
    plates.extend(white_plates)
    
    # Method 3: Enhanced white plate detection with better parameters
    white_plates_enhanced = detect_white_license_plates_enhanced(image)
    print(f"Enhanced white plates found: {len(white_plates_enhanced)}")
    plates.extend(white_plates_enhanced)
    
    # Method 4: Color-based detection for blue license plates (government vehicles)
    blue_plates = detect_blue_license_plates(image)
    print(f"Blue plates found: {len(blue_plates)}")
    plates.extend(blue_plates)
    
    # Method 5: Traditional OpenCV detection (fallback)
    opencv_plates = detect_license_plates_opencv(image)
    print(f"OpenCV plates found: {len(opencv_plates)}")
    plates.extend(opencv_plates)
    
    # Method 6: YOLO detection (if available)
    if YOLO_AVAILABLE:
        yolo_plates = detect_license_plates_yolo(image)
        print(f"YOLO plates found: {len(yolo_plates)}")
        plates.extend(yolo_plates)
    
    print(f"Total plates before deduplication: {len(plates)}")
    
    # Remove duplicates and return best detections
    final_plates = remove_duplicate_detections(plates)
    print(f"Final plates after deduplication: {len(final_plates)}")

    return final_plates

def detect_yellow_license_plates(image):
    """Detect yellow license plates using HSV color space"""
    plates = []
    
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow1 = np.array([15, 50, 50])
        upper_yellow1 = np.array([35, 255, 255])
        
        yellow_mask = cv2.inRange(hsv, lower_yellow1, upper_yellow1)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if 1.8 < aspect_ratio < 6.0:
                    plates.append({
                        'bbox': [x, y, x+w, y+h],
                        'confidence': 0.9,
                        'method': 'yellow_color',
                        'color': 'yellow'
                    })
        
    except Exception as e:
        print(f"Yellow plate detection error: {e}")
    
    return plates

def detect_white_license_plates(image):
    """Detect white license plates using HSV color space"""
    plates = []
    
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        white_ranges = [
            ([0, 0, 200], [180, 30, 255]),
            ([0, 0, 150], [180, 50, 255]),
            ([0, 0, 120], [180, 60, 255]),
        ]
        
        for i, (lower, upper) in enumerate(white_ranges):
            lower_white = np.array(lower)
            upper_white = np.array(upper)
            
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 800:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    if 1.2 < aspect_ratio < 8.0:
                        plates.append({
                            'bbox': [x, y, x+w, y+h],
                            'confidence': 0.85 - (i * 0.1),
                            'method': f'white_color_range_{i+1}',
                            'color': 'white'
                        })
        
        # Also try grayscale-based detection for white plates
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if 1.5 < aspect_ratio < 6.0:
                    plates.append({
                        'bbox': [x, y, x+w, y+h],
                        'confidence': 0.75,
                        'method': 'white_grayscale',
                        'color': 'white'
                    })
        
    except Exception as e:
        print(f"White plate detection error: {e}")
    
    return plates

def detect_white_license_plates_enhanced(image):
    """Enhanced white license plate detection with multiple approaches"""
    plates = []
    
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        white_ranges = [
            ([0, 0, 200], [180, 30, 255]),
            ([0, 0, 150], [180, 50, 255]),
            ([0, 0, 180], [180, 40, 255]),
        ]
        
        for lower, upper in white_ranges:
            lower_white = np.array(lower)
            upper_white = np.array(upper)
            
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    if 1.3 < aspect_ratio < 8.0:
                        plates.append({
                            'bbox': [x, y, x+w, y+h],
                            'confidence': 0.88,
                            'method': 'white_color_enhanced',
                            'color': 'white'
                        })
        
        # Method 2: Grayscale-based detection for white plates
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1500:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if 1.8 < aspect_ratio < 6.0:
                    plates.append({
                        'bbox': [x, y, x+w, y+h],
                        'confidence': 0.82,
                        'method': 'white_grayscale',
                        'color': 'white'
                    })
        
    except Exception as e:
        print(f"Enhanced white plate detection error: {e}")
    
    return plates

def detect_blue_license_plates(image):
    """Detect blue license plates using HSV color space"""
    plates = []
    
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1500:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if 1.8 < aspect_ratio < 6.0:
                    plates.append({
                        'bbox': [x, y, x+w, y+h],
                        'confidence': 0.8,
                        'method': 'blue_color',
                        'color': 'blue'
                    })
        
    except Exception as e:
        print(f"Blue plate detection error: {e}")
    
    return plates

def detect_license_plates_opencv(image):
    """Detect license plates using OpenCV methods"""
    plates = []
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 2.0 < aspect_ratio < 5.5:
                    plates.append({
                        'bbox': [x, y, x+w, y+h],
                        'confidence': 0.7,
                        'method': 'opencv',
                        'color': 'unknown'
                    })
    except Exception as e:
        print(f"OpenCV detection error: {e}")
    
    return plates

def detect_license_plates_yolo(image):
    """Detect license plates using YOLO if available"""
    if not YOLO_AVAILABLE:
        return []
    
    try:
        model_path = 'yolov8n.pt'
        if not os.path.exists(model_path):
            print(f"YOLO model not found at {model_path}")
            return []
            
        model = YOLO(model_path)
        results = model(image, verbose=False)
        plates = []
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                for box in boxes:
                    if box.conf > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        plates.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(box.conf[0]),
                            'method': 'yolo'
                        })
        
        return plates
    except Exception as e:
        print(f"YOLO detection failed: {e}")
        return []

def remove_duplicate_detections(plates):
    """Remove duplicate detections and return the best ones"""
    if not plates:
        return []
    
    plates.sort(key=lambda x: x['confidence'], reverse=True)
    
    unique_plates = []
    seen_bboxes = set()
    
    for plate in plates:
        bbox = tuple(plate['bbox'])
        
        is_duplicate = False
        for seen_bbox in seen_bboxes:
            if bbox_overlap(bbox, seen_bbox) > 0.3:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_plates.append(plate)
            seen_bboxes.add(bbox)
    
    return unique_plates[:5]

def bbox_overlap(bbox1, bbox2):
    """Calculate overlap ratio between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    return intersection / min(area1, area2)

def enhance_plate_region(image, plate_bbox):
    """Enhance the license plate region using proven character segmentation techniques"""
    x1, y1, x2, y2 = plate_bbox
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return image
    
    plate_region = image[y1:y2, x1:x2]
    
    if plate_region.size == 0:
        return image
    
    # Apply proven license plate enhancement techniques
    fixed_width = 400
    if plate_region.shape[1] != fixed_width:
        plate_region = cv2.resize(plate_region, (fixed_width, int(plate_region.shape[0] * fixed_width / plate_region.shape[1])))
    
    hsv = cv2.cvtColor(plate_region, cv2.COLOR_BGR2HSV)
    V = hsv[:,:,2]
    
    thresh = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 4)
    thresh = cv2.bitwise_not(thresh)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    thresh = cv2.GaussianBlur(thresh, (3, 3), 0)
    
    gaussian = cv2.GaussianBlur(thresh, (0, 0), 1.2)
    thresh = cv2.addWeighted(thresh, 1.5, gaussian, -0.5, 0)
    
    enhanced = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.15, beta=12)
    enhanced = cv2.bilateralFilter(enhanced, 5, 30, 30)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    return enhanced

def main():
    parser = argparse.ArgumentParser(description='Phase 2: License Plate Detection on Dehazed Images')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input dehazed images folder')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output folder for enhanced images and plate crops')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Find all dehazed images
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*_dehazed.png')))
    
    if not paths:
        print(f"No dehazed images found in {args.input}")
        return
    
    print(f"Found {len(paths)} dehazed images to process")
    
    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        imgname = os.path.splitext(os.path.basename(path))[0].replace('_dehazed', '')
        pbar.set_description(f'Processing {idx} {imgname}')
        
        # Load dehazed image
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Could not load image: {path}")
            continue
        
        print(f"\n=== Processing {imgname} ===")
        print(f"Image shape: {image.shape}")
        
        # Detect license plates on the clear dehazed image
        plates = detect_license_plates_advanced(image)
        
        if not plates:
            print("No license plates detected")
            # Save original dehazed image as final result
            output_path = os.path.join(args.output, f'{imgname}.png')
            cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"Saved original dehazed image: {output_path}")
        else:
            print(f"Detected {len(plates)} license plate(s)")
            
            # Create enhanced image with plate regions enhanced
            enhanced_image = image.copy()
            plate_crops = []
            
            for i, plate in enumerate(plates):
                bbox = plate['bbox']
                color = plate.get('color', 'unknown')
                method = plate.get('method', 'unknown')
                confidence = plate.get('confidence', 0.0)
                
                print(f"  Plate {i+1}: {color} plate detected by {method} (confidence: {confidence:.2f})")
                
                # Enhance the plate region
                enhanced_plate = enhance_plate_region(enhanced_image, bbox)
                
                # Try OCR on the enhanced plate
                plate_text = recognize_plate_text(enhanced_plate)
                if plate_text:
                    print(f"    Recognized text: {plate_text}")
                    plate['text'] = plate_text
                
                # Replace the enhanced plate back in the image
                x1, y1, x2, y2 = bbox
                if 0 <= y1 < y2 <= enhanced_image.shape[0] and 0 <= x1 < x2 <= enhanced_image.shape[1]:
                    target_h = y2 - y1
                    target_w = x2 - x1
                    if enhanced_plate.shape[0] != target_h or enhanced_plate.shape[1] != target_w:
                        enhanced_plate = cv2.resize(enhanced_plate, (target_w, target_h))
                    enhanced_image[y1:y2, x1:x2] = enhanced_plate
                    
                    # Save high-res crop for UI
                    crop = enhanced_image[y1:y2, x1:x2].copy()
                    target_w = 900
                    scale = target_w / max(1, crop.shape[1])
                    crop_resized = cv2.resize(crop, (int(crop.shape[1]*scale), int(crop.shape[0]*scale)))
                    plate_crops.append({
                        'crop': crop_resized,
                        'bbox': bbox,
                        'color': color,
                        'method': method,
                        'confidence': confidence,
                        'text': plate.get('text', '')
                    })
            
            # Save enhanced image
            output_path = os.path.join(args.output, f'{imgname}.png')
            cv2.imwrite(output_path, enhanced_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"Saved enhanced image: {output_path}")
            
            # Save plate crops
            for i, plate_crop in enumerate(plate_crops):
                crop_path = os.path.join(args.output, f'{imgname}_plate_{i+1}.png')
                cv2.imwrite(crop_path, plate_crop['crop'], [cv2.IMWRITE_PNG_COMPRESSION, 0])
                print(f"Saved plate crop: {crop_path}")
        
        pbar.update(1)
    
    pbar.close()
    print("\n=== Phase 2 Complete ===")
    print("License plate detection and enhancement finished on all dehazed images.")

if __name__ == '__main__':
    main()
