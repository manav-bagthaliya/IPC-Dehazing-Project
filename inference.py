import os
import argparse
import glob
import torch
import cv2
import numpy as np
from tqdm import tqdm
from basicsr.utils import imwrite, tensor2img, img2tensor, parallel_decode
from basicsr.archs.dehazeToken_arch import DehazeTokenNet, Critic

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

def validate_indian_license_plate(text: str) -> dict:
    """
    Validate Indian license plate format (XXNNXXNNNN)
    Returns detailed validation results with confidence score
    """
    result = {
        'is_valid': False,
        'formatted_text': '',
        'confidence': 0.0,
        'error': '',
        'segments': {}
    }
    
    if not text:
        result['error'] = 'Empty text'
        return result
    
    # Clean the text
    cleaned = text.replace(' ', '').replace('-', '').replace('.', '').upper()
    
    # Check length
    if len(cleaned) != 10:
        result['error'] = f'Invalid length: {len(cleaned)} (expected 10)'
        return result
    
    # Extract segments
    state_code = cleaned[0:2]      # First 2 characters (letters)
    district_code = cleaned[2:4]   # Next 2 characters (numbers)
    series = cleaned[4:6]         # Next 2 characters (letters)
    unique_number = cleaned[6:10] # Last 4 characters (numbers)
    
    # Validate each segment
    errors = []
    confidence_factors = []
    
    # Segment 1: State Code (2 letters)
    if not state_code.isalpha() or len(state_code) != 2:
        errors.append(f'State code invalid: {state_code}')
    else:
        confidence_factors.append(1.0)
        result['segments']['state_code'] = state_code
    
    # Segment 2: District Code (2 numbers)
    if not district_code.isdigit() or len(district_code) != 2:
        errors.append(f'District code invalid: {district_code}')
    else:
        confidence_factors.append(1.0)
        result['segments']['district_code'] = district_code
    
    # Segment 3: Series (2 letters)
    if not series.isalpha() or len(series) != 2:
        errors.append(f'Series invalid: {series}')
    else:
        confidence_factors.append(1.0)
        result['segments']['series'] = series
    
    # Segment 4: Unique Number (4 numbers)
    if not unique_number.isdigit() or len(unique_number) != 4:
        errors.append(f'Unique number invalid: {unique_number}')
    else:
        confidence_factors.append(1.0)
        result['segments']['unique_number'] = unique_number
    
    # Calculate confidence based on valid segments
    if len(errors) == 0:
        result['is_valid'] = True
        result['formatted_text'] = f"{state_code} {district_code} {series} {unique_number}"
        result['confidence'] = sum(confidence_factors) / len(confidence_factors)
        result['error'] = ''
    else:
        result['error'] = '; '.join(errors)
        result['confidence'] = len(confidence_factors) / 4.0  # Partial credit for valid segments
    
    return result

def validate_plate_format(text: str) -> bool:
    """Legacy function for backward compatibility"""
    result = validate_indian_license_plate(text)
    return result['is_valid']

def format_plate_text(text: str) -> str:
    """Legacy function for backward compatibility"""
    result = validate_indian_license_plate(text)
    return result['formatted_text'] if result['is_valid'] else text

def recognize_plate_text(plate_bgr: np.ndarray) -> str:
    """Recognize license plate text with OCR after aggressive, OCR-focused preprocessing.
    Produces a crisp, high-contrast plate image (B/W) for OCR only, keeps UI visuals unchanged.
    Applies Indian format validation (XXNNXXNNNN) and common-character corrections.
    Returns formatted text string or empty string on failure.
    """
    global _easyocr_reader
    if not EASYOCR_AVAILABLE or plate_bgr is None or plate_bgr.size == 0:
        return ''
    try:
        # Initialize OCR reader once
        if _easyocr_reader is None:
            _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)

        # 1) Prepare OCR-optimized grayscale image
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        # Normalize size for easier OCR (target width ~ 480px)
        h, w = gray.shape[:2]
        target_w = 480
        if w < target_w:
            scale = target_w / max(1, w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        # Contrast equalization (CLAHE) and unsharp masking
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(gray)
        blurred = cv2.GaussianBlur(l_eq, (0, 0), 1.2)
        sharp = cv2.addWeighted(l_eq, 1.6, blurred, -0.6, 0)

        # Adaptive threshold to get crisp B/W characters
        bw = cv2.adaptiveThreshold(
            sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )

        # Ensure black-on-white (invert if needed based on average)
        if np.mean(bw) < 127:
            bw = cv2.bitwise_not(bw)

        # Morphological cleanup (remove speckles, connect broken strokes)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 2) Run OCR (detail=1 to obtain confidences)
        ocr_results = _easyocr_reader.readtext(bw, detail=1, paragraph=False)
        if not ocr_results:
            return ''

        # Concatenate all alphanumeric tokens by confidence order
        ocr_results.sort(key=lambda x: float(x[2]) if len(x) > 2 else 0.0, reverse=True)
        raw_text = ''.join(''.join(ch for ch in r[1] if ch.isalnum()) for r in ocr_results)
        raw_text = raw_text.upper()

        # 3) Try direct validation first
        direct = validate_indian_license_plate(raw_text)
        if direct['is_valid']:
            return direct['formatted_text']

        # 4) Format-aware correction heuristics
        # Expected pattern: L L D D L L D D D D
        # Common confusions maps for letters/digits
        to_digit = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'}
        to_letter = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}

        candidates = []
        # Consider first best 12 alphanumerics, then trim to 10 later
        base = ''.join(ch for ch in raw_text if ch.isalnum())[:12]
        if base:
            candidates.append(base)

        def coerce_to_pattern(s: str) -> str:
            s = ''.join(ch for ch in s if ch.isalnum()).upper()
            if len(s) < 10:
                s = s.ljust(10, '0')
            s = s[:10]
            out = list(s)
            # positions 0,1,4,5 letters
            for pos in [0, 1, 4, 5]:
                ch = out[pos]
                if not ch.isalpha():
                    out[pos] = to_letter.get(ch, 'A')
            # positions 2,3,6,7,8,9 digits
            for pos in [2, 3, 6, 7, 8, 9]:
                ch = out[pos]
                if not ch.isdigit():
                    out[pos] = to_digit.get(ch, '0')
            return ''.join(out)

        # Generate a few coerced variants
        for s in [base, base.replace(' ', ''), base.replace('-', '')]:
            if s:
                candidates.append(coerce_to_pattern(s))

        # Also try character-wise mapped versions
        mapped_digits = ''.join(to_digit.get(ch, ch) for ch in base)
        mapped_letters = ''.join(to_letter.get(ch, ch) for ch in base)
        candidates += [coerce_to_pattern(mapped_digits), coerce_to_pattern(mapped_letters)]

        best_text = ''
        best_conf = -1.0
        for cand in candidates:
            res = validate_indian_license_plate(cand)
            if res['is_valid'] and res['confidence'] > best_conf:
                best_conf = res['confidence']
                best_text = res['formatted_text']

        if best_text:
            return best_text

        # Fallback: return cleaned raw if nothing validated
        cleaned = ''.join(ch for ch in raw_text if ch.isalnum())
        return cleaned.strip()[:10]
    except Exception as _:
        return ''

def detect_license_plates_advanced(image):
    """Advanced license plate detection using multiple methods including color detection"""
    plates = []
    
    print(f"Starting license plate detection on image shape: {image.shape}")
    
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
    
    # Method 6: Edge-based detection for better coverage
    edge_plates = detect_plates_by_edges(image)
    print(f"Edge-based plates found: {len(edge_plates)}")
    plates.extend(edge_plates)
    
    # Method 6: YOLO detection (if available)
    if YOLO_AVAILABLE:
        yolo_plates = detect_license_plates_yolo(image)
        print(f"YOLO plates found: {len(yolo_plates)}")
        plates.extend(yolo_plates)
    
    print(f"Total plates before deduplication: {len(plates)}")
    
    # Remove duplicates and return best detections
    final_plates = remove_duplicate_detections(plates)
    print(f"Final plates after deduplication: {len(final_plates)}")

    # If still none, try a last-pass high-contrast enhancement and re-run once
    if len(final_plates) == 0:
        try:
            print("No plates found; running last-pass enhancement for detection...")
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            retry = remove_duplicate_detections(
                detect_white_license_plates(enhanced_img) +
                detect_white_license_plates_enhanced(enhanced_img) +
                detect_yellow_license_plates(enhanced_img)
            )
            print(f"Retry detection found: {len(retry)}")
            if retry:
                return retry
        except Exception as _:
            pass

    return final_plates

def detect_yellow_license_plates(image):
    """Detect yellow license plates using HSV color space"""
    plates = []
    
    try:
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define yellow color range in HSV
        # Yellow license plates typically have hue around 20-30
        lower_yellow1 = np.array([15, 50, 50])
        upper_yellow1 = np.array([35, 255, 255])
        
        # Create mask for yellow regions
        yellow_mask = cv2.inRange(hsv, lower_yellow1, upper_yellow1)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:  # Filter by area
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # License plate aspect ratio check (typically 2.0 to 5.5)
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
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Multiple white color ranges for different lighting conditions
        white_ranges = [
            ([0, 0, 180], [180, 25, 255]),  # Pure white
            ([0, 0, 140], [180, 40, 255]),  # Off-white
            ([0, 0, 100], [180, 70, 255]),  # Light gray-white
            ([0, 0, 80], [180, 80, 255]),   # Even more sensitive
        ]
        
        for i, (lower, upper) in enumerate(white_ranges):
            lower_white = np.array(lower)
            upper_white = np.array(upper)
            
            # Create mask for white regions
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Even lower area threshold for better detection
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    if 1.0 < aspect_ratio < 10.0:  # Very broad aspect ratio range
                        plates.append({
                            'bbox': [x, y, x+w, y+h],
                            'confidence': 0.9 - (i * 0.15),  # Higher confidence for better detection
                            'method': f'white_color_range_{i+1}',
                            'color': 'white'
                        })
        
        # Also try grayscale-based detection for white plates
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive threshold for better white plate detection
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
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
        # Method 1: HSV-based detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Multiple white color ranges for different lighting conditions
        white_ranges = [
            ([0, 0, 200], [180, 30, 255]),  # Pure white
            ([0, 0, 150], [180, 50, 255]),  # Off-white
            ([0, 0, 180], [180, 40, 255]),  # Bright white
        ]
        
        for lower, upper in white_ranges:
            lower_white = np.array(lower)
            upper_white = np.array(upper)
            
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Even lower threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    if 1.3 < aspect_ratio < 8.0:  # Very broad aspect ratio range
                        plates.append({
                            'bbox': [x, y, x+w, y+h],
                            'confidence': 0.88,
                            'method': 'white_color_enhanced',
                            'color': 'white'
                        })
        
        # Method 2: Grayscale-based detection for white plates
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold for bright regions (white plates)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
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
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define blue color range in HSV
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create mask for blue regions
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1500:  # Filter by area
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

def remove_duplicate_detections(plates):
    """Remove duplicate detections and return the best ones"""
    if not plates:
        return []
    
    # Sort by confidence
    plates.sort(key=lambda x: x['confidence'], reverse=True)
    
    unique_plates = []
    seen_bboxes = set()
    
    for plate in plates:
        bbox = tuple(plate['bbox'])
        
        # Check if this bbox overlaps significantly with existing ones
        is_duplicate = False
        for seen_bbox in seen_bboxes:
            if bbox_overlap(bbox, seen_bbox) > 0.3:  # 30% overlap threshold
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_plates.append(plate)
            seen_bboxes.add(bbox)
    
    return unique_plates[:5]  # Return top 5 detections

def detect_license_plates_opencv(image):
    """Detect license plates using OpenCV methods"""
    plates = []
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter by area
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 2.0 < aspect_ratio < 5.5:  # License plate aspect ratio
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
        # Check if model file exists
        model_path = 'yolov8n.pt'
        if not os.path.exists(model_path):
            print(f"YOLO model not found at {model_path}")
            return []
            
        # Load pre-trained model (you can replace with custom license plate model)
        model = YOLO(model_path)
        results = model(image, verbose=False)
        plates = []
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                for box in boxes:
                    if box.conf > 0.5:  # Confidence threshold
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

def enhance_plate_region(image, plate_bbox):
    """Enhance the license plate region while preserving natural colors"""
    x1, y1, x2, y2 = plate_bbox
    
    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return image
    
    # Extract plate region
    plate_region = image[y1:y2, x1:x2]
    
    if plate_region.size == 0:
        return image
    
    # Step 1: Resize for consistent processing
    fixed_width = 400
    if plate_region.shape[1] != fixed_width:
        plate_region = cv2.resize(plate_region, (fixed_width, int(plate_region.shape[0] * fixed_width / plate_region.shape[1])))
    
    # Step 2: Apply gentle contrast enhancement to make text more readable
    enhanced = cv2.convertScaleAbs(plate_region, alpha=1.2, beta=10)
    
    # Step 3: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Step 4: Apply gentle sharpening to enhance text edges
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel * 0.3) + enhanced * 0.7
    
    # Step 5: Apply bilateral filter to reduce noise while preserving edges
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Step 6: Final contrast adjustment
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
    
    # Ensure values are in valid range
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    return enhanced

def process_with_license_plate_detection(img_tensor, output_img_tensor, device):
    """
    Sequential Processing Workflow:
    1. Fog Removal (Dehazing) - Already done in main processing
    2. License Plate Detection - Using dehazed image
    3. OCR with Indian Format Validation
    """
    try:
        # Convert tensors to numpy arrays
        output_img_np = tensor2img(output_img_tensor, rgb2bgr=True)
        original_img_np = tensor2img(img_tensor, rgb2bgr=True)
        
        print("=" * 60)
        print("SEQUENTIAL IMAGE PROCESSING WORKFLOW")
        print("=" * 60)
        
        # Step 1: Fog Removal (Dehazing) - Already completed
        print("✅ Step 1: Fog Removal (Dehazing) - COMPLETED")
        print(f"   Input image shape: {original_img_np.shape}")
        print(f"   Dehazed image shape: {output_img_np.shape}")
        
        # Step 2: License Plate Detection using dehazed image
        print("\n🔍 Step 2: License Plate Detection")
        plates = detect_license_plates_advanced(output_img_np)  # Use dehazed image
        print(f"   Detected {len(plates)} potential license plates")
        
        # Step 3: OCR and Indian Format Validation
        print("\n📝 Step 3: OCR and Format Validation")
        enhanced_count = 0
        valid_plates = []
        
        for i, plate in enumerate(plates):
            bbox = plate['bbox']
            color = plate.get('color', 'unknown')
            method = plate.get('method', 'unknown')
            confidence = plate.get('confidence', 0.0)
            
            print(f"\n   Plate {i+1}: {color} plate ({method}) - Confidence: {confidence:.2f}")
            
            # Enhance the plate region for better OCR
            enhanced_plate = enhance_plate_region(output_img_np, bbox)
            
            # Perform OCR
            plate_text = recognize_plate_text(enhanced_plate)
            print(f"   Raw OCR text: '{plate_text}'")
            
            # Validate Indian format and get confidence
            validation_result = validate_indian_license_plate(plate_text)
            
            if validation_result['is_valid']:
                print(f"   ✅ Valid Indian format: {validation_result['formatted_text']}")
                print(f"   📊 Format confidence: {validation_result['confidence']:.2f}")
                
                valid_plates.append({
                    'bbox': bbox,
                    'raw_text': plate_text,
                    'formatted_text': validation_result['formatted_text'],
                    'confidence': validation_result['confidence'],
                    'color': color,
                    'method': method,
                    'segments': validation_result['segments']
                })
                
                # Replace the enhanced plate back in the image
                x1, y1, x2, y2 = bbox
                if 0 <= y1 < y2 <= output_img_np.shape[0] and 0 <= x1 < x2 <= output_img_np.shape[1]:
                    target_h = y2 - y1
                    target_w = x2 - x1
                    if enhanced_plate.shape[0] != target_h or enhanced_plate.shape[1] != target_w:
                        enhanced_plate = cv2.resize(enhanced_plate, (target_w, target_h))
                    output_img_np[y1:y2, x1:x2] = enhanced_plate
                    enhanced_count += 1
            else:
                print(f"   ❌ Invalid format: {validation_result['error']}")
        
        # Final Results Summary
        print("\n" + "=" * 60)
        print("FINAL PROCESSING RESULTS")
        print("=" * 60)
        
        if valid_plates:
            print(f"✅ Successfully processed {len(valid_plates)} valid license plate(s):")
            for i, plate in enumerate(valid_plates):
                print(f"   Plate {i+1}: {plate['formatted_text']}")
                print(f"   Confidence: {plate['confidence']:.2f}")
                print(f"   Segments: {plate['segments']}")
                # Emit machine-readable JSON line for the Flask app to parse
                try:
                    import json as _json
                    _summary = {
                        'formatted_text': plate['formatted_text'],
                        'confidence': float(plate['confidence']),
                        'segments': plate.get('segments', {}),
                        'bbox': plate.get('bbox', None),
                        'method': plate.get('method', 'unknown'),
                        'color': plate.get('color', 'unknown')
                    }
                    print('PLATE_JSON:' + _json.dumps(_summary, ensure_ascii=False))
                except Exception:
                    pass
        else:
            print("⚠️ No valid Indian license plates detected")
        
        print("=" * 60)
        
        # Convert back to tensor
        return img2tensor(output_img_np)
    except Exception as e:
        print(f"Error in sequential processing: {e}")
        import traceback
        traceback.print_exc()
        return output_img_tensor

def detect_plates_by_edges(image):
    """Detect license plates using edge detection and contour analysis"""
    plates = []
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (license plates are typically wider than tall)
            aspect_ratio = w / h
            if 2.0 <= aspect_ratio <= 6.0 and w > 50 and h > 15:
                # Check if the region has enough edges (text-like features)
                roi = edges[y:y+h, x:x+w]
                edge_density = np.sum(roi > 0) / (w * h)
                
                if edge_density > 0.1:  # At least 10% edge pixels
                    plates.append({
                        'bbox': (x, y, x+w, y+h),
                        'confidence': min(0.8, edge_density * 2),
                        'method': 'edge_detection',
                        'color': 'unknown'
                    })
        
        print(f"Edge-based detection found {len(plates)} potential plates")
        
    except Exception as e:
        print(f"Error in edge-based detection: {e}")
    
    return plates

def bbox_overlap(bbox1, bbox2):
    """Calculate overlap ratio between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
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

def enhance_output_quality(image):
    """Apply advanced quality enhancement to the dehazed output"""
    try:
        # Step 1: Convert to LAB color space for better enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Step 2: Apply stronger CLAHE to L channel for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Step 3: Apply gamma correction for better brightness
        l = np.power(l / 255.0, 0.85) * 255.0
        l = np.clip(l, 0, 255).astype(np.uint8)
        
        # Step 4: Merge channels back
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Step 5: Apply unsharp masking for better sharpness
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
        enhanced = cv2.addWeighted(enhanced, 1.3, gaussian, -0.3, 0)
        
        # Step 6: Apply bilateral filter for noise reduction while preserving edges
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Step 7: Final contrast adjustment
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.15, beta=15)
        
        # Ensure values are in valid range
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
        
    except Exception as e:
        print(f"Error enhancing output quality: {e}")
        return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictor_path',type=str,default='pretrained_models/predictor.pth')
    parser.add_argument('--critic_path',type=str,default=  'pretrained_models/critic.pth')
    parser.add_argument('-i','--input', type=str, default='examples', help='input test image folder')
    parser.add_argument('-o','--output', type=str, default='results', help='output test image folder')
    parser.add_argument('-n', type=int, default=8, help='num_iterations')
    parser.add_argument('--max_size', type=int, default=1500, help='max_size')
    parser.add_argument('--resize_to', type=int, default=0, help='If > 0, upscale inputs whose long edge is smaller than this value to this size before processing')
    parser.add_argument('--force_width', type=int, default=0, help='If > 0 and used with --force_height, resize ALL inputs to this width before processing')
    parser.add_argument('--force_height', type=int, default=0, help='If > 0 and used with --force_width, resize ALL inputs to this height before processing')
    parser.add_argument('--enhance_plates', action='store_true', help='Enable license plate enhancement')
    parser.add_argument('--dehaze_only', action='store_true', help='Only perform dehazing, skip plate detection')
    
    args = parser.parse_args()
    
    num_iterations = args.n
    os.makedirs(args.output, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net_g = DehazeTokenNet(codebook_params=[64, 1024, 256], blk_depth=16, LQ_stage=True, predictor_name='swinLayer').to(device)
    net_g.load_state_dict(torch.load(args.predictor_path, map_location=device)['params'], strict=True)
    net_g.eval()
    
    net_critic= Critic().to(device)
    net_critic.load_state_dict(torch.load(args.critic_path, map_location=device)['params'], strict=True)
    net_critic.eval()

    def tokens_to_logits(seq: torch.Tensor,h=0,w=0,critic=False) -> torch.Tensor:
        if critic:
            logits=net_critic(seq,h,w)
        else:
            logits = net_g.transformer(seq,critic)
        return logits 
    
    def tokens_to_feats(seq:torch.Tensor)->torch.Tensor:
        '''
        seq :b, 1, h, w
        '''
        feats = net_g.vqgan.quantize.get_codebook_entry(seq.long())
        return feats

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))
   
    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        imgname = os.path.splitext(os.path.basename(path))[0]
        pbar.set_description(f'Test {idx} {imgname}')
        
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Error: Could not load image {path}")
                continue
            img = img/255.0
            img = img2tensor(img)
            img = img.unsqueeze(0).to(device)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            continue

        try:
            with torch.no_grad():
                _,_,origin_h,origin_w = img.shape

                # Determine resizing strategy
                forced = (args.force_width > 0 and args.force_height > 0)
                if forced:
                    # Resize ALL inputs to the forced target size
                    img = torch.nn.UpsamplingBilinear2d((args.force_height, args.force_width))(img)
                else:
                    # Optionally upscale small images to a target long-edge size
                    if args.resize_to and max(origin_h, origin_w) < args.resize_to:
                        scale = args.resize_to / max(origin_h, origin_w)
                        img = torch.nn.UpsamplingBilinear2d(scale_factor=scale)(img)

                    # Downscale overlarge images for compute/memory
                    if origin_h*origin_w >= args.max_size**2:
                        scale = args.max_size/max(origin_h, origin_w)
                        img = torch.nn.UpsamplingBilinear2d(scale_factor=scale)(img)
    
                # padding to multiple of window_size * 8 (guard against None shapes)
                wsz = 32
                try:
                    _, _, h_old, w_old = img.shape
                except Exception:
                    img = img if len(img.shape) == 4 else img.unsqueeze(0)
                    _, _, h_old, w_old = img.shape
                h_pad = (h_old // wsz + 1) * wsz - h_old
                w_pad = (w_old // wsz + 1) * wsz - w_old
                img = torch.cat([img, torch.flip(img, [2])],
                                2)[:, :, :h_old + h_pad, :]
                img = torch.cat([img, torch.flip(img, [3])],
                                3)[:, :, :, :w_old + w_pad]
    
                ################### Encoder #####################
                try:
                    enc_feats = net_g.vqgan.multiscale_encoder(img.detach())
                except Exception as e:
                    print(f"Encoder failed: {e}")
                    # Try fallback to CPU execution for this step
                    try:
                        enc_feats = net_g.vqgan.multiscale_encoder(img.detach().to('cpu'))
                        img = img.to('cpu')
                    except Exception as _:
                        raise
                enc_feats = enc_feats[::-1]

                # Check if enc_feats is not empty
                if len(enc_feats) == 0:
                    print(f"Error: No encoder features generated for {imgname}")
                    continue
                
                x = enc_feats[0]
                b, c, h, w = x.shape

                feat_to_quant = net_g.vqgan.before_quant(x)
                _,_,lq_tokens=net_g.vqgan.encode(img)
                
                mask_tokens=-1*torch.ones(b,h*w).to(feat_to_quant.device).long()

                output_tokens,mask_tokens = parallel_decode.decode_critic(
                    mask_tokens,
                    feat_to_quant,
                    tokens_to_logits,
                    tokens_to_feats,
                    num_iter=num_iterations,
                    )
                
                quant_feats=tokens_to_feats(output_tokens[:, -1,:].reshape(b,1,h,w))

                # ################## Generator ####################
                after_quant_feat = net_g.vqgan.after_quant(quant_feats)

                x=after_quant_feat
                for i in range(net_g.max_depth):
                    cur_res = net_g.gt_res // 2**net_g.max_depth * 2**i
                    
                    # Check if we have enough encoder features
                    if i >= len(enc_feats):
                        print(f"Warning: Not enough encoder features for depth {i}, skipping")
                        break
                        
                    x = net_g.fuse_convs_dict[str(cur_res)](enc_feats[i].detach(), x, 1)
                    x = net_g.vqgan.decoder_group[i](x)
                    
                output_img = net_g.vqgan.out_conv(x)
                output_img = output_img[..., :h_old , :w_old ]
                
                # License plate enhancement (only if not dehaze_only mode)
                if args.enhance_plates and not args.dehaze_only:
                    output_img = process_with_license_plate_detection(img, output_img, device)
                
                # Preserve consistent output size and quality
                if forced:
                    # Ensure exact forced output size
                    if len(output_img.shape) == 4:
                        output_img = torch.nn.UpsamplingBilinear2d((args.force_height, args.force_width))(output_img)
                    else:
                        print(f"Warning: Unexpected tensor shape {output_img.shape} for forced resize")
                else:
                    # If we downscaled due to exceeding max_size, upscale back to original size
                    if origin_h*origin_w >= args.max_size**2:
                        if len(output_img.shape) == 4:
                            output_img = torch.nn.UpsamplingBilinear2d((origin_h, origin_w))(output_img)
                        else:
                            print(f"Warning: Unexpected tensor shape {output_img.shape} for max_size resize")
                    # If we upscaled small images, keep the upscaled size for better quality
                    elif args.resize_to and max(origin_h, origin_w) < args.resize_to:
                        # Keep the upscaled size for better quality
                        pass
                    else:
                        # Ensure we return to original size if no resizing was done
                        if output_img.shape[2:] != (origin_h, origin_w):
                            # Check if tensor has correct dimensions (B, C, H, W)
                            if len(output_img.shape) == 4:
                                output_img = torch.nn.UpsamplingBilinear2d((origin_h, origin_w))(output_img)
                            else:
                                print(f"Warning: Unexpected tensor shape {output_img.shape}, skipping resize")
                
                # Save with high quality settings
                output_img_np = tensor2img(output_img, rgb2bgr=True)

                # Phase 1: save dehazed (pre-plate enhancement) image for UI
                if args.dehaze_only:
                    # In dehaze_only mode, save the clean dehazed image
                    dehazed_path = os.path.join(args.output, f'{imgname}_dehazed.png')
                    cv2.imwrite(dehazed_path, output_img_np, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    print(f"Saved dehazed image: {dehazed_path}")
                else:
                    # In normal mode, save both dehazed and final processed versions
                    dehazed_np = output_img_np.copy()
                    dehazed_path = os.path.join(args.output, f'{imgname}_dehazed.png')
                    cv2.imwrite(dehazed_path, dehazed_np, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    print(f"Saved dehazed image: {dehazed_path}")
                
                # Apply final quality enhancement to the output
                output_img_np = enhance_output_quality(output_img_np)

                # Guard: if image looks blank (very low variance or too bright/too dark), fallback
                try:
                    variance = float(np.var(output_img_np))
                    mean_val = float(np.mean(output_img_np))
                    if variance < 1.0 or mean_val > 250 or mean_val < 5:
                        # Fallback to original-sized input to avoid white/black outputs
                        print(f"Warning: Low-content output detected (var={variance:.3f}, mean={mean_val:.1f}). Falling back to original image.")
                        output_img_np = tensor2img(img if forced else torch.nn.UpsamplingBilinear2d((origin_h, origin_w))(img), rgb2bgr=True)
                except Exception as _:
                    pass
                
                # Save with high quality settings
                if not args.dehaze_only:
                    output_path = os.path.join(args.output, f'{imgname}.png')
                    cv2.imwrite(output_path, output_img_np, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                # Save resized input for reference so user can see the processing size
                resized_inputs_dir = os.path.join(args.output, 'resized_inputs')
                os.makedirs(resized_inputs_dir, exist_ok=True)
                cv2.imwrite(os.path.join(resized_inputs_dir, f'{imgname}.png'), tensor2img(img, rgb2bgr=True))

        except Exception as e:
            print(f"Error processing {imgname}: {e}")
            import traceback
            traceback.print_exc()
            continue

        pbar.update(1)
           
    pbar.close()


if __name__ == '__main__':
    main()
