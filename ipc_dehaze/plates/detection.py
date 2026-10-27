"""Licence plate localisation: colour (HSV), edge/contour and optional YOLO detectors.

Every detector returns a list of dicts: ``{'bbox': [x1, y1, x2, y2], 'confidence',
'method', 'color'}``. ``detect_license_plates_advanced`` runs them all and merges.
"""

import os

import cv2
import numpy as np

from ipc_dehaze.config import MAX_PLATES, YOLO_PATH

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available. Install with: pip install ultralytics")

# Loaded once on first use
_yolo_model = None

_KERNEL_3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
_KERNEL_5x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))


def _clean_mask(mask, kernel=_KERNEL_3x3):
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _boxes_from_mask(mask, min_area, aspect_range, confidence, method, color):
    """Bounding boxes of mask blobs whose area and aspect ratio look like a plate."""
    plates = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) <= min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if aspect_range[0] < w / h < aspect_range[1]:
            plates.append({
                'bbox': [x, y, x + w, y + h],
                'confidence': confidence,
                'method': method,
                'color': color
            })
    return plates


def detect_yellow_license_plates(image):
    """Yellow plates (commercial vehicles) via HSV thresholding."""
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = _clean_mask(cv2.inRange(hsv, np.array([15, 50, 50]), np.array([35, 255, 255])))
        return _boxes_from_mask(mask, 2000, (1.8, 6.0), 0.9, 'yellow_color', 'yellow')
    except Exception as e:
        print(f"Yellow plate detection error: {e}")
        return []


def detect_white_license_plates(image):
    """White plates (private vehicles): several HSV ranges plus adaptive-threshold pass."""
    plates = []
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white_ranges = [
            ([0, 0, 180], [180, 25, 255]),  # pure white
            ([0, 0, 140], [180, 40, 255]),  # off-white
            ([0, 0, 100], [180, 70, 255]),  # light grey-white
            ([0, 0, 80], [180, 80, 255]),   # most sensitive
        ]
        for i, (lower, upper) in enumerate(white_ranges):
            mask = _clean_mask(cv2.inRange(hsv, np.array(lower), np.array(upper)))
            plates += _boxes_from_mask(mask, 500, (1.0, 10.0), 0.9 - i * 0.15,
                                       f'white_color_range_{i + 1}', 'white')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh = _clean_mask(thresh, _KERNEL_5x3)
        plates += _boxes_from_mask(thresh, 1000, (1.5, 6.0), 0.75, 'white_grayscale', 'white')
    except Exception as e:
        print(f"White plate detection error: {e}")
    return plates


def detect_white_license_plates_enhanced(image):
    """Second white-plate pass with stricter HSV ranges plus a global bright threshold."""
    plates = []
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white_ranges = [
            ([0, 0, 200], [180, 30, 255]),  # pure white
            ([0, 0, 150], [180, 50, 255]),  # off-white
            ([0, 0, 180], [180, 40, 255]),  # bright white
        ]
        for lower, upper in white_ranges:
            mask = _clean_mask(cv2.inRange(hsv, np.array(lower), np.array(upper)))
            plates += _boxes_from_mask(mask, 1000, (1.3, 8.0), 0.88, 'white_color_enhanced', 'white')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        thresh = _clean_mask(thresh, _KERNEL_5x3)
        plates += _boxes_from_mask(thresh, 1500, (1.8, 6.0), 0.82, 'white_grayscale', 'white')
    except Exception as e:
        print(f"Enhanced white plate detection error: {e}")
    return plates


def detect_blue_license_plates(image):
    """Blue plates (government / embassy vehicles) via HSV thresholding."""
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = _clean_mask(cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255])))
        return _boxes_from_mask(mask, 1500, (1.8, 6.0), 0.8, 'blue_color', 'blue')
    except Exception as e:
        print(f"Blue plate detection error: {e}")
        return []


def detect_license_plates_opencv(image):
    """Blur + wide horizontal closing, then plate-shaped contours."""
    try:
        gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        return _boxes_from_mask(gray, 1000, (2.0, 5.5), 0.7, 'opencv', 'unknown')
    except Exception as e:
        print(f"OpenCV detection error: {e}")
        return []


def detect_plates_by_edges(image):
    """Canny edges; keeps plate-shaped boxes with enough edge density (text-like)."""
    plates = []
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 2.0 <= w / h <= 6.0 and w > 50 and h > 15:
                edge_density = np.sum(edges[y:y + h, x:x + w] > 0) / (w * h)
                if edge_density > 0.1:
                    plates.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': min(0.8, edge_density * 2),
                        'method': 'edge_detection',
                        'color': 'unknown'
                    })
    except Exception as e:
        print(f"Error in edge-based detection: {e}")
    return plates


def detect_license_plates_yolo(image, model_path=YOLO_PATH):
    """YOLO detections with confidence > 0.5.

    NOTE: the default ``yolov8n.pt`` is the stock COCO model, so it returns any object
    (cars, people, ...), not just plates. Point ``model_path`` at a plate-trained model.
    """
    global _yolo_model
    if not YOLO_AVAILABLE:
        return []
    try:
        if _yolo_model is None:
            if not os.path.exists(model_path):
                print(f"YOLO model not found at {model_path}")
                return []
            _yolo_model = YOLO(model_path)

        plates = []
        for result in _yolo_model(image, verbose=False):
            if getattr(result, 'boxes', None) is None:
                continue
            for box in result.boxes:
                if box.conf > 0.5:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    plates.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(box.conf[0]),
                        'method': 'yolo',
                        'color': 'unknown'
                    })
        return plates
    except Exception as e:
        print(f"YOLO detection failed: {e}")
        return []


def bbox_overlap(bbox1, bbox2):
    """Intersection area divided by the smaller box's area (0..1)."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    x1_i, y1_i = max(x1_1, x1_2), max(y1_1, y1_2)
    x2_i, y2_i = min(x2_1, x2_2), min(y2_1, y2_2)
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    return intersection / min(area1, area2)


def remove_duplicate_detections(plates, max_plates=MAX_PLATES, overlap_threshold=0.3):
    """Greedy de-duplication: keep the most confident box of each overlapping group."""
    unique = []
    for plate in sorted(plates, key=lambda p: p['confidence'], reverse=True):
        if all(bbox_overlap(plate['bbox'], kept['bbox']) <= overlap_threshold for kept in unique):
            unique.append(plate)
    return unique[:max_plates]


def detect_license_plates_advanced(image):
    """Run every detector, merge the results, and retry once on a contrast-boosted image."""
    print(f"Starting license plate detection on image shape: {image.shape}")

    detectors = [
        ('Yellow', detect_yellow_license_plates),
        ('White', detect_white_license_plates),
        ('Enhanced white', detect_white_license_plates_enhanced),
        ('Blue', detect_blue_license_plates),
        ('OpenCV', detect_license_plates_opencv),
        ('Edge-based', detect_plates_by_edges),
    ]
    if YOLO_AVAILABLE:
        detectors.append(('YOLO', detect_license_plates_yolo))

    plates = []
    for name, detector in detectors:
        found = detector(image)
        print(f"{name} plates found: {len(found)}")
        plates.extend(found)

    print(f"Total plates before deduplication: {len(plates)}")
    final_plates = remove_duplicate_detections(plates)
    print(f"Final plates after deduplication: {len(final_plates)}")

    if not final_plates:
        try:
            print("No plates found; running last-pass enhancement for detection...")
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
            enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            retry = remove_duplicate_detections(
                detect_white_license_plates(enhanced_img) +
                detect_white_license_plates_enhanced(enhanced_img) +
                detect_yellow_license_plates(enhanced_img)
            )
            print(f"Retry detection found: {len(retry)}")
            return retry
        except Exception:
            pass

    return final_plates
