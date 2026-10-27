"""Licence plate detection, OCR and validation."""

from ipc_dehaze.plates.detection import bbox_overlap, detect_license_plates_advanced, remove_duplicate_detections
from ipc_dehaze.plates.ocr import recognize_plate_text, validate_indian_license_plate
from ipc_dehaze.plates.pipeline import emit_plate_json, process_plates

__all__ = [
    'bbox_overlap',
    'detect_license_plates_advanced',
    'emit_plate_json',
    'process_plates',
    'recognize_plate_text',
    'remove_duplicate_detections',
    'validate_indian_license_plate',
]
