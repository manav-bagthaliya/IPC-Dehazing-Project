"""Detect -> enhance -> OCR -> validate, on an already dehazed BGR image."""

import json

import cv2

from ipc_dehaze.enhance import enhance_plate_region
from ipc_dehaze.plates.detection import detect_license_plates_advanced
from ipc_dehaze.plates.ocr import recognize_plate_text, validate_indian_license_plate


def paste_region(image, bbox, region):
    """Resize ``region`` to ``bbox`` and write it into ``image`` in place. Returns success."""
    x1, y1, x2, y2 = bbox
    if not (0 <= y1 < y2 <= image.shape[0] and 0 <= x1 < x2 <= image.shape[1]):
        return False
    if region.shape[:2] != (y2 - y1, x2 - x1):
        region = cv2.resize(region, (x2 - x1, y2 - y1))
    image[y1:y2, x1:x2] = region
    return True


def process_plates(image, enhancer=enhance_plate_region, only_valid=True):
    """Find plates in ``image``, enhance and OCR each one.

    Args:
        image: dehazed BGR uint8 image (not modified).
        enhancer: ``f(image, bbox) -> crop`` used to enhance each plate.
        only_valid: paste back only plates whose text is a valid Indian plate.

    Returns:
        (output_image, plates) where each plate dict gains ``text``, ``valid``,
        ``formatted_text``, ``segments`` and, if pasted, ``pasted=True``.
    """
    output = image.copy()
    plates = detect_license_plates_advanced(image)
    print(f"   Detected {len(plates)} potential license plates")

    for i, plate in enumerate(plates):
        print(f"\n   Plate {i + 1}: {plate.get('color', 'unknown')} plate "
              f"({plate.get('method', 'unknown')}) - Confidence: {plate['confidence']:.2f}")

        enhanced = enhancer(image, plate['bbox'])
        text = recognize_plate_text(enhanced)
        validation = validate_indian_license_plate(text)
        plate.update({
            'text': text,
            'valid': validation['is_valid'],
            'formatted_text': validation['formatted_text'],
            'segments': validation['segments'],
            'pasted': False,
        })
        print(f"   Raw OCR text: '{text}'")

        if validation['is_valid']:
            plate['confidence'] = validation['confidence']
            print(f"   ✅ Valid Indian format: {validation['formatted_text']}")
        else:
            print(f"   ❌ Invalid format: {validation['error']}")

        if validation['is_valid'] or not only_valid:
            plate['pasted'] = paste_region(output, plate['bbox'], enhanced)

    return output, plates


def emit_plate_json(plate):
    """Print one machine-readable line; the web app parses lines starting with PLATE_JSON:."""
    summary = {
        'formatted_text': plate['formatted_text'],
        'confidence': float(plate['confidence']),
        'segments': plate.get('segments', {}),
        'bbox': [int(v) for v in plate['bbox']],
        'method': plate.get('method', 'unknown'),
        'color': plate.get('color', 'unknown')
    }
    print('PLATE_JSON:' + json.dumps(summary, ensure_ascii=False))
