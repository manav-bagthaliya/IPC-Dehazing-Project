"""Plate text recognition (EasyOCR) and Indian plate format validation."""

import cv2
import numpy as np

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: EasyOCR not available. Install with: pip install easyocr")

# Lazy-initialised: loading the OCR model is slow
_easyocr_reader = None

# Common OCR confusions, used to coerce text into the LL DD LL DDDD pattern
TO_DIGIT = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'}
TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
LETTER_POSITIONS = (0, 1, 4, 5)
DIGIT_POSITIONS = (2, 3, 6, 7, 8, 9)


def validate_indian_license_plate(text: str) -> dict:
    """Validate the Indian plate format XX NN XX NNNN (e.g. ``MH 01 AB 1234``).

    Returns a dict with ``is_valid``, ``formatted_text``, ``confidence`` (fraction of
    valid segments), ``error`` and the parsed ``segments``.
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

    cleaned = text.replace(' ', '').replace('-', '').replace('.', '').upper()
    if len(cleaned) != 10:
        result['error'] = f'Invalid length: {len(cleaned)} (expected 10)'
        return result

    segments = [
        ('state_code', cleaned[0:2], str.isalpha, 'State code'),
        ('district_code', cleaned[2:4], str.isdigit, 'District code'),
        ('series', cleaned[4:6], str.isalpha, 'Series'),
        ('unique_number', cleaned[6:10], str.isdigit, 'Unique number'),
    ]

    errors = []
    for key, value, check, label in segments:
        if check(value):
            result['segments'][key] = value
        else:
            errors.append(f'{label} invalid: {value}')

    if errors:
        result['error'] = '; '.join(errors)
        result['confidence'] = len(result['segments']) / 4.0  # partial credit
    else:
        result['is_valid'] = True
        result['formatted_text'] = ' '.join(value for _, value, _, _ in segments)
        result['confidence'] = 1.0

    return result


def coerce_to_pattern(s: str) -> str:
    """Force a string into the LL DD LL DDDD shape using the confusion maps."""
    s = ''.join(ch for ch in s if ch.isalnum()).upper()
    out = list(s.ljust(10, '0')[:10])
    for pos in LETTER_POSITIONS:
        if not out[pos].isalpha():
            out[pos] = TO_LETTER.get(out[pos], 'A')
    for pos in DIGIT_POSITIONS:
        if not out[pos].isdigit():
            out[pos] = TO_DIGIT.get(out[pos], '0')
    return ''.join(out)


def preprocess_for_ocr(plate_bgr: np.ndarray) -> np.ndarray:
    """Crisp black-on-white version of a plate crop, used only as OCR input."""
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)

    # Upscale small crops (target width ~480px)
    h, w = gray.shape[:2]
    target_w = 480
    if w < target_w:
        scale = target_w / max(1, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    blurred = cv2.GaussianBlur(equalized, (0, 0), 1.2)
    sharp = cv2.addWeighted(equalized, 1.6, blurred, -0.6, 0)

    bw = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    if np.mean(bw) < 127:
        bw = cv2.bitwise_not(bw)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    return bw


def best_plate_candidate(raw_text: str) -> str:
    """Pick the best Indian-format reading of raw OCR text, or the cleaned raw text."""
    direct = validate_indian_license_plate(raw_text)
    if direct['is_valid']:
        return direct['formatted_text']

    base = ''.join(ch for ch in raw_text if ch.isalnum())[:12]
    if not base:
        return ''
    candidates = [
        base,
        coerce_to_pattern(base),
        coerce_to_pattern(''.join(TO_DIGIT.get(ch, ch) for ch in base)),
        coerce_to_pattern(''.join(TO_LETTER.get(ch, ch) for ch in base)),
    ]
    for cand in candidates:
        res = validate_indian_license_plate(cand)
        if res['is_valid']:
            return res['formatted_text']

    return ''.join(ch for ch in raw_text if ch.isalnum()).strip()[:10]


def recognize_plate_text(plate_bgr: np.ndarray) -> str:
    """OCR a plate crop. Returns formatted Indian plate text, best-effort raw text, or ''."""
    global _easyocr_reader
    if not EASYOCR_AVAILABLE or plate_bgr is None or plate_bgr.size == 0:
        return ''
    try:
        if _easyocr_reader is None:
            _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)

        ocr_results = _easyocr_reader.readtext(preprocess_for_ocr(plate_bgr), detail=1, paragraph=False)
        if not ocr_results:
            return ''

        # Concatenate alphanumerics, most confident tokens first
        ocr_results.sort(key=lambda x: float(x[2]) if len(x) > 2 else 0.0, reverse=True)
        raw_text = ''.join(''.join(ch for ch in r[1] if ch.isalnum()) for r in ocr_results).upper()

        return best_plate_candidate(raw_text)
    except Exception:
        return ''
