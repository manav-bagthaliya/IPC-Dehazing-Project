"""Plate text recognition (EasyOCR) and Indian plate format validation."""

import re

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

ALPHANUMERIC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Common OCR confusions, used to repair characters that are in the wrong class for their position
TO_DIGIT = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'}
TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}
# Registration codes of Indian states and union territories (incl. legacy OR/UA and DD/DN)
INDIAN_STATE_CODES = frozenset(
    'AN AP AR AS BR CG CH DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ NL OD OR PB PY RJ SK TN TR TS UA UK '
    'UP WB'.split())
LETTER_POSITIONS = (0, 1, 4, 5)
DIGIT_POSITIONS = (2, 3, 6, 7, 8, 9)

# Indian registration: state (2 letters), district (1-2 digits), series (1-3 letters), number (4 digits)
PLATE_PATTERN = re.compile(r'^([A-Z]{2})(0[1-9]|[1-9]\d?)([A-Z]{1,3})(\d{4})$')  # district 01-99, never 0/00
# (district digits, series letters) -> penalty. Layouts are ordered by how common they are, and the
# penalty makes the decoder prefer a common layout needing one repair (UP5OAS4535 -> UP 50 AS 4535)
# over a rare layout that happens to fit unchanged (UP 5 OAS 4535).
LAYOUT_PENALTY = {(2, 2): 0.0, (2, 1): 0.5, (2, 3): 0.5, (1, 2): 1.5, (1, 1): 1.5, (1, 3): 1.5}
LAYOUTS = tuple(LAYOUT_PENALTY)
MAX_CORRECTIONS = 2  # never repair more characters than this: more would be guessing
MAX_EDIT_COST = 3    # corrected characters + ignored (noise) characters; beyond this it is a guess


def _clean(text: str) -> str:
    return ''.join(ch for ch in str(text) if ch.isalnum()).upper()


def validate_indian_license_plate(text: str) -> dict:
    """Validate an Indian plate such as ``MH 01 AB 1234`` or ``DL 1C 1234``-style layouts.

    Accepts a real state/UT code, a 1-2 digit district, 1-3 series letters and a 4 digit number.

    Returns a dict with ``is_valid``, ``formatted_text``, ``confidence`` (1.0 if valid, else the
    fraction of valid segments for 10-character text), ``error`` and the parsed ``segments``.
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

    match = PLATE_PATTERN.match(cleaned)
    if match and match.group(1) in INDIAN_STATE_CODES:
        state, district, series, number = match.groups()
        result['segments'] = {'state_code': state, 'district_code': district,
                              'series': series, 'unique_number': number}
        result['formatted_text'] = f'{state} {district} {series} {number}'
        result['is_valid'] = True
        result['confidence'] = 1.0
        return result

    if not 8 <= len(cleaned) <= 11:
        result['error'] = f'Invalid length: {len(cleaned)} (expected 8-11)'
        return result

    if len(cleaned) == 10:  # most common layout: say which part is wrong, with partial credit
        segments = [
            ('state_code', cleaned[0:2], lambda v: v in INDIAN_STATE_CODES, 'State code'),
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
        result['error'] = '; '.join(errors) or 'Invalid format'
        result['confidence'] = len(result['segments']) / 4.0
    else:
        result['error'] = 'Invalid format'
    return result


def coerce_to_pattern(s: str) -> str:
    """Fix confusable characters of a 10-character string so it fits LL DD LL DDDD.

    Text of any other length returns '' - padding or truncating would invent plate
    characters that were never read.
    """
    s = _clean(s)
    if len(s) != 10:
        return ''
    out = list(s)
    for pos in LETTER_POSITIONS:
        if not out[pos].isalpha():
            out[pos] = TO_LETTER.get(out[pos], 'A')
    for pos in DIGIT_POSITIONS:
        if not out[pos].isdigit():
            out[pos] = TO_DIGIT.get(out[pos], '0')
    return ''.join(out)


def _fit_layout(window: str, district_digits: int, series_letters: int):
    """Repair ``window`` to the layout; returns (fixed_text, n_corrections) or (None, 0) if impossible."""
    letter_slots = set(range(2)) | set(range(2 + district_digits, 2 + district_digits + series_letters))
    fixed, changes = [], 0
    for pos, ch in enumerate(window):
        want_letter = pos in letter_slots
        if want_letter == ch.isalpha():
            fixed.append(ch)
            continue
        repaired = (TO_LETTER if want_letter else TO_DIGIT).get(ch)
        if repaired is None:
            return None, 0
        fixed.append(repaired)
        changes += 1
    return ''.join(fixed), changes


def extract_plate_text(raw_text: str) -> str:
    """Find an Indian plate inside noisy OCR text, or return ''.

    Searches every window of the text for a valid layout, ignoring the 'IND' mark printed on
    HSRP plates, and repairs at most ``MAX_CORRECTIONS`` confusable characters (O/0, I/1, ...).
    Corrected plus ignored characters may not exceed ``MAX_EDIT_COST``; a parse that explains
    more of the text is preferred over one that drops characters that were read.
    Never pads or truncates: every returned character was read from the image.
    """
    cleaned = _clean(raw_text)
    variants = [cleaned] + ([cleaned.replace('IND', '')] if 'IND' in cleaned else [])

    best = None  # (edit cost, layout order, start, text)
    for text in variants:
        for order, (district, series) in enumerate(LAYOUTS):
            size = 2 + district + series + 4
            for start in range(len(text) - size + 1):
                fixed, changes = _fit_layout(text[start:start + size], district, series)
                cost = changes + len(text) - size
                if (fixed is None or changes > MAX_CORRECTIONS or cost > MAX_EDIT_COST
                        or fixed[:2] not in INDIAN_STATE_CODES):
                    continue
                candidate = (cost + LAYOUT_PENALTY[(district, series)], order, start, fixed)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
    if best is None:
        return ''
    result = validate_indian_license_plate(best[3])
    return result['formatted_text'] if result['is_valid'] else ''


def best_plate_candidate(raw_text: str) -> str:
    """Best Indian-format reading of raw OCR text; otherwise the cleaned raw text (at most 10 chars)."""
    direct = validate_indian_license_plate(raw_text)
    if direct['is_valid']:
        return direct['formatted_text']

    found = extract_plate_text(raw_text)
    if found:
        return found

    return _clean(raw_text)[:10]


def reading_order_text(results) -> str:
    """Join EasyOCR boxes in reading order: lines top to bottom, each line left to right.

    Handles two-line plates and boxes that EasyOCR returns in arbitrary order.
    """
    if not results:
        return ''
    items = []
    for box, text, *_ in results:
        ys = [p[1] for p in box]
        items.append((float(np.mean(ys)), float(np.mean([p[0] for p in box])), max(ys) - min(ys), _clean(text)))
    median_height = float(np.median([i[2] for i in items])) or 1.0

    items.sort(key=lambda i: i[0])
    lines, current = [], [items[0]]
    for item in items[1:]:
        if abs(item[0] - np.mean([c[0] for c in current])) > 0.6 * median_height:
            lines.append(current)
            current = [item]
        else:
            current.append(item)
    lines.append(current)
    return ''.join(i[3] for line in lines for i in sorted(line, key=lambda i: i[1]))


def prepare_for_ocr(plate_bgr: np.ndarray, min_width: int = 320) -> np.ndarray:
    """Grayscale + local contrast boost, upscaled to ``min_width``.

    Deliberately NOT binarised: measured on 100 Indian plate crops, hard thresholding before
    EasyOCR cut exact-match accuracy from ~50% to ~11%.
    """
    h, w = plate_bgr.shape[:2]
    if w < min_width:
        plate_bgr = cv2.resize(plate_bgr, (min_width, max(1, int(h * min_width / w))), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)


def preprocess_for_ocr(plate_bgr: np.ndarray) -> np.ndarray:
    """Crisp black-on-white version of a plate crop (kept for visualisation; not used for OCR)."""
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


def _get_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import torch
        _easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
    return _easyocr_reader


def read_raw_text(image) -> str:
    """EasyOCR text of ``image`` (grayscale or BGR) in reading order, letters and digits only."""
    results = _get_reader().readtext(image, detail=1, paragraph=False, allowlist=ALPHANUMERIC)
    return reading_order_text(results)


def recognize_plate_text(plate_bgr: np.ndarray) -> str:
    """OCR a raw plate crop. Returns formatted Indian plate text, best-effort raw text, or ''.

    Pass the crop as it appears in the (dehazed) image, not a binarised version.
    """
    if not EASYOCR_AVAILABLE or plate_bgr is None or plate_bgr.size == 0:
        return ''
    try:
        text = best_plate_candidate(read_raw_text(prepare_for_ocr(plate_bgr)))
        if validate_indian_license_plate(text)['is_valid']:
            return text
        # Second chance: the plain colour crop sometimes reads what the contrast-boosted one misses
        h, w = plate_bgr.shape[:2]
        plain = plate_bgr if w >= 320 else cv2.resize(plate_bgr, (320, max(1, int(h * 320 / w))),
                                                      interpolation=cv2.INTER_CUBIC)
        second = best_plate_candidate(read_raw_text(plain))
        return second if validate_indian_license_plate(second)['is_valid'] else text
    except Exception:
        return ''
