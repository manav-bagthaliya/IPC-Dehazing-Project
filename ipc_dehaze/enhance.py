"""Classical (OpenCV) image enhancement applied after dehazing."""

import cv2
import numpy as np


def enhance_output_quality(image):
    """Global post-processing of a dehazed BGR image: CLAHE, gamma, sharpening, denoising."""
    try:
        # CLAHE + gamma on the L channel only, so colours are preserved
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lightness = clahe.apply(lightness)
        lightness = np.power(lightness / 255.0, 0.85) * 255.0
        lightness = np.clip(lightness, 0, 255).astype(np.uint8)
        enhanced = cv2.cvtColor(cv2.merge([lightness, a, b]), cv2.COLOR_LAB2BGR)

        # Unsharp mask, edge-preserving denoise, final contrast lift
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
        enhanced = cv2.addWeighted(enhanced, 1.3, gaussian, -0.3, 0)
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.15, beta=15)

        return np.clip(enhanced, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"Error enhancing output quality: {e}")
        return image


def _crop(image, bbox):
    """Crop ``bbox`` (x1, y1, x2, y2) clamped to the image; returns None if empty."""
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    region = image[y1:y2, x1:x2]
    return region if region.size else None


def _resize_to_width(region, width=400):
    if region.shape[1] != width:
        region = cv2.resize(region, (width, int(region.shape[0] * width / region.shape[1])))
    return region


def enhance_plate_region(image, plate_bbox):
    """Colour-preserving contrast/sharpness boost of a plate crop. Returns the enhanced crop."""
    region = _crop(image, plate_bbox)
    if region is None:
        return image
    region = _resize_to_width(region)

    enhanced = cv2.convertScaleAbs(region, alpha=1.2, beta=10)

    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel * 0.3) + enhanced * 0.7
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)

    return np.clip(enhanced, 0, 255).astype(np.uint8)


def binarize_plate(region):
    """High-contrast black/white rendering of a plate crop, for reading characters."""
    try:
        region = _resize_to_width(region)

        value = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)[:, :, 2]
        thresh = cv2.adaptiveThreshold(value, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 4)
        thresh = cv2.bitwise_not(thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.GaussianBlur(thresh, (3, 3), 0)

        gaussian = cv2.GaussianBlur(thresh, (0, 0), 1.2)
        thresh = cv2.addWeighted(thresh, 1.5, gaussian, -0.5, 0)

        enhanced = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.15, beta=12)
        enhanced = cv2.bilateralFilter(enhanced, 5, 30, 30)

        return np.clip(enhanced, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"Error binarizing plate region: {e}")
        return region


def binarize_plate_region(image, plate_bbox):
    """``binarize_plate`` applied to ``plate_bbox`` of ``image``. Returns the enhanced crop."""
    region = _crop(image, plate_bbox)
    if region is None:
        return image
    return binarize_plate(region)
