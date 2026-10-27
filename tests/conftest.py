import os
import sys

import cv2
import numpy as np
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, 'tests', 'data')


@pytest.fixture
def yellow_plate_image():
    """Grey scene with one yellow plate at (200, 200)-(500, 250)."""
    img = np.full((600, 800, 3), 150, dtype=np.uint8)
    cv2.rectangle(img, (200, 200), (500, 250), (0, 255, 255), -1)
    cv2.putText(img, "MH01AB1234", (215, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return img


@pytest.fixture
def data_image():
    def load(name):
        img = cv2.imread(os.path.join(DATA_DIR, name))
        assert img is not None, name
        return img
    return load
