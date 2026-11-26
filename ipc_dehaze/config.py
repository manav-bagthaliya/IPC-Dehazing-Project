"""Project-wide paths and defaults."""

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRETRAINED_DIR = os.path.join(ROOT_DIR, 'pretrained_models')
PREDICTOR_PATH = os.path.join(PRETRAINED_DIR, 'predictor.pth')
CRITIC_PATH = os.path.join(PRETRAINED_DIR, 'critic.pth')
# Licence-plate-trained YOLO11 (morsetechlab/yolov11-license-plate-detection, AGPL-3.0).
# 'm' is the accuracy default; 'license_plate_yolo11s.pt' is a faster, smaller alternative.
YOLO_PATH = os.path.join(PRETRAINED_DIR, 'license_plate_yolo11m.pt')
YOLO_CONF = 0.25  # minimum plate-detector confidence (Ultralytics default)

# Dehazing defaults (shared by the CLI and the web app)
DEFAULT_NUM_ITERATIONS = 8
DEFAULT_MAX_SIZE = 1500
PAD_MULTIPLE = 32

# Detection returns at most this many plates per image
MAX_PLATES = 5
