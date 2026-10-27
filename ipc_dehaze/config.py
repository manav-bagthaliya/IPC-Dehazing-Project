"""Project-wide paths and defaults."""

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRETRAINED_DIR = os.path.join(ROOT_DIR, 'pretrained_models')
PREDICTOR_PATH = os.path.join(PRETRAINED_DIR, 'predictor.pth')
CRITIC_PATH = os.path.join(PRETRAINED_DIR, 'critic.pth')
# Stock COCO YOLOv8n. Replace with a licence-plate-trained model for real plate detection.
YOLO_PATH = os.path.join(PRETRAINED_DIR, 'yolov8n.pt')

# Dehazing defaults (shared by the CLI and the web app)
DEFAULT_NUM_ITERATIONS = 8
DEFAULT_MAX_SIZE = 1500
PAD_MULTIPLE = 32

# Detection returns at most this many plates per image
MAX_PLATES = 5
