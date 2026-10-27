"""Check that this machine can run IPC-Dehaze: Python, packages, device and model files.

Usage:
    python scripts/check_env.py
"""

import importlib
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from ipc_dehaze.config import CRITIC_PATH, PREDICTOR_PATH, YOLO_PATH  # noqa: E402

REQUIRED = ['torch', 'torchvision', 'cv2', 'numpy', 'timm', 'einops', 'pyiqa', 'flask', 'tqdm', 'yaml']
OPTIONAL = {'ultralytics': 'YOLO plate detection', 'easyocr': 'plate OCR', 'gdown': 'weight download'}


def main():
    ok = True

    print(f"Python {sys.version.split()[0]}")
    if sys.version_info < (3, 8):
        print("  ✗ Python 3.8+ required")
        ok = False

    print("\nRequired packages:")
    for name in REQUIRED:
        try:
            importlib.import_module(name)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name}  (pip install -r requirements.txt)")
            ok = False

    print("\nOptional packages:")
    for name, purpose in OPTIONAL.items():
        try:
            importlib.import_module(name)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ⚠ {name} missing - {purpose} disabled")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"\nDevice: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            print(f"\nDevice: CPU only ({torch.get_num_threads()} threads) - dehazing will be slow")
    except ImportError:
        pass

    print("\nModel files:")
    for path in (PREDICTOR_PATH, CRITIC_PATH, YOLO_PATH):
        rel = os.path.relpath(path, ROOT_DIR)
        if os.path.exists(path):
            print(f"  ✓ {rel}")
        else:
            print(f"  ✗ {rel}  (python scripts/download_weights.py)")
            ok = ok and path == YOLO_PATH

    print("\n" + ("Ready." if ok else "Not ready - fix the ✗ items above."))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
