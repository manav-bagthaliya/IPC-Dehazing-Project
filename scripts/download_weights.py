"""Download the official IPC-Dehaze weights (predictor.pth, critic.pth) into pretrained_models/.

Weights are published by the paper authors:
    https://github.com/Jiayi-Fu/IPC-Dehaze
    Google Drive: https://drive.google.com/drive/folders/1i5DS0jjTlm1nSXSAPd4Qp5_7dLSva_1E
    Baidu Disk:   https://pan.baidu.com/s/1W4Mt_uZprSjxj1oH8MU59w  (pwd: jfsp)

Usage:
    python scripts/download_weights.py
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from ipc_dehaze.config import CRITIC_PATH, PREDICTOR_PATH, PRETRAINED_DIR  # noqa: E402

GDRIVE_FOLDER = 'https://drive.google.com/drive/folders/1i5DS0jjTlm1nSXSAPd4Qp5_7dLSva_1E'


def main():
    missing = [p for p in (PREDICTOR_PATH, CRITIC_PATH) if not os.path.exists(p)]
    if not missing:
        print('All weights already present.')
        return

    try:
        import gdown
    except ImportError:
        print('gdown is not installed (pip install gdown). Download manually from:')
        print(f'  {GDRIVE_FOLDER}')
        print(f'and place predictor.pth and critic.pth in {PRETRAINED_DIR}')
        sys.exit(1)

    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    gdown.download_folder(GDRIVE_FOLDER, output=PRETRAINED_DIR, quiet=False)

    still_missing = [p for p in (PREDICTOR_PATH, CRITIC_PATH) if not os.path.exists(p)]
    if still_missing:
        print('Download finished but these files are still missing (check the folder contents / names):')
        for p in still_missing:
            print(f'  {p}')
        sys.exit(1)
    print('Done.')


if __name__ == '__main__':
    main()
