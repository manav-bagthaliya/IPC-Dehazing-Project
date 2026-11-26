# Pretrained models

Weight files are not stored in git (see `.gitignore`). Put them here:

| File | What it is | Where to get it |
|---|---|---|
| `predictor.pth` | IPC-Dehaze Predictor (DehazeTokenNet) | `python scripts/download_weights.py` |
| `critic.pth` | IPC-Dehaze Critic | `python scripts/download_weights.py` |
| `license_plate_yolo11m.pt` | Licence-plate detector (YOLO11-medium) - used by default | [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) (`license-plate-finetune-v1m.pt`, AGPL-3.0) |
| `license_plate_yolo11s.pt` | Smaller/faster plate detector (YOLO11-small), alternative | same repo (`license-plate-finetune-v1s.pt`); point `YOLO_PATH` in `ipc_dehaze/config.py` at it |
| `vqgan.pth`, `vqgan_d.pth` | Stage 1 VQGAN + discriminator | Only needed for training (Stage 2/3) |

`.pt` files are Python pickles and can run code when loaded. Only use weights from sources you trust,
and inspect the pickle's imports before loading anything new.
