# Pretrained models

Weight files are not stored in git (see `.gitignore`). Put them here:

| File | What it is | Where to get it |
|---|---|---|
| `predictor.pth` | IPC-Dehaze Predictor (DehazeTokenNet) | `python scripts/download_weights.py` |
| `critic.pth` | IPC-Dehaze Critic | `python scripts/download_weights.py` |
| `yolov8n.pt` | Stock COCO YOLOv8-nano (not plate-specific) | Ultralytics; replace with a plate-trained model |
| `vqgan.pth`, `vqgan_d.pth` | Stage 1 VQGAN + discriminator | Only needed for training (Stage 2/3) |
