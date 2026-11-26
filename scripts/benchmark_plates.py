"""Accuracy benchmark for plate detection, OCR and dehazing.

Ground truth (pick one):
    --coco  annotations.json    COCO json (bbox = [x, y, w, h]); images are looked up in --images
    --labels labels_dir         YOLO txt files (class cx cy w h, normalised), one per image
Optional OCR ground truth:
    --texts plates.csv          CSV with header ``filename,text`` (one plate per image)

Conditions (``--conditions``, comma separated):
    clear     the image as given
    hazy      synthetic haze added to the image (atmospheric scattering model)
    dehazed   hazy image restored with IPC-Dehaze        (slow on CPU; use a GPU)

Reports, per condition:
    detection  precision / recall / F1 / AP50 / mean IoU of matched boxes
    OCR        exact-match rate and character accuracy (crop from the *ground-truth* box = OCR alone,
               crop from the *detected* box = full pipeline)         [needs --texts]
    restoration PSNR / SSIM against the clear image                   [hazy / dehazed only]

Examples:
    python scripts/benchmark_plates.py --images data/test --coco data/test/_annotations.coco.json --limit 100
    python scripts/benchmark_plates.py --images data/test --coco data/test/_annotations.coco.json \\
        --conditions clear,hazy,dehazed --limit 20 --out benchmark_report.json
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
import time

import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


# ----------------------------------------------------------------------------- ground truth
def load_coco(json_path, images_dir):
    with open(json_path, encoding='utf-8') as f:
        coco = json.load(f)
    by_id = {im['id']: im['file_name'] for im in coco['images']}
    boxes = {name: [] for name in by_id.values()}
    for ann in coco['annotations']:
        x, y, w, h = ann['bbox']
        boxes[by_id[ann['image_id']]].append([x, y, x + w, y + h])
    return {os.path.join(images_dir, n): b for n, b in boxes.items()}


def load_yolo_labels(labels_dir, images_dir):
    gt = {}
    for path in sorted(glob.glob(os.path.join(images_dir, '*'))):
        if not path.lower().endswith(IMAGE_EXTS):
            continue
        label = os.path.join(labels_dir, os.path.splitext(os.path.basename(path))[0] + '.txt')
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = []
        if os.path.exists(label):
            for line in open(label, encoding='utf-8'):
                parts = line.split()
                if len(parts) >= 5:
                    cx, cy, bw, bh = (float(v) for v in parts[1:5])
                    boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        gt[path] = boxes
    return gt


def load_texts(csv_path, images_dir):
    texts = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            texts[os.path.join(images_dir, row['filename'])] = normalise_text(row['text'])
    return texts


# ----------------------------------------------------------------------------- metrics
def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(preds, gts, thr):
    """Greedy match by confidence. Returns [(pred_idx, gt_idx, iou)] for matches with IoU >= thr."""
    order = sorted(range(len(preds)), key=lambda i: preds[i]['confidence'], reverse=True)
    used, matches = set(), []
    for pi in order:
        best, best_iou = None, thr
        for gi, g in enumerate(gts):
            if gi in used:
                continue
            v = iou(preds[pi]['bbox'], g)
            if v >= best_iou:
                best, best_iou = gi, v
        if best is not None:
            used.add(best)
            matches.append((pi, best, best_iou))
    return matches


def average_precision(records, n_gt):
    """All-point interpolated AP from (confidence, is_tp) records."""
    if n_gt == 0 or not records:
        return 0.0
    records = sorted(records, key=lambda r: r[0], reverse=True)
    tp = np.cumsum([r[1] for r in records])
    fp = np.cumsum([not r[1] for r in records])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def normalise_text(text):
    return ''.join(ch for ch in str(text) if ch.isalnum()).upper()


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_accuracy(pred, truth):
    if not truth:
        return 0.0
    return max(0.0, 1.0 - edit_distance(pred, truth) / max(len(truth), 1))


# ----------------------------------------------------------------------------- haze
def synth_haze(image, beta=1.0, atmosphere=0.85, seed=0):
    """Atmospheric scattering model I = J*t + A*(1-t), t = exp(-beta*depth).

    No depth map is available, so depth is a deterministic proxy: farther towards the top of the
    image plus smooth random variation. Seeded per image so conditions are reproducible.
    """
    rng = np.random.RandomState(seed)
    h, w = image.shape[:2]
    ramp = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
    noise = cv2.GaussianBlur(rng.rand(h, w).astype(np.float32), (0, 0), max(h, w) / 8)
    noise = (noise - noise.min()) / max(float(np.ptp(noise)), 1e-6)
    depth = 0.35 + 0.45 * ramp + 0.35 * noise
    t = np.exp(-beta * depth)[:, :, None]
    hazy = image.astype(np.float32) / 255.0 * t + atmosphere * (1.0 - t)
    return np.clip(hazy * 255.0, 0, 255).astype(np.uint8)


def psnr_ssim(ref, test):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    if ref.shape != test.shape:
        test = cv2.resize(test, (ref.shape[1], ref.shape[0]))
    return (peak_signal_noise_ratio(ref, test, data_range=255),
            structural_similarity(ref, test, channel_axis=2, data_range=255))


# ----------------------------------------------------------------------------- evaluation
class Accumulator:
    def __init__(self):
        self.tp = self.fp = self.fn = 0
        self.ious, self.records, self.n_gt = [], [], 0
        self.ocr = {'gt_crop': [], 'pred_crop': []}  # (exact, char_acc, valid_format)
        self.psnr, self.ssim = [], []

    def summary(self):
        p = self.tp / max(self.tp + self.fp, 1)
        r = self.tp / max(self.n_gt, 1)
        out = {
            'images_gt_plates': self.n_gt, 'tp': self.tp, 'fp': self.fp, 'fn': self.fn,
            'precision': round(p, 4), 'recall': round(r, 4),
            'f1': round(2 * p * r / max(p + r, 1e-9), 4),
            'ap50': round(average_precision(self.records, self.n_gt), 4),
            'mean_iou_matched': round(float(np.mean(self.ious)), 4) if self.ious else 0.0,
        }
        for key, rows in self.ocr.items():
            if rows:
                out[f'ocr_{key}'] = {
                    'n': len(rows),
                    'exact_match': round(float(np.mean([r[0] for r in rows])), 4),
                    'char_accuracy': round(float(np.mean([r[1] for r in rows])), 4),
                    'valid_indian_format': round(float(np.mean([r[2] for r in rows])), 4),
                }
        if self.psnr:
            out['psnr_db'] = round(float(np.mean(self.psnr)), 3)
            out['ssim'] = round(float(np.mean(self.ssim)), 4)
        return out


def crop(image, bbox, pad=0.05):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1, x2, y2 = int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(w, x2 + px)), int(min(h, y2 + py))
    return image[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None


def read_plate(image, bbox):
    """OCR as the pipeline does it: enhance crop, EasyOCR, Indian-format correction."""
    from ipc_dehaze.plates.ocr import recognize_plate_text, validate_indian_license_plate
    region = crop(image, bbox)
    if region is None:
        return '', False
    text = recognize_plate_text(region)  # same plain-crop OCR input as the pipeline
    return normalise_text(text), validate_indian_license_plate(text)['is_valid']


def evaluate_image(acc, image, gt_boxes, text, args, detect):
    preds = detect(image)
    matches = match_boxes(preds, gt_boxes, args.iou)
    matched_pred = {m[0] for m in matches}
    acc.n_gt += len(gt_boxes)
    acc.tp += len(matches)
    acc.fp += len(preds) - len(matches)
    acc.fn += len(gt_boxes) - len(matches)
    acc.ious += [m[2] for m in matches]
    acc.records += [(p['confidence'], i in matched_pred) for i, p in enumerate(preds)]

    if text and gt_boxes:
        truth = text
        main_gt = max(range(len(gt_boxes)), key=lambda i: (gt_boxes[i][2] - gt_boxes[i][0]) *
                                                           (gt_boxes[i][3] - gt_boxes[i][1]))
        read, valid = read_plate(image, gt_boxes[main_gt])
        acc.ocr['gt_crop'].append((read == truth, char_accuracy(read, truth), valid))
        pred_for_gt = next((preds[pi] for pi, gi, _ in matches if gi == main_gt), None)
        read, valid = read_plate(image, pred_for_gt['bbox']) if pred_for_gt else ('', False)
        acc.ocr['pred_crop'].append((read == truth, char_accuracy(read, truth), valid))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', required=True, help='folder of images')
    ap.add_argument('--coco', help='COCO annotations json')
    ap.add_argument('--labels', help='folder of YOLO txt labels')
    ap.add_argument('--texts', help='CSV (filename,text) for OCR ground truth')
    ap.add_argument('--limit', type=int, default=100, help='max images, randomly sampled (0 = all)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--conditions', default='clear', help='comma list of: clear,hazy,dehazed')
    ap.add_argument('--beta', type=float, default=1.0, help='synthetic haze density')
    ap.add_argument('--atmosphere', type=float, default=0.85, help='synthetic airlight A (0-1)')
    ap.add_argument('--iou', type=float, default=0.5, help='IoU threshold for a correct detection')
    ap.add_argument('--detector', choices=['yolo', 'full', 'heuristics'], default='full',
                    help="yolo = plate model alone; full = plate model + heuristic fallback (pipeline "
                         "default); heuristics = old colour/edge detectors only (baseline)")
    ap.add_argument('-n', '--iterations', type=int, default=8, help='IPC-Dehaze decoding iterations')
    ap.add_argument('--max_size', type=int, default=1024, help='dehaze max_size')
    ap.add_argument('--out', help='write the JSON report here')
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')

    if bool(args.coco) == bool(args.labels):
        ap.error('give exactly one of --coco or --labels')
    gt = load_coco(args.coco, args.images) if args.coco else load_yolo_labels(args.labels, args.images)
    texts = load_texts(args.texts, args.images) if args.texts else {}

    paths = sorted(p for p in gt if os.path.exists(p))
    if args.limit and len(paths) > args.limit:
        paths = random.Random(args.seed).sample(paths, args.limit)
        paths.sort()
    conditions = [c.strip() for c in args.conditions.split(',') if c.strip()]
    print(f'{len(paths)} images, {sum(len(gt[p]) for p in paths)} ground-truth plates, '
          f'conditions: {conditions}')

    import ipc_dehaze.plates.detection as detection
    if args.detector == 'heuristics':
        detection.yolo_model_available = lambda *a, **k: False  # old behaviour, for baseline numbers
    print('Detector:', args.detector, '| YOLO plate model:',
          'available' if detection.yolo_model_available() else 'not used / missing')
    detect = lambda im: detection.detect_license_plates_advanced(im, use_heuristics=args.detector != 'yolo')

    dehazer = None
    if 'dehazed' in conditions:
        from ipc_dehaze.dehaze import Dehazer
        dehazer = Dehazer()
        print('Dehazer device:', dehazer.device)

    accs = {c: Accumulator() for c in conditions}
    start = time.time()
    # keep the per-plate prints of the pipeline out of the benchmark log
    import contextlib
    import io
    for i, path in enumerate(paths, 1):
        clear = cv2.imread(path, cv2.IMREAD_COLOR)
        if clear is None:
            continue
        hazy = None
        for cond in conditions:
            if cond == 'clear':
                image = clear
            else:
                hazy = hazy if hazy is not None else synth_haze(clear, args.beta, args.atmosphere, args.seed + i)
                image = hazy
                if cond == 'dehazed':
                    image, _ = dehazer.dehaze(hazy, args.iterations, args.max_size)
            if cond != 'clear':
                psnr, ssim = psnr_ssim(clear, image)
                accs[cond].psnr.append(psnr)
                accs[cond].ssim.append(ssim)
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate_image(accs[cond], image, gt[path], texts.get(path), args, detect)
        if i % 10 == 0 or i == len(paths):
            print(f'  {i}/{len(paths)} images  ({time.time() - start:.0f}s)', flush=True)

    report = {'config': {k: v for k, v in vars(args).items() if k != 'out'},
              'n_images': len(paths), 'results': {c: a.summary() for c, a in accs.items()}}
    print('\n' + '=' * 78)
    for cond, res in report['results'].items():
        print(f'[{cond}]')
        for k, v in res.items():
            print(f'   {k:22s} {v}')
    print('=' * 78)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print('Report written to', args.out)


if __name__ == '__main__':
    main()
