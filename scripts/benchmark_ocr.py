"""OCR accuracy benchmark on cropped plate images.

Needs a folder of plate crops and a CSV with header ``filename,text`` (the true plate text).

Reports:
    exact_match         predicted text == true text (letters/digits only, case-insensitive)
    char_accuracy       1 - edit distance / true length, averaged
    valid_format        share of reads that are a valid Indian plate
    false_valid         valid-looking reads that are WRONG (the dangerous kind: confident but incorrect)
    precision_of_valid  of the reads reported as valid, how many are exactly right

Example:
    python scripts/benchmark_ocr.py --images benchmarks/plates_ocr/images --texts benchmarks/plates_ocr/plates.csv
"""

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'scripts'))

from benchmark_plates import char_accuracy, normalise_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', required=True, help='folder of plate crops')
    ap.add_argument('--texts', required=True, help='CSV with header filename,text')
    ap.add_argument('--limit', type=int, default=0, help='max crops (0 = all)')
    ap.add_argument('--show_errors', type=int, default=15, help='print this many wrong reads')
    ap.add_argument('--out', help='write the JSON report here')
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')

    from ipc_dehaze.plates.ocr import recognize_plate_text, validate_indian_license_plate

    with open(args.texts, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    exact = valid = false_valid = 0
    accs, errors = [], []
    start = time.time()
    for row in rows:
        image = cv2.imread(os.path.join(args.images, row['filename']), cv2.IMREAD_COLOR)
        if image is None:
            continue
        truth = normalise_text(row['text'])
        pred_text = recognize_plate_text(image)
        pred = normalise_text(pred_text)
        is_valid = validate_indian_license_plate(pred_text)['is_valid']
        exact += pred == truth
        valid += is_valid
        false_valid += is_valid and pred != truth
        accs.append(char_accuracy(pred, truth))
        if pred != truth:
            errors.append((row['filename'], truth, pred_text))

    n = len(accs)
    report = {
        'n_crops': n,
        'exact_match': round(exact / n, 4),
        'char_accuracy': round(float(np.mean(accs)), 4),
        'valid_format': round(valid / n, 4),
        'false_valid': round(false_valid / n, 4),
        'precision_of_valid': round((valid - false_valid) / valid, 4) if valid else None,
        'seconds_per_crop': round((time.time() - start) / n, 3),
    }
    print(json.dumps(report, indent=2))
    if errors and args.show_errors:
        print(f'\nFirst {min(args.show_errors, len(errors))} wrong reads (file, truth -> read):')
        for name, truth, pred in errors[:args.show_errors]:
            print(f'  {name:14s} {truth:12s} -> {pred}')
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
