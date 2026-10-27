"""Phase 2 CLI: licence plate detection + OCR on images already dehazed by inference.py.

Example:
    python inference.py -i raw_images -o dehazed --dehaze_only
    python detect_plates.py -i dehazed -o final
"""

import argparse
import glob
import os

import cv2
from tqdm import tqdm

from ipc_dehaze.enhance import binarize_plate_region
from ipc_dehaze.plates import process_plates

PNG_LOSSLESS = [cv2.IMWRITE_PNG_COMPRESSION, 0]
CROP_WIDTH = 900


def main():
    parser = argparse.ArgumentParser(description='Phase 2: License Plate Detection on Dehazed Images')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='a dehazed image, or a folder of *_dehazed.png files')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='output folder for enhanced images and plate crops')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    paths = [args.input] if os.path.isfile(args.input) else sorted(
        glob.glob(os.path.join(args.input, '*_dehazed.png')))
    if not paths:
        print(f"No dehazed images found in {args.input}")
        return
    print(f"Found {len(paths)} dehazed images to process")

    for path in tqdm(paths, unit='image'):
        imgname = os.path.splitext(os.path.basename(path))[0].replace('_dehazed', '')
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Could not load image: {path}")
            continue

        print(f"\n=== Processing {imgname} ===")
        enhanced_image, plates = process_plates(image, enhancer=binarize_plate_region, only_valid=False)

        output_path = os.path.join(args.output, f'{imgname}.png')
        cv2.imwrite(output_path, enhanced_image, PNG_LOSSLESS)
        print(f"Saved image: {output_path}")

        # High-res crops of every pasted plate, for the UI
        crop_index = 0
        for plate in plates:
            if not plate['pasted']:
                continue
            crop_index += 1
            x1, y1, x2, y2 = plate['bbox']
            crop = enhanced_image[y1:y2, x1:x2]
            scale = CROP_WIDTH / max(1, crop.shape[1])
            crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
            crop_path = os.path.join(args.output, f'{imgname}_plate_{crop_index}.png')
            cv2.imwrite(crop_path, crop, PNG_LOSSLESS)
            print(f"Saved plate crop: {crop_path}")

    print("\n=== Phase 2 Complete ===")


if __name__ == '__main__':
    main()
