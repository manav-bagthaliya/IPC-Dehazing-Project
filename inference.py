"""Phase 1 CLI: dehaze images, optionally followed by licence plate detection + OCR.

Examples:
    python inference.py -i examples -o results
    python inference.py -i examples -o results --enhance_plates
    python inference.py -i examples -o results --dehaze_only
"""

import argparse
import glob
import os

import cv2
from tqdm import tqdm

from ipc_dehaze.config import CRITIC_PATH, DEFAULT_MAX_SIZE, DEFAULT_NUM_ITERATIONS, PREDICTOR_PATH
from ipc_dehaze.dehaze import Dehazer, looks_blank
from ipc_dehaze.enhance import enhance_output_quality

PNG_LOSSLESS = [cv2.IMWRITE_PNG_COMPRESSION, 0]


def parse_args():
    parser = argparse.ArgumentParser(description='IPC-Dehaze: dehaze images (and optionally enhance plates)')
    parser.add_argument('--predictor_path', type=str, default=PREDICTOR_PATH)
    parser.add_argument('--critic_path', type=str, default=CRITIC_PATH)
    parser.add_argument('-i', '--input', type=str, default='examples', help='input image or folder')
    parser.add_argument('-o', '--output', type=str, default='results', help='output folder')
    parser.add_argument('-n', type=int, default=DEFAULT_NUM_ITERATIONS, help='number of decoding iterations')
    parser.add_argument('--max_size', type=int, default=DEFAULT_MAX_SIZE,
                        help='downscale images larger than max_size^2 pixels for processing')
    parser.add_argument('--resize_to', type=int, default=0,
                        help='if > 0, upscale inputs whose long edge is smaller than this value')
    parser.add_argument('--force_width', type=int, default=0,
                        help='with --force_height, resize ALL inputs to this width')
    parser.add_argument('--force_height', type=int, default=0,
                        help='with --force_width, resize ALL inputs to this height')
    parser.add_argument('--enhance_plates', action='store_true', help='detect, enhance and OCR licence plates')
    parser.add_argument('--dehaze_only', action='store_true', help='only dehaze; skip plate detection')
    return parser.parse_args()


def run_plate_stage(image):
    """Plate detection + OCR on a dehazed image; prints PLATE_JSON lines for the web app."""
    from ipc_dehaze.plates import emit_plate_json, process_plates

    print("=" * 60)
    print("LICENSE PLATE DETECTION + OCR")
    print("=" * 60)
    output, plates = process_plates(image)

    valid = [p for p in plates if p['valid']]
    if valid:
        print(f"\n✅ Successfully processed {len(valid)} valid license plate(s):")
        for plate in valid:
            print(f"   Plate: {plate['formatted_text']}  Segments: {plate['segments']}")
            emit_plate_json(plate)
    else:
        print("\n⚠️ No valid Indian license plates detected")
    print("=" * 60)
    return output


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    resized_inputs_dir = os.path.join(args.output, 'resized_inputs')
    os.makedirs(resized_inputs_dir, exist_ok=True)

    dehazer = Dehazer(args.predictor_path, args.critic_path)
    force_size = (args.force_height, args.force_width) if args.force_width > 0 and args.force_height > 0 else None
    with_plates = args.enhance_plates and not args.dehaze_only

    paths = [args.input] if os.path.isfile(args.input) else sorted(glob.glob(os.path.join(args.input, '*')))

    for path in tqdm(paths, unit='image'):
        imgname = os.path.splitext(os.path.basename(path))[0]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Error: Could not load image {path}")
            continue

        try:
            dehazed, processed_input = dehazer.dehaze(
                image, args.n, args.max_size, args.resize_to, force_size)
        except Exception as e:
            print(f"Error processing {imgname}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Clean dehazed result (before any plate/colour enhancement)
        dehazed_path = os.path.join(args.output, f'{imgname}_dehazed.png')
        cv2.imwrite(dehazed_path, dehazed, PNG_LOSSLESS)
        print(f"Saved dehazed image: {dehazed_path}")
        cv2.imwrite(os.path.join(resized_inputs_dir, f'{imgname}.png'), processed_input)

        if args.dehaze_only:
            continue

        final = run_plate_stage(dehazed) if with_plates else dehazed
        final = enhance_output_quality(final)
        if looks_blank(final):
            print(f"Warning: low-content output for {imgname}; falling back to the input image.")
            final = cv2.resize(image, (final.shape[1], final.shape[0]))
        cv2.imwrite(os.path.join(args.output, f'{imgname}.png'), final, PNG_LOSSLESS)


if __name__ == '__main__':
    main()
