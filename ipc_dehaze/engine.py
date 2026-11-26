"""One shared pipeline for the CLI and the web app: dehaze -> (plates) -> optional post-enhance.

The networks are loaded once (lazily, thread-safe) and reused for every image, so a long-running
server pays the model-loading cost a single time instead of once per request.
"""

import os
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ipc_dehaze.config import CRITIC_PATH, DEFAULT_MAX_SIZE, DEFAULT_NUM_ITERATIONS, PREDICTOR_PATH


@dataclass
class DehazeResult:
    dehazed: np.ndarray            # clean model output (BGR uint8)
    final: np.ndarray              # dehazed + enhanced plate regions (+ optional global enhance)
    processed_input: np.ndarray    # the resized image the network actually saw
    plates: list = field(default_factory=list)   # JSON-safe dicts, see plates.pipeline.summarize_plate
    seconds: float = 0.0
    used_fallback: bool = False    # True if the output looked blank and the input was returned instead


class Engine:
    """Lazy, thread-safe holder of the Dehazer plus the full per-image pipeline."""

    def __init__(self, predictor_path=PREDICTOR_PATH, critic_path=CRITIC_PATH, device=None, half=None):
        self._paths = (predictor_path, critic_path)
        self._device = device
        # fp16 on the GPU: ~1.7x faster, same accuracy (-0.02 dB PSNR vs clear images, measured).
        # Set IPC_FP32=1 (or half=False) for exact full precision.
        self._half = (os.environ.get('IPC_FP32') != '1') if half is None else half
        self._dehazer = None
        self._lock = threading.Lock()
        self.error = None

    @property
    def dehazer(self):
        if self._dehazer is None:
            with self._lock:
                if self._dehazer is None:
                    from ipc_dehaze.dehaze import Dehazer
                    try:
                        self._dehazer = Dehazer(*self._paths, device=self._device, half=self._half)
                    except Exception as exc:  # keep the reason so /health can report it
                        self.error = str(exc)
                        raise
        return self._dehazer

    @property
    def ready(self):
        return self._dehazer is not None

    def warm_up(self, with_plates=True):
        """Load the networks (and the plate models) now, so the first request is fast."""
        _ = self.dehazer
        if with_plates:
            from ipc_dehaze.plates import detection
            if detection.yolo_model_available():
                import numpy as _np
                detection.detect_license_plates_yolo(_np.zeros((64, 64, 3), _np.uint8))

    def info(self):
        import torch
        device = str(self._dehazer.device) if self._dehazer else ('cuda' if torch.cuda.is_available() else 'cpu')
        return {
            'ready': self.ready,
            'error': self.error,
            'device': device,
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'half_precision': bool(self._dehazer.half) if self._dehazer else None,
        }

    def process(self, image, iterations=DEFAULT_NUM_ITERATIONS, max_size=DEFAULT_MAX_SIZE, resize_to=0,
                force_size=None, detect_plates=True, post_enhance=False):
        """Run the whole pipeline on one BGR uint8 image."""
        from ipc_dehaze.dehaze import looks_blank
        from ipc_dehaze.enhance import enhance_output_quality

        start = time.time()
        dehazed, processed_input = self.dehazer.dehaze(image, iterations, max_size, resize_to, force_size)

        plates = []
        final = dehazed
        if detect_plates:
            from ipc_dehaze.plates import process_plates
            from ipc_dehaze.plates.pipeline import summarize_plate
            final, found = process_plates(dehazed)
            plates = [summarize_plate(p) for p in found]
        if post_enhance:
            final = enhance_output_quality(final)

        used_fallback = False
        if looks_blank(final):
            final = cv2.resize(image, (final.shape[1], final.shape[0]))
            used_fallback = True

        return DehazeResult(dehazed, final, processed_input, plates, time.time() - start, used_fallback)
