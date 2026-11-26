import os

import cv2
import pytest
import torch

from ipc_dehaze.config import CRITIC_PATH, PREDICTOR_PATH
from ipc_dehaze.dehaze import looks_blank, mirror_pad


def test_mirror_pad_to_multiple_of_32():
    padded = mirror_pad(torch.rand(1, 3, 50, 70))
    assert padded.shape == (1, 3, 64, 96)


def test_looks_blank():
    import numpy as np
    assert looks_blank(np.zeros((10, 10, 3), np.uint8))
    assert looks_blank(np.full((10, 10, 3), 255, np.uint8))
    assert not looks_blank(np.random.randint(0, 255, (10, 10, 3), np.uint8))


def test_dehaze_retries_smaller_on_cuda_oom(monkeypatch):
    from ipc_dehaze.dehaze import Dehazer
    sizes = []

    def fake_once(self, image, n, max_size, resize_to, force_size):
        sizes.append(max_size)
        if len(sizes) < 3:
            raise torch.cuda.OutOfMemoryError('simulated')
        return 'ok'

    monkeypatch.setattr(Dehazer, '_dehaze_once', fake_once)
    assert Dehazer.__new__(Dehazer).dehaze(None, 8, 1000) == 'ok'
    assert sizes == [1000, 750, 550]


def test_dehaze_oom_with_forced_size_is_not_retried(monkeypatch):
    from ipc_dehaze.dehaze import Dehazer

    def always_oom(self, *args):
        raise torch.cuda.OutOfMemoryError('simulated')

    monkeypatch.setattr(Dehazer, '_dehaze_once', always_oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        Dehazer.__new__(Dehazer).dehaze(None, 8, 1000, 0, (480, 640))


@pytest.mark.skipif(not (os.path.exists(PREDICTOR_PATH) and os.path.exists(CRITIC_PATH)),
                    reason='pretrained weights not downloaded')
def test_dehaze_end_to_end(data_image):
    from ipc_dehaze.dehaze import Dehazer
    image = cv2.resize(data_image('hazy_plate_original.jpg'), (128, 96))
    dehazed, _ = Dehazer().dehaze(image, num_iterations=2)
    assert dehazed.shape == image.shape
