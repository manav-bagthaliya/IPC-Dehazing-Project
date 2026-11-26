import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

import benchmark_plates as bp  # noqa: E402


def test_iou():
    assert bp.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert bp.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert bp.iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_match_boxes_is_one_to_one_and_confidence_ordered():
    gts = [[0, 0, 10, 10]]
    preds = [{'bbox': [0, 0, 10, 10], 'confidence': 0.4}, {'bbox': [1, 1, 10, 10], 'confidence': 0.9}]
    matches = bp.match_boxes(preds, gts, 0.5)
    assert len(matches) == 1 and matches[0][0] == 1  # the more confident box wins, the other is a FP


def test_average_precision_perfect_and_empty():
    assert bp.average_precision([(0.9, True), (0.8, True)], 2) == pytest.approx(1.0)
    assert bp.average_precision([], 3) == 0.0
    assert bp.average_precision([(0.9, False)], 1) == 0.0


def test_text_metrics():
    assert bp.normalise_text('mh 01-ab.1234') == 'MH01AB1234'
    assert bp.edit_distance('KITTEN', 'SITTING') == 3
    assert bp.char_accuracy('MH01AB1234', 'MH01AB1234') == 1.0
    assert bp.char_accuracy('MH01AB1239', 'MH01AB1234') == pytest.approx(0.9)
    assert bp.char_accuracy('', '') == 0.0


def test_synth_haze_is_deterministic_and_lifts_dark_images():
    img = np.full((64, 96, 3), 20, np.uint8)
    a, b = bp.synth_haze(img, seed=3), bp.synth_haze(img, seed=3)
    assert np.array_equal(a, b) and a.shape == img.shape
    assert a.mean() > img.mean() + 50  # airlight washes out the dark scene


def test_load_coco_converts_xywh_to_xyxy(tmp_path):
    import json
    ann = {'images': [{'id': 1, 'file_name': 'a.jpg'}, {'id': 2, 'file_name': 'b.jpg'}],
           'annotations': [{'image_id': 1, 'bbox': [10, 20, 30, 40]}]}
    path = tmp_path / 'ann.json'
    path.write_text(json.dumps(ann))
    gt = bp.load_coco(str(path), 'imgs')
    assert gt[os.path.join('imgs', 'a.jpg')] == [[10, 20, 40, 60]]
    assert gt[os.path.join('imgs', 'b.jpg')] == []
