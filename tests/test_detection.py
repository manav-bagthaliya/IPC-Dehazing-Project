import pytest

from ipc_dehaze.plates.detection import (bbox_overlap, detect_license_plates_advanced, detect_yellow_license_plates,
                                         remove_duplicate_detections)


def test_bbox_overlap():
    assert bbox_overlap((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert bbox_overlap((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_overlap((0, 0, 10, 10), (5, 0, 15, 10)) == 0.5


def test_remove_duplicates_keeps_most_confident():
    plates = [
        {'bbox': [0, 0, 100, 30], 'confidence': 0.5},
        {'bbox': [2, 2, 100, 30], 'confidence': 0.9},
        {'bbox': [300, 300, 400, 330], 'confidence': 0.7},
    ]
    kept = remove_duplicate_detections(plates)
    assert [p['confidence'] for p in kept] == [0.9, 0.7]


def test_remove_duplicates_caps_count():
    plates = [{'bbox': [i * 100, 0, i * 100 + 50, 20], 'confidence': 0.5} for i in range(10)]
    assert len(remove_duplicate_detections(plates)) == 5


def test_yellow_plate_found(yellow_plate_image):
    plates = detect_yellow_license_plates(yellow_plate_image)
    assert any(bbox_overlap(p['bbox'], (200, 200, 501, 251)) > 0.9 for p in plates)


@pytest.mark.parametrize('name', ['yellow_plate_test.jpg', 'white_plate_test.jpg', 'blue_plate_test.jpg',
                                  'multiple_plates_test.jpg', 'hazy_plate_original.jpg'])
def test_advanced_detection_runs_on_samples(data_image, name):
    plates = detect_license_plates_advanced(data_image(name))
    assert len(plates) <= 5
    for p in plates:
        assert len(p['bbox']) == 4 and {'confidence', 'method', 'color'} <= p.keys()
