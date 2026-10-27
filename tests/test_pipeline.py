import numpy as np

from ipc_dehaze.plates import pipeline


def test_valid_plate_is_pasted(monkeypatch, yellow_plate_image):
    monkeypatch.setattr(pipeline, 'recognize_plate_text', lambda crop: 'MH 01 AB 1234')
    output, plates = pipeline.process_plates(yellow_plate_image)
    assert plates and all(p['valid'] and p['pasted'] for p in plates)
    assert not np.array_equal(output, yellow_plate_image)


def test_invalid_plate_not_pasted_unless_requested(monkeypatch, yellow_plate_image):
    monkeypatch.setattr(pipeline, 'recognize_plate_text', lambda crop: '')
    output, plates = pipeline.process_plates(yellow_plate_image)
    assert not any(p['pasted'] for p in plates)
    assert np.array_equal(output, yellow_plate_image)

    _, plates = pipeline.process_plates(yellow_plate_image, only_valid=False)
    assert any(p['pasted'] for p in plates)


def test_emit_plate_json(capsys):
    pipeline.emit_plate_json({'formatted_text': 'MH 01 AB 1234', 'confidence': 1.0,
                              'bbox': [np.int64(1), 2, 3, 4], 'segments': {}})
    assert capsys.readouterr().out.startswith('PLATE_JSON:{"formatted_text": "MH 01 AB 1234"')
