from ipc_dehaze.plates.ocr import best_plate_candidate, coerce_to_pattern, validate_indian_license_plate


def test_valid_plate_is_formatted():
    result = validate_indian_license_plate('mh-01-ab-1234')
    assert result['is_valid']
    assert result['formatted_text'] == 'MH 01 AB 1234'
    assert result['confidence'] == 1.0
    assert result['segments'] == {'state_code': 'MH', 'district_code': '01', 'series': 'AB', 'unique_number': '1234'}


def test_wrong_length_rejected():
    result = validate_indian_license_plate('MH01AB123')
    assert not result['is_valid']
    assert 'length' in result['error']


def test_partial_segments_give_partial_confidence():
    result = validate_indian_license_plate('MH0XAB1234')
    assert not result['is_valid']
    assert result['confidence'] == 0.75


def test_empty_text():
    assert validate_indian_license_plate('')['error'] == 'Empty text'


def test_coerce_fixes_common_confusions():
    # O->0 in digit slots, 8->B in letter slots
    assert coerce_to_pattern('MHO1A81234') == 'MH01AB1234'


def test_best_candidate_corrects_ocr_noise():
    assert best_plate_candidate('MHO1A81234') == 'MH 01 AB 1234'
    assert best_plate_candidate('') == ''
