from ipc_dehaze.plates.ocr import (best_plate_candidate, extract_plate_text, reading_order_text,
                                   validate_indian_license_plate)


def test_valid_plate_is_formatted():
    result = validate_indian_license_plate('mh-01-ab-1234')
    assert result['is_valid']
    assert result['formatted_text'] == 'MH 01 AB 1234'
    assert result['confidence'] == 1.0
    assert result['segments'] == {'state_code': 'MH', 'district_code': '01', 'series': 'AB', 'unique_number': '1234'}


def test_wrong_length_rejected():
    result = validate_indian_license_plate('MH01AB')
    assert not result['is_valid']
    assert 'length' in result['error']
    assert not validate_indian_license_plate('MH01AB123')['is_valid']  # number must have 4 digits


def test_other_real_indian_layouts_are_valid():
    for plate, formatted in [('DL1CAB1234', 'DL 1 CAB 1234'), ('MH14A1234', 'MH 14 A 1234'),
                             ('KA05MN4321', 'KA 05 MN 4321')]:
        result = validate_indian_license_plate(plate)
        assert result['is_valid'] and result['formatted_text'] == formatted


def test_extract_ignores_ind_mark_and_noise():
    assert extract_plate_text('JNDHR26DK0830') == 'HR 26 DK 0830'  # 'IND' mark glued to the front
    assert extract_plate_text('xMH14EH7958y') == 'MH 14 EH 7958'
    assert extract_plate_text('MHL4EH7958') == 'MH 14 EH 7958'      # L read for 1
    assert extract_plate_text('DO60CZ') == ''


def test_extract_prefers_the_parse_that_uses_every_character_read():
    # a shorter parse would silently drop the final '4'
    assert extract_plate_text('MHO1A81234') == 'MH 01 AB 1234'


def test_extract_prefers_common_layout_with_one_repair_over_rare_layout():
    assert extract_plate_text('UP5OAS4535') == 'UP 50 AS 4535'   # not 'UP 5 OAS 4535'
    assert extract_plate_text('HR2GDK0830') == 'HR 26 DK 0830'
    assert extract_plate_text('DL1CAB1234') == 'DL 1 CAB 1234'    # rare layout still found when nothing else fits


def test_extract_refuses_to_guess():
    assert extract_plate_text('MH8888888888') == ''     # 2 fixes + 2 ignored characters: too much guessing
    assert extract_plate_text('MH0XAB1234') == ''       # district 0 does not exist


def test_reading_order_two_line_plate():
    box = lambda x, y: [[x, y], [x + 40, y], [x + 40, y + 20], [x, y + 20]]
    # EasyOCR returns boxes in arbitrary order: bottom line first, right piece first
    results = [(box(60, 50), 'AB', 0.9), (box(0, 50), 'MH12', 0.8), (box(0, 0), 'ZZ99', 0.4), (box(60, 0), 'x', 0.1)]
    assert reading_order_text(results) == 'ZZ99XMH12AB'


def test_partial_segments_give_partial_confidence():
    result = validate_indian_license_plate('MH0XAB1234')
    assert not result['is_valid']
    assert result['confidence'] == 0.75


def test_empty_text():
    assert validate_indian_license_plate('')['error'] == 'Empty text'


def test_short_text_is_never_padded_into_a_plate():
    assert extract_plate_text('DO60CZ') == ''
    assert extract_plate_text('MH01AB12345') == 'MH 01 AB 1234'  # trailing noise is ignored, nothing is invented
    assert best_plate_candidate('DO60CZ') == 'DO60CZ'  # raw text kept, but not "valid"
    assert not validate_indian_license_plate(best_plate_candidate('DO60CZ'))['is_valid']


def test_unknown_state_code_is_rejected():
    result = validate_indian_license_plate('AA20OO0000')
    assert not result['is_valid'] and 'State code' in result['error']
    assert validate_indian_license_plate('KA 05 MN 4321')['is_valid']


def test_best_candidate_corrects_ocr_noise():
    assert best_plate_candidate('MHO1A81234') == 'MH 01 AB 1234'
    assert best_plate_candidate('') == ''
