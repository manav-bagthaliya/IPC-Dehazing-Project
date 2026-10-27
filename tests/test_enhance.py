import numpy as np

from ipc_dehaze.enhance import binarize_plate, binarize_plate_region, enhance_output_quality, enhance_plate_region


def test_enhance_plate_region_returns_uint8_crop(yellow_plate_image):
    crop = enhance_plate_region(yellow_plate_image, [200, 200, 500, 250])
    assert crop.dtype == np.uint8
    assert crop.shape[1] == 400 and crop.shape[2] == 3


def test_enhance_plate_region_empty_bbox_returns_image(yellow_plate_image):
    assert enhance_plate_region(yellow_plate_image, [10, 10, 10, 10]) is yellow_plate_image


def test_binarize(yellow_plate_image):
    crop = binarize_plate_region(yellow_plate_image, [200, 200, 500, 250])
    assert crop.dtype == np.uint8 and crop.shape[1] == 400
    assert binarize_plate(yellow_plate_image[200:250, 200:500]).shape == crop.shape


def test_output_quality_keeps_shape(yellow_plate_image):
    out = enhance_output_quality(yellow_plate_image)
    assert out.shape == yellow_plate_image.shape and out.dtype == np.uint8
