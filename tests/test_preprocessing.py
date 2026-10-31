
import numpy as np
import pytest
from src.data_preprocessing import preprocess_image

def test_preprocess_grayscale_dimensions_and_range():
    image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    result = preprocess_image(image)
    assert result.shape == (48, 48, 1)
    assert result.dtype == np.float32
    assert result.min() >= 0.0 and result.max() <= 1.0

def test_preprocess_rgb_image():
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    assert preprocess_image(image).shape == (48, 48, 1)

def test_invalid_image_raises():
    with pytest.raises(ValueError):
        preprocess_image(None)
