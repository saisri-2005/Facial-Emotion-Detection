
import numpy as np
from src.model import build_model
from config import NUM_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH

def test_model_output_shape_and_probabilities():
    model = build_model()
    output = model.predict(np.zeros((2, IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32), verbose=0)
    assert output.shape == (2, NUM_CLASSES)
    assert np.allclose(output.sum(axis=1), 1.0, atol=1e-5)
