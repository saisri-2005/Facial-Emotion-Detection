
"""Model loading and face-level emotion predictions."""
from pathlib import Path
import numpy as np
import tensorflow as tf
from config import MODEL_PATH, EMOTION_LABELS
from src.data_preprocessing import preprocess_image
from src.face_detection import detect_faces, extract_face

def load_trained_model(model_path=MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run 'python train.py' first."
        )
    return tf.keras.models.load_model(model_path)

def predict_faces(image, model):
    boxes = detect_faces(image)
    predictions = []
    for box in boxes:
        face = extract_face(image, box)
        face_input = preprocess_image(face)
        probabilities = model.predict(np.expand_dims(face_input, axis=0), verbose=0)[0]
        class_index = int(np.argmax(probabilities))
        predictions.append({
            "box": tuple(int(v) for v in box),
            "emotion": EMOTION_LABELS[class_index],
            "confidence": float(probabilities[class_index]),
            "probabilities": probabilities.tolist()
        })
    return predictions
