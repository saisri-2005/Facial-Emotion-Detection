
"""Command-line emotion prediction for an image."""
import argparse
import cv2
from src.prediction import load_trained_model, predict_faces

def main():
    parser = argparse.ArgumentParser(description="Predict emotions in faces in an image.")
    parser.add_argument("image", help="Path to a JPG, JPEG or PNG image.")
    args = parser.parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model = load_trained_model()
    results = predict_faces(image, model)
    if not results:
        print("No face detected in the supplied image.")
        return
    for index, result in enumerate(results, start=1):
        print(
            f"Face {index}: {result['emotion']} "
            f"({result['confidence'] * 100:.2f}% confidence)"
        )

if __name__ == "__main__":
    main()
