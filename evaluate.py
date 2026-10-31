
"""Evaluate a saved CNN on the held-out test split."""
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from config import MODEL_PATH, RESULTS_DIR, EMOTION_LABELS
from src.data_preprocessing import load_dataset
from src.prediction import load_trained_model
from src.visualization import save_confusion_matrix

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model is missing. Train it first with: python train.py")
    (_, _), (_, _), (x_test, y_test) = load_dataset()
    model = load_trained_model()
    probabilities = model.predict(x_test, batch_size=64, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    report = classification_report(
        y_test, y_pred, labels=list(range(len(EMOTION_LABELS))),
        target_names=EMOTION_LABELS, zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy), "weighted_precision": float(precision),
        "weighted_recall": float(recall), "weighted_f1_score": float(f1)
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(y_test, y_pred, EMOTION_LABELS, RESULTS_DIR / "confusion_matrix.png")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n", report)
    print(f"Evaluation files saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
