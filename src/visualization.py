
"""Plot training history and evaluation confusion matrices."""
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def save_training_history(history, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hist = history.history
    for metric, title, filename in [
        ("accuracy", "Training and validation accuracy", "accuracy.png"),
        ("loss", "Training and validation loss", "loss.png")
    ]:
        plt.figure(figsize=(8, 5))
        plt.plot(hist.get(metric, []), label=f"Training {metric}")
        val_key = f"val_{metric}"
        if val_key in hist:
            plt.plot(hist[val_key], label=f"Validation {metric}")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150)
        plt.close()

def save_confusion_matrix(y_true, y_pred, labels, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, xticks_rotation=35,
        cmap="Blues", normalize=None
    )
    display.figure_.set_size_inches(9, 7)
    display.figure_.tight_layout()
    display.figure_.savefig(output_path, dpi=150)
    plt.close(display.figure_)
