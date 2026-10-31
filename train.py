
"""Train the facial emotion CNN using FER-2013."""
import json
import tensorflow as tf
from config import BATCH_SIZE, EPOCHS, MODEL_PATH, RESULTS_DIR, RANDOM_SEED
from src.data_preprocessing import load_dataset
from src.model import build_model
from src.visualization import save_training_history

def main():
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_dataset()
    print(f"Train: {len(x_train)} | Validation: {len(x_val)} | Test: {len(x_test)}")
    model = build_model()
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH), monitor="val_loss",
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
        )
    ]
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_training_history(history, RESULTS_DIR)
    with open(RESULTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in values] for k, values in history.history.items()}, f, indent=2)
    print(f"Best model checkpoint saved to: {MODEL_PATH}")
    print(f"Training history and plots saved in: {RESULTS_DIR}")
    print(f"Test split reserved for final evaluation ({len(x_test)} samples). Run python evaluate.py.")

if __name__ == "__main__":
    main()
