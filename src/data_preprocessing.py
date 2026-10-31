
"""Dataset loading and preprocessing for FER-2013."""
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from config import (
    IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES, RANDOM_SEED,
    FER_CSV_PATH, FOLDER_DATASET_DIR
)

USAGE_MAP = {
    "Training": "train", "PublicTest": "validation", "PrivateTest": "test",
    "train": "train", "training": "train", "validation": "validation",
    "valid": "validation", "val": "validation", "test": "test"
}

def preprocess_image(image):
    """Convert an image to grayscale 48x48 float32 data in [0, 1]."""
    if image is None:
        raise ValueError("Image is empty or could not be read.")
    tensor = tf.convert_to_tensor(image)
    if tensor.shape.rank == 2:
        tensor = tf.expand_dims(tensor, axis=-1)
    elif tensor.shape.rank == 3 and tensor.shape[-1] == 3:
        tensor = tf.image.rgb_to_grayscale(tensor)
    elif tensor.shape.rank != 3 or tensor.shape[-1] != 1:
        raise ValueError("Expected a grayscale or RGB image.")
    tensor = tf.image.resize(tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
    tensor = tf.cast(tensor, tf.float32)
    if float(tf.reduce_max(tensor).numpy()) > 1.0:
        tensor = tensor / 255.0
    return tf.clip_by_value(tensor, 0.0, 1.0).numpy().astype("float32")

def _split_arrays(x, y):
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def load_fer_csv(csv_path=FER_CSV_PATH):
    """Load common FER-2013 CSV format (emotion, pixels, optional Usage)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found: {csv_path}. See dataset/README.md for setup."
        )
    df = pd.read_csv(csv_path)
    required = {"emotion", "pixels"}
    if not required.issubset(df.columns):
        raise ValueError("FER CSV must contain 'emotion' and 'pixels' columns.")

    images, labels = [], []
    for row in df[["emotion", "pixels"]].itertuples(index=False):
        try:
            label = int(row.emotion)
            pixels = np.fromstring(str(row.pixels), sep=" ", dtype=np.float32)
            if label not in range(NUM_CLASSES) or pixels.size != IMAGE_HEIGHT * IMAGE_WIDTH:
                continue
            images.append(pixels.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 1) / 255.0)
            labels.append(label)
        except (TypeError, ValueError):
            continue
    if not images:
        raise ValueError("No valid FER-2013 rows were found.")
    x = np.asarray(images, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    if "Usage" in df.columns:
        valid_indices = []
        usage = []
        # Match cleaned rows to the original rows without relying on dataframe indices.
        for idx, row in enumerate(df[["emotion", "pixels", "Usage"]].itertuples(index=False)):
            try:
                label = int(row.emotion)
                pixels = np.fromstring(str(row.pixels), sep=" ", dtype=np.float32)
                if label in range(NUM_CLASSES) and pixels.size == IMAGE_HEIGHT * IMAGE_WIDTH:
                    key = str(row.Usage).strip()
                    valid_indices.append(idx)
                    usage.append(USAGE_MAP.get(key, USAGE_MAP.get(key.lower())))
            except (TypeError, ValueError):
                pass
        if len(usage) == len(x) and all(usage):
            usage = np.asarray(usage)
            splits = []
            for name in ("train", "validation", "test"):
                mask = usage == name
                if mask.sum() == 0:
                    raise ValueError(f"No rows found for the '{name}' split in Usage column.")
                splits.append((x[mask], y[mask]))
            return tuple(splits)
    return _split_arrays(x, y)

def load_folder_dataset(data_dir=FOLDER_DATASET_DIR):
    """Load class-folder images: train|validation|test/<emotion>/*."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Image dataset directory not found: {data_dir}")
    class_names = {name.lower(): i for i, name in enumerate(
        ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    )}
    def load_split(split_name):
        split_dir = data_dir / split_name
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing folder: {split_dir}")
        xs, ys = [], []
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name.lower() not in class_names:
                continue
            label = class_names[class_dir.name.lower()]
            for path in class_dir.iterdir():
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                image = tf.io.decode_image(tf.io.read_file(str(path)), channels=1, expand_animations=False)
                xs.append(preprocess_image(image))
                ys.append(label)
        if not xs:
            raise ValueError(f"No supported images found in {split_dir}.")
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)
    return load_split("train"), load_split("validation"), load_split("test")

def load_dataset():
    """Prefer the standard FER CSV; otherwise use folder-based images."""
    if FER_CSV_PATH.exists():
        return load_fer_csv(FER_CSV_PATH)
    return load_folder_dataset(FOLDER_DATASET_DIR)
