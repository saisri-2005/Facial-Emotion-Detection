
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMAGE_HEIGHT = 48
IMAGE_WIDTH = 48
CHANNELS = 1
NUM_CLASSES = 7
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

DATASET_DIR = BASE_DIR / "dataset"
FER_CSV_PATH = DATASET_DIR / "fer2013.csv"
FOLDER_DATASET_DIR = DATASET_DIR / "images"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "facial_emotion_cnn.keras"
RESULTS_DIR = BASE_DIR / "results"
ASSETS_DIR = BASE_DIR / "assets"

for directory in (MODEL_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
