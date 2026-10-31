
# Deep Learning-Based Facial Emotion Detection System

**Institution:** Prasad V Potluri Siddhartha Institute of Technology  
**Department:** Computer Science & Engineering  
**Academic Year:** 2026–2027

## 1. Project overview

This academic project builds a facial-expression classification prototype using a Convolutional Neural Network (CNN). OpenCV detects face regions in an uploaded image; a TensorFlow/Keras model predicts one of seven FER-2013 expression labels for each detected face.

The model predicts visible expression categories. It does not establish what a person is actually feeling, and predictions may be affected by lighting, pose, image quality, demographic imbalance and dataset limitations.

## 2. Problem statement

Manual identification of facial expressions from images is subjective and time-consuming. This project explores an automated image-classification pipeline that detects faces and classifies facial-expression patterns into seven predefined categories.

## 3. Objectives and features

- Load and preprocess FER-2013 images.
- Train a CNN on 48 × 48 grayscale facial images.
- Evaluate the model with accuracy, weighted precision, recall, F1-score, classification report and confusion matrix.
- Detect multiple faces in a JPG, JPEG or PNG image.
- Display a predicted class and model confidence for each detected face.
- Provide a Streamlit interface and a command-line prediction script.

## 4. Emotion classes

The FER-2013 label order used by this implementation is:

| Label | Emotion |
|---:|---|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

## 5. Technologies

Python, TensorFlow/Keras, CNN, OpenCV, NumPy, Pandas, scikit-learn, Matplotlib, Seaborn, Streamlit and pytest.

## 6. System architecture

```text
FER-2013 dataset
      |
      v
Data loading and preprocessing
(grayscale, 48x48, normalization, label mapping)
      |
      v
CNN training and validation
      |
      +------> Saved Keras model
      |               |
      v               v
Test evaluation    Uploaded image
                      |
                      v
                OpenCV face detection
                      |
                      v
                Face crop and preprocessing
                      |
                      v
                CNN emotion prediction
                      |
                      v
                Streamlit / CLI results
```

## 7. CNN architecture

The model uses three convolutional blocks (32, 64 and 128 filters). Each block contains convolution, batch normalization, ReLU activation and max pooling, followed by dropout. The feature maps are flattened, passed through a 128-unit dense layer and dropout, and classified by a seven-unit softmax layer. The model is compiled with Adam and sparse categorical cross-entropy.

## 8. Dataset

Use FER-2013. It is not included in this repository. Download it from a legitimate source (for example, search Kaggle for “FER-2013 facial expression recognition”) and follow its current terms. Setup instructions and both supported layouts are in [`dataset/README.md`](dataset/README.md).

The preferred CSV has `emotion`, `pixels`, and optionally `Usage` columns. If `Usage` is present, Training/PublicTest/PrivateTest map to train/validation/test. If it is absent, the loader creates a stratified 80/10/10 split. A folder-based alternative is also supported.

## 9. Folder structure

```text
Facial-Emotion-Detection/
├── app.py
├── train.py
├── evaluate.py
├── predict.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
├── dataset/README.md
├── models/README.md
├── notebooks/facial_emotion_detection.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── face_detection.py
│   ├── prediction.py
│   └── visualization.py
├── results/README.md
├── assets/README.md
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py
    └── test_model.py
```

## 10. Windows setup (PowerShell)

Open PowerShell in the project root (the folder containing `train.py`).

```powershell
# 1. Create a virtual environment
py -3.11 -m venv .venv

# 2. Activate it
.\.venv\Scripts\Activate.ps1

# If PowerShell blocks activation for this terminal session:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Python 3.10 or 3.11 is recommended for compatibility with the listed TensorFlow range. If `py -3.11` is unavailable, install a supported Python version and use its launcher command.

## 11. Dataset setup

Download `fer2013.csv` and copy it to `dataset\fer2013.csv`. Do not rename the CSV columns. Alternatively, follow the folder layout in `dataset/README.md`.

## 12. Train the CNN

```powershell
python train.py
```

The best validation-loss checkpoint is saved to `models/facial_emotion_cnn.keras`. Training history and accuracy/loss plots are saved in `results/`. Training may take time and benefits from a compatible GPU, but CPU training is supported.

## 13. Evaluate the model

```powershell
python evaluate.py
```

This evaluates the held-out test split and writes `metrics.json`, `classification_report.txt` and `confusion_matrix.png` to `results/`. The metrics are generated only after you run the experiment; this repository does not claim precomputed results.

## 14. Predict an image from the command line

```powershell
python predict.py "C:\path\to\your\photo.jpg"
```

The command prints the detected face count implicitly through per-face results, predicted class and confidence. If no face is detected, it prints a message.

## 15. Run the Streamlit application

```powershell
streamlit run app.py
```

Open the local address printed by Streamlit in your browser. Upload a JPG, JPEG or PNG image and select **Detect emotions**. Train the model first; the app displays an error if model weights are missing.

## 16. Run tests

```powershell
pytest -q
```

Tests cover grayscale/RGB preprocessing, output dimensions, invalid input, model output shape and softmax probability behavior. TensorFlow must be installed.

## 17. Sample output format

After running inference, terminal output will look like this (illustrative format only, not a measured result):

```text
Face 1: Happy (82.34% confidence)
Face 2: Neutral (67.10% confidence)
```

The numbers above are examples of formatting, not project results.

## 18. Results

Complete this section after training and evaluation:

- Test accuracy: _to be measured_
- Weighted precision: _to be measured_
- Weighted recall: _to be measured_
- Weighted F1-score: _to be measured_
- Confusion matrix: generated at `results/confusion_matrix.png`

Do not report example values as experimental findings.

## 19. Future enhancements

- Compare the baseline CNN with transfer-learning approaches.
- Apply data augmentation and class-imbalance strategies.
- Add webcam/video inference with clear consent and privacy safeguards.
- Improve face detection for varied pose and lighting.
- Conduct subgroup performance and bias analysis.
- Add model calibration and more detailed error analysis.

## 20. References

1. Goodfellow, I. J., et al. (2013). *Challenges in Representation Learning: A report on three machine learning contests*. Neural Information Processing.
2. TensorFlow/Keras documentation: https://www.tensorflow.org/guide/keras
3. OpenCV documentation: https://docs.opencv.org/
4. Streamlit documentation: https://docs.streamlit.io/
5. FER-2013 dataset source: use the dataset publisher's page and cite the exact version/access date used in your final academic report.

## 21. GitHub upload

The dataset, trained model weights and generated experiment outputs are ignored by Git. From the project root:

```powershell
git init
git add .
git status
git commit -m "Add facial emotion detection CNN project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/Facial-Emotion-Detection.git
git push -u origin main
```

Replace the remote URL with your own GitHub repository URL. Check `git status` and `git ls-files` before pushing to confirm that no dataset or model weights are staged. Share large trained weights separately if needed, using an approved release or storage location.

## 22. Important limitations

- Haar cascade detection may miss faces or produce false detections.
- FER-2013 images are small and collected under constrained conditions; real-world performance may differ.
- A softmax confidence is the model's score, not a guarantee of correctness.
- Facial expression categories should not be treated as definitive evidence of a person's internal emotional state.
