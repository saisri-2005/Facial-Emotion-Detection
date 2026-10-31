# Facial Emotion Detection System

A Deep Learning-based Facial Emotion Detection System that identifies human emotions from facial images using a Convolutional Neural Network (CNN).

## Overview

This project uses the FER-2013 dataset to classify facial expressions into seven emotion categories using deep learning.

## Emotions Detected

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* Streamlit

## Project Structure

```text
Facial-Emotion-Detection/
├── assets/
├── dataset/
├── models/
├── notebooks/
├── results/
├── src/
├── tests/
├── app.py
├── train.py
├── predict.py
├── evaluate.py
├── config.py
└── requirements.txt
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/saisri-2005/Facial-Emotion-Detection.git
```

2. Navigate to the project:

```bash
cd Facial-Emotion-Detection
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Train the model:

```bash
python train.py
```

Run emotion prediction:

```bash
python predict.py
```

Launch the application:

```bash
streamlit run app.py
```

## Dataset

FER-2013 (Facial Expression Recognition 2013) dataset.
