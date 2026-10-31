
"""Streamlit interface for the Facial Emotion Detection System."""
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from config import EMOTION_LABELS, MODEL_PATH
from src.prediction import load_trained_model, predict_faces

st.set_page_config(page_title="Facial Emotion Detection", page_icon="🙂", layout="wide")
st.title("🙂 Deep Learning-Based Facial Emotion Detection System")
st.write(
    "A CNN-based academic prototype that predicts one of seven facial-expression "
    "categories from faces detected in an uploaded image."
)

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Home", "Detect Emotion", "About"])

if page == "Home":
    st.subheader("Project overview")
    st.write(
        "Upload a photograph on the Detect Emotion page. The application uses "
        "OpenCV to locate faces and a trained TensorFlow/Keras CNN to estimate "
        "the expression category for each detected face."
    )
    st.subheader("Supported emotion categories")
    st.write(" • ".join(EMOTION_LABELS))
    st.info("Predictions are model estimates, not reliable measures of a person's inner emotional state.")

elif page == "Detect Emotion":
    st.subheader("Upload an image")
    uploaded = st.file_uploader("Choose a JPG, JPEG or PNG image", type=["jpg", "jpeg", "png"])
    if uploaded:
        pil_image = Image.open(uploaded).convert("RGB")
        st.image(pil_image, caption="Uploaded image", use_container_width=True)
        if st.button("Detect emotions", type="primary"):
            if not MODEL_PATH.exists():
                st.error("The trained model is not available. Follow the README instructions and run `python train.py` first.")
            else:
                try:
                    image_array = np.asarray(pil_image)
                    model = load_trained_model()
                    results = predict_faces(image_array, model)
                    if not results:
                        st.warning("No face was detected. Try a clearer, front-facing photograph.")
                    else:
                        annotated = pil_image.copy()
                        draw = ImageDraw.Draw(annotated)
                        st.subheader("Prediction results")
                        for i, result in enumerate(results, start=1):
                            x, y, w, h = result["box"]
                            text = f"{result['emotion']} ({result['confidence']*100:.1f}%)"
                            draw.rectangle((x, y, x+w, y+h), outline="lime", width=3)
                            draw.text((x, max(0, y-20)), text, fill="lime", stroke_width=1, stroke_fill="black")
                            st.write(f"**Face {i}: {result['emotion']}** — confidence: {result['confidence']*100:.2f}%")
                        st.image(annotated, caption="Detected faces and predictions", use_container_width=True)
                except Exception as exc:
                    st.error(f"Prediction could not be completed: {exc}")

else:
    st.subheader("About this project")
    st.write(
        "The system uses a Convolutional Neural Network (CNN). Convolution and "
        "pooling layers learn visual patterns from 48×48 grayscale face images; "
        "dense layers produce a seven-class softmax output."
    )
    st.write("Dataset: FER-2013. The project supports the standard CSV format and a class-folder image format.")
    st.write("Technologies: Python, TensorFlow/Keras, OpenCV, NumPy, Pandas, scikit-learn, Matplotlib and Streamlit.")
