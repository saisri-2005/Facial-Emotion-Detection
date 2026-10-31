
# FER-2013 dataset setup

The project does **not** include FER-2013. Obtain it from a legitimate dataset source such as the FER-2013 dataset page on Kaggle (search for **FER-2013 facial expression recognition**). Follow the source's current access and licensing terms; a Kaggle account may be required.

## Option A: Standard CSV (recommended)

1. Download the `fer2013.csv` file.
2. Place it directly in this directory:

   `Facial-Emotion-Detection/dataset/fer2013.csv`

3. The expected columns are `emotion`, `pixels`, and optionally `Usage`.
   - `emotion`: integer label from 0 to 6.
   - `pixels`: 2,304 space-separated grayscale pixel values for a 48 × 48 image.
   - `Usage`: commonly `Training`, `PublicTest`, or `PrivateTest`.

When `Usage` exists, the code uses those official splits as train, validation, and test. Without `Usage`, it creates a stratified 80/10/10 split.

FER-2013 label mapping:
`0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral`.

## Option B: Folder-based images

Place images in this structure under `dataset/images/`:

```text
images/
  train/
    Angry/
    Disgust/
    Fear/
    Happy/
    Sad/
    Surprise/
    Neutral/
  validation/
    Angry/
    ...
  test/
    Angry/
    ...
```

Supported extensions: JPG, JPEG, PNG and BMP. Folder names must match the emotion names (case-insensitive). Each split must contain images.

Do not commit the dataset to GitHub. The root `.gitignore` excludes dataset contents while retaining this README.
