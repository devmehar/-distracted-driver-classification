# Distracted Driver Multi-Action Classification

A deep learning model that classifies driver behavior from dashboard camera images into 10 categories (e.g., safe driving, texting, talking on phone, drinking, reaching behind), aimed at powering in-vehicle driver-distraction alert systems.

## Problem Statement

Distracted driving is a leading cause of road accidents. Automatically detecting *what* a driver is doing from a dashboard camera feed — rather than just whether they're looking at the road — enables real-time alerts and supports insurance/fleet-safety use cases. This project builds an image classifier that recognizes 10 distinct driver actions from a single frame.

## Demo

*(Add a screenshot here of a sample prediction — e.g., an input image next to the model's predicted class and confidence score. A GIF cycling through a few predictions works great too.)*

```
📷 [input image] → Predicted: "texting - right"  (confidence: 0.9X)
```

## Dataset

- Source: State Farm Distracted Driver Detection dataset (Kaggle) — *link the exact dataset you used*
- 10 classes: c0 (safe driving) through c9 (talking to passenger)
- Images resized to a fixed input size and normalized before training
- Split: 60% train / 40% test (see notebook for exact split logic)

## Approach

1. **Preprocessing:** Loaded images per class folder using OpenCV, resized to a uniform shape, and shuffled to remove class-order bias.
2. **Modeling:** Built a CNN with 3 convolutional blocks (Conv2D + BatchNormalization + MaxPooling + Dropout) followed by dense layers, using softmax output over 10 classes.
3. **Training:** Compiled with categorical cross-entropy and Adam optimizer. Used `ModelCheckpoint`, `EarlyStopping`, and `ReduceLROnPlateau` callbacks to avoid overfitting and wasted compute.
4. **Evaluation:** Assessed with accuracy/loss curves (train vs. validation), a confusion matrix across all 10 classes, and a full classification report (precision/recall/F1 per class).

## Results

| Metric | Value |
|---|---|
| Validation Accuracy | *add your result, e.g. 96.X%* |
| Macro F1-score | *add your result* |
| Notes | See `confusion matrix` and `classification report` cells in the notebook for per-class breakdown |

*(Tip: mention how this compares to a naive baseline — e.g., random guessing across 10 classes would be ~10% accuracy — to give the number context.)*

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)
![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

## How to Run

```bash
git clone https://github.com/<your-username>/distracted-driver-classification.git
cd distracted-driver-classification
pip install -r requirements.txt
```

1. Download the dataset (link above) and place it in a `data/` folder in the structure the notebook expects.
2. Open `DL_P1_Distracted_driver_multiaction_classification.ipynb` in Jupyter or Google Colab.
3. Run all cells in order — training will take a while on CPU; a GPU runtime (e.g., Colab GPU) is recommended.

## Project Structure

```
├── DL_P1_Distracted_driver_multiaction_classification.ipynb
├── requirements.txt
├── README.md
└── data/                  # not included — see Dataset section
```

## Future Improvements

- Add data augmentation (rotation, brightness, occlusion) to improve robustness to real-world camera angles/lighting
- Try transfer learning (e.g., MobileNetV2, ResNet50) for higher accuracy with less training data
- Convert the model to TensorFlow Lite for real-time, on-device inference in a vehicle
- Deploy a live demo (Streamlit/Gradio) where a user can upload an image and see the predicted class

## License

This project is licensed under the MIT License.
