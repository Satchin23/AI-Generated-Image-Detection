# AI-Generated Image Detection Using Spatio-Frequency Features

## About
This project detects whether an uploaded image is real or AI-generated using spatial and frequency-domain features.

It combines ResNet50, FFT-based frequency analysis and Grad-CAM to provide predictions along with visual explanations.

## Key Features
- Real vs AI-generated image classification
- Spatial feature extraction using ResNet50
- Frequency analysis using FFT
- Dual-branch feature fusion
- Grad-CAM heatmap visualization
- Image segment analysis
- Model performance evaluation

## Technologies Used
Python, PyTorch, Streamlit, OpenCV, NumPy and Matplotlib.

## Installation
1. Clone this repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Download the required datasets and model checkpoints.
4. Place them in their respective project folders.
5. Run the application using `streamlit run app.py`.

## Dataset and Model Files
Large datasets and trained model checkpoints are maintained separately due to their storage requirements. Download instructions will be provided separately.

## Objective
To improve the detection of AI-generated images through spatial and frequency-based analysis while providing visual explanations of model predictions.
