import os
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from explainability.gradcam import GradCAM
from preprocessing.frequency_transform import compute_fft_spectrum, fft_spectrum_to_tensor
from models.dual_branch import DualBranchFusion
from config.config import BASE_DIR, IMAGE_SIZE

class DummyPredictor:
    """
    A dummy classifier for initial testing.
    """
    def __init__(self):
        self.classes = ["REAL", "AI-GENERATED"]
        
    def predict(self, preprocessed_image):
        import random
        prediction_idx = random.randint(0, 1)
        prediction_label = self.classes[prediction_idx]
        confidence = random.uniform(0.5, 1.0)
        if isinstance(preprocessed_image, Image.Image):
            w, h = preprocessed_image.size
        else:
            h, w = preprocessed_image.shape[:2]
        dummy_heatmap = np.random.rand(h, w)
        dummy_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        dummy_fft = np.random.rand(h, w)
        return prediction_label, confidence, dummy_heatmap, dummy_overlay, dummy_fft

class SpatialPredictor:
    """
    Real Spatial ResNet-50 Predictor equipped with Grad-CAM explainability.
    Loads trained weights from models/checkpoints/spatial_resnet50_best.pth.
    """
    def __init__(self, checkpoint_path=None):
        self.classes = ["REAL", "AI-GENERATED"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(BASE_DIR, "models", "checkpoints", "spatial_resnet50_best.pth")
            
        self.model = models.resnet50(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded trained spatial checkpoint from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}, using uninitialized weights")
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Target layer for Grad-CAM: last bottleneck layer in layer4
        self.target_layer = self.model.layer4[-1]
        self.gradcam = GradCAM(self.model, self.target_layer)
        
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image, target_class=None):
        """
        Accepts a PIL Image or np.ndarray, returns:
        - prediction_label (str: 'REAL' or 'AI-GENERATED')
        - confidence (float: 0.0 to 1.0)
        - heatmap (np.ndarray: normalized [0, 1] 2D heatmap)
        - overlay (np.ndarray: RGB overlay on original image)
        - fft_spectrum (np.ndarray: 2D FFT log-magnitude spectrum)
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype('uint8'))
        else:
            pil_image = image
            
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        heatmap, pred_idx, probs = self.gradcam.generate(input_tensor, target_class=target_class)
        
        prediction_label = self.classes[pred_idx]
        confidence = float(probs[pred_idx])
        overlay, colored_heatmap = self.gradcam.overlay(pil_image, heatmap)
        fft_spectrum = compute_fft_spectrum(pil_image, size=IMAGE_SIZE)
        
        return prediction_label, confidence, colored_heatmap, overlay, fft_spectrum

class DualBranchPredictor:
    """
    Dual-Branch Spatio-Frequency Predictor.
    Combines spatial feature patterns (ResNet-50) and FFT frequency anomalies.
    Equipped with Grad-CAM on the spatial branch and calibrated decision thresholding.
    """
    def __init__(self, checkpoint_path=None):
        self.classes = ["REAL", "AI-GENERATED"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = 0.5
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(BASE_DIR, "models", "checkpoints", "dual_branch_fusion_best.pth")
            
        self.model = DualBranchFusion()
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
            self.threshold = checkpoint.get("calibrated_threshold", 0.5)
            print(f"Loaded trained DualBranchFusion checkpoint (calibrated threshold={self.threshold:.2f}) from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        from preprocessing.frequency_transform import compute_multi_band_fft
        self.compute_multi_band_fft = compute_multi_band_fft
        self.spatial_pred = SpatialPredictor()

    def predict(self, image):
        """
        Accepts a PIL Image or np.ndarray, returns:
        - prediction_label (str: 'REAL' or 'AI-GENERATED')
        - confidence (float: 0.0 to 1.0)
        - heatmap (np.ndarray: normalized [0, 1] 2D heatmap from spatial branch)
        - overlay (np.ndarray: RGB overlay on original image)
        - fft_spectrum (np.ndarray: 2D FFT log-magnitude spectrum)
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype('uint8'))
        else:
            pil_image = image

        spatial_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        multi_band = self.compute_multi_band_fft(pil_image, size=IMAGE_SIZE)
        fft_tensor = fft_spectrum_to_tensor(multi_band).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs, alpha = self.model(spatial_tensor, fft_tensor, return_gate=True)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            prob_ai = float(probs[1])

        # Apply calibrated threshold
        pred_idx = 1 if prob_ai >= self.threshold else 0
        prediction_label = self.classes[pred_idx]
        confidence = prob_ai if pred_idx == 1 else (1.0 - prob_ai)

        # Generate spatial heatmap & overlay for the predicted class
        _, _, heatmap, overlay, _ = self.spatial_pred.predict(pil_image, target_class=pred_idx)

        # Single-channel spectrum for display
        fft_display = multi_band[0]

        return prediction_label, confidence, heatmap, overlay, fft_display
