import os
import sys
import pytest
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predictor import DummyPredictor, SpatialPredictor, DualBranchPredictor
from preprocessing.image_preprocessing import preprocess_image_dummy

def test_preprocessing():
    img = Image.new('RGB', (500, 500), color='red')
    processed = preprocess_image_dummy(img)
    assert processed.shape == (224, 224, 3)
    assert np.max(processed) <= 1.0
    assert np.min(processed) >= 0.0

def test_dummy_predictor():
    predictor = DummyPredictor()
    dummy_input = np.zeros((224, 224, 3))
    label, conf, heatmap, overlay, fft = predictor.predict(dummy_input)
    assert label in ["REAL", "AI-GENERATED"]
    assert 0.5 <= conf <= 1.0
    assert heatmap.shape == (224, 224)

def test_spatial_predictor():
    predictor = SpatialPredictor()
    test_img = Image.new('RGB', (224, 224), color='green')
    label, conf, heatmap, overlay, fft = predictor.predict(test_img)
    assert label in ["REAL", "AI-GENERATED"]
    assert 0.0 <= conf <= 1.0
    assert heatmap.shape[:2] == (224, 224)
    assert overlay.shape == (224, 224, 3)
    assert fft.shape == (224, 224)

def test_dual_branch_predictor():
    predictor = DualBranchPredictor()
    test_img = Image.new('RGB', (224, 224), color='yellow')
    label, conf, heatmap, overlay, fft = predictor.predict(test_img)
    assert label in ["REAL", "AI-GENERATED"]
    assert 0.0 <= conf <= 1.0
    assert heatmap.shape[:2] == (224, 224)
    assert fft.shape == (224, 224)
