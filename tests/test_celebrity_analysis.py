import os
import sys
import numpy as np
from PIL import Image

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from views.celebrity_analysis import _real_class_score
from preprocessing.semantic_segmentation import segment_portrait_regions


def test_real_class_score_uses_real_probability_for_either_prediction():
    assert _real_class_score({"label": "REAL", "confidence": 0.8}) == 0.8
    assert _real_class_score({"label": "AI-GENERATED", "confidence": 0.8}) == 0.2


def test_semantic_segmentation_extracts_regions():
    # Create a synthetic portrait-sized image
    test_img = Image.new("RGB", (200, 300), color=(180, 150, 120))
    res = segment_portrait_regions(test_img)
    
    assert "regions" in res
    assert "overlay" in res
    assert len(res["regions"]) > 0
    assert res["total_pixels"] == 200 * 300
    for r in res["regions"]:
        assert r["pixel_count"] > 0
        assert r["crop"] is not None
        assert r["name"] is not None
