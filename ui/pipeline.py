"""
Shared Inference Pipeline & State Management for AI Image Detection.
Provides cached model loading, Dual-Branch Spatio-Frequency Fusion inference, and session management.
"""

import os
import sys
import io
import time
import hashlib
from datetime import datetime
from PIL import Image
import numpy as np
import torch
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from inference.predictor import SpatialPredictor, DualBranchPredictor
from preprocessing.frequency_transform import compute_fft_spectrum
from config.config import IMAGE_SIZE
from database.db_manager import init_db, insert_detection, get_all_detections

ACTIVE_ENGINE_NAME = "Dual-Branch Spatio-Frequency Fusion Model"

@st.cache_resource
def load_spatial_model():
    """Loads and caches the Spatial ResNet-50 model used for Grad-CAM explainability."""
    ckpt = os.path.join(BASE_DIR, "models", "checkpoints", "spatial_resnet50_best.pth")
    if os.path.exists(ckpt):
        try:
            return SpatialPredictor(ckpt)
        except Exception as e:
            st.error(f"Error loading spatial checkpoint: {e}")
            raise e
    raise FileNotFoundError(f"Spatial model checkpoint not found at: {ckpt}")

@st.cache_resource
def load_fusion_model():
    """Loads and caches the primary Dual-Branch Spatio-Frequency Fusion model."""
    ckpt = os.path.join(BASE_DIR, "models", "checkpoints", "dual_branch_fusion_best.pth")
    if os.path.exists(ckpt):
        try:
            return DualBranchPredictor(ckpt)
        except Exception as e:
            st.error(f"Error loading fusion checkpoint: {e}")
            raise e
    raise FileNotFoundError(f"Dual-branch fusion checkpoint not found at: {ckpt}")

def get_compute_device():
    """Returns 'cuda' if GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_calibrated_threshold():
    """Returns the calibrated decision threshold from the fusion model."""
    try:
        fusion_pred = load_fusion_model()
        return getattr(fusion_pred, "threshold", 0.55)
    except Exception:
        return 0.55

def get_benchmark_samples():
    """Returns dictionary of available benchmark samples that exist on disk."""
    sample_options = {
        "🌿 Authentic Real (ImageNet-1k)": os.path.join(BASE_DIR, "data", "raw", "REAL", "imagenet", "real_0000.JPEG"),
        "🤖 Synthetic ADM Diffusion": os.path.join(BASE_DIR, "data", "raw", "AI-GENERATED", "adm", "adm_0000.PNG"),
        "🤖 Synthetic Midjourney (Held-Out)": os.path.join(BASE_DIR, "data", "raw", "AI-GENERATED", "midjourney", "midjourney_0000.png"),
    }
    return {k: v for k, v in sample_options.items() if os.path.exists(v)}

def init_session_state():
    """Initializes all required session state variables and verifies database."""
    try:
        init_db()
    except Exception:
        pass

    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"
    if "main_top_nav" not in st.session_state:
        st.session_state.main_top_nav = "Home"
    if "detect_subtab" not in st.session_state:
        st.session_state.detect_subtab = "Single Image Detection"
    if "detect_mode_pills" not in st.session_state:
        st.session_state.detect_mode_pills = "Single Image Detection"
    if "analysis_subtab" not in st.session_state:
        st.session_state.analysis_subtab = "Spatio-Frequency / FFT"
    if "analysis_mode_pills" not in st.session_state:
        st.session_state.analysis_mode_pills = "Spatio-Frequency / FFT"
    if "perf_subtab" not in st.session_state:
        st.session_state.perf_subtab = "Model Performance Dashboard"
    if "perf_mode_pills" not in st.session_state:
        st.session_state.perf_mode_pills = "Model Performance Dashboard"
    if "reports_subtab" not in st.session_state:
        st.session_state.reports_subtab = "Detection History"
    if "reports_mode_pills" not in st.session_state:
        st.session_state.reports_mode_pills = "Detection History"
    if "history" not in st.session_state:
        st.session_state.history = []
    if "active_single_result" not in st.session_state:
        st.session_state.active_single_result = None
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = {}
    if "alpha_val" not in st.session_state:
        st.session_state.alpha_val = 0.5

def navigate_to(page: str, subtab: str = None):
    """
    Safely navigates to a target page and optional subtab,
    updating session state and triggering a clean rerun.
    """
    init_session_state()
    st.session_state.current_page = page

    if subtab:
        if page == "Detect Image":
            st.session_state.detect_subtab = subtab
        elif page == "Analysis":
            st.session_state.analysis_subtab = subtab
        elif page == "Performance":
            st.session_state.perf_subtab = subtab
        elif page == "History & Reports":
            st.session_state.reports_subtab = subtab

    st.rerun()

def predict_fusion_model(pil_image, alpha: float = 0.5):
    """
    Primary Dual-Branch Spatio-Frequency Fusion Inference:
    Evaluates both RGB spatial features (ResNet-50) and 2D Fourier magnitude spectra
    through the gated attention fusion network with calibrated thresholding (tau=0.55).
    """
    fusion_predictor = load_fusion_model()
    label, conf, heatmap, overlay, fft_display = fusion_predictor.predict(pil_image)

    # Re-blend Grad-CAM overlay if user adjusted alpha from default 0.5
    if abs(alpha - 0.5) > 0.01:
        spatial_predictor = load_spatial_model()
        overlay, _ = spatial_predictor.gradcam.overlay(pil_image, heatmap, alpha=alpha)

    return {
        "label": label,
        "confidence": float(conf),
        "heatmap": heatmap,
        "overlay": overlay,
        "fft": fft_display
    }

# Keep alias for backward compatibility
predict_adaptive_ensemble = predict_fusion_model

def analyze_single_image(img_input, filename: str, alpha: float = 0.5, record_history: bool = True):
    """Executes fusion model inference, computes Grad-CAM and FFT, and records session history."""
    try:
        if isinstance(img_input, Image.Image):
            pil_image = img_input.convert("RGB")
        elif isinstance(img_input, str):
            pil_image = Image.open(img_input).convert("RGB")
        else:
            pil_image = Image.open(img_input).convert("RGB")

        t0 = time.time()
        res = predict_fusion_model(pil_image, alpha=alpha)
        latency_ms = (time.time() - t0) * 1000

        tau = get_calibrated_threshold()

        result_data = {
            "id": hashlib.md5((filename + str(time.time())).encode()).hexdigest()[:8],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename,
            "label": res["label"],
            "confidence": float(res["confidence"]),
            "model_name": ACTIVE_ENGINE_NAME,
            "threshold": float(tau),
            "latency_ms": latency_ms,
            "image": pil_image,
            "fft": res["fft"],
            "heatmap": res["heatmap"],
            "overlay": res["overlay"]
        }

        if record_history:
            init_session_state()
            st.session_state.history.insert(0, result_data)
            if len(st.session_state.history) > 100:
                st.session_state.history.pop()

            # Persist to SQLite relational database
            try:
                orig_score = float(res["confidence"] if res["label"] == "REAL" else (1.0 - res["confidence"]))
                ai_score = float(res["confidence"] if res["label"] == "AI-GENERATED" else (1.0 - res["confidence"]))
                db_record_id = insert_detection({
                    "id": result_data["id"],
                    "filename": filename,
                    "file_size_bytes": getattr(img_input, "size", 0) if hasattr(img_input, "size") else 0,
                    "image_width": pil_image.width,
                    "image_height": pil_image.height,
                    "label": res["label"],
                    "confidence": float(res["confidence"]),
                    "threshold": float(tau),
                    "model_name": ACTIVE_ENGINE_NAME,
                    "latency_ms": latency_ms,
                    "heatmap": res.get("heatmap"),
                    "overlay": res.get("overlay"),
                    "fft": res.get("fft"),
                    "analysis_mode": "Single Image",
                    "originality_score": orig_score,
                    "ai_score": ai_score,
                    "execution_device": get_compute_device().upper(),
                    "timestamp": result_data["timestamp"]
                })
                result_data["db_record_id"] = db_record_id
            except Exception as db_err:
                pass

        return result_data

    except Exception as exc:
        st.error(f"Analysis Exception: {exc}")
        return None
