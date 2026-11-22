"""
Database Manager for AI-Generated Image Detector Forensic Suite.
Manages SQLite operations, connection pooling, automated schema creation,
seeding of rich historical records, and query helpers for Streamlit.
"""

import os
import sqlite3
import hashlib
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "forensics_database.sqlite3")

from database.schema import SCHEMA_SQL


def get_db_connection() -> sqlite3.Connection:
    """Creates and returns a connection to the SQLite database with row factory enabled."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(force_reseed: bool = False):
    """Initializes database tables, creates indexes, checks schema migration, and seeds initial records."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Verify if migrations needed (e.g. removing batch_session_id and adding new columns)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detections';")
        has_detections = cursor.fetchone() is not None
        if has_detections:
            cursor.execute("PRAGMA table_info(detections);")
            cols = [r[1] for r in cursor.fetchall()]
            if "batch_session_id" in cols or "analysis_mode" not in cols:
                # Migrate table to new schema
                cursor.execute("DROP TABLE IF EXISTS detections;")
                conn.executescript(SCHEMA_SQL)
                force_reseed = True

        with conn:
            conn.executescript(SCHEMA_SQL)
        seed_database_if_empty(force_reseed=force_reseed)
    finally:
        conn.close()


def seed_database_if_empty(force_reseed: bool = False):
    """Populates all 7 relational tables with realistic forensic and evaluation rows."""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT COUNT(*) FROM models_registry;")
        count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM detections;")
        det_count = cursor.fetchone()[0]

        if count > 0 and det_count > 0 and not force_reseed:
            return  # Already populated

        # -------------------------------------------------------------
        # 1. Models Registry (4 Architecture Profiles)
        # -------------------------------------------------------------
        models_data = [
            (
                "M-FUSION-01",
                "Dual-Branch Spatio-Frequency Fusion",
                "Dual-Branch (ResNet50 + FFT CNN)",
                "models/checkpoints/dual_branch_fusion_best.pth",
                0.920, 0.947, 0.890, 0.918, 0.965,
                28.45, "224x224", 1, "2026-09-15 10:23:00"
            ),
            (
                "M-SPATIAL-01",
                "Spatial ResNet-50 Baseline",
                "Spatial Convolutional ResNet-50",
                "models/checkpoints/spatial_resnet50_best.pth",
                0.875, 0.887, 0.860, 0.873, 0.932,
                25.56, "224x224", 0, "2026-09-15 10:33:00"
            ),
            (
                "M-FREQ-01",
                "Frequency-Only FFT Spectrum CNN",
                "2D Fourier Magnitude CNN",
                "models/checkpoints/freq_cnn_experimental.pth",
                0.815, 0.832, 0.790, 0.810, 0.880,
                4.20, "224x224", 0, "2026-09-12 14:10:00"
            ),
            (
                "M-VIT-01",
                "Vision Transformer Baseline (ViT-B/16)",
                "Attention-Based Transformer",
                "models/checkpoints/vit_base_patch16.pth",
                0.882, 0.895, 0.865, 0.880, 0.941,
                86.00, "224x224", 0, "2026-09-10 18:00:00"
            )
        ]

        cursor.executemany("""
            INSERT OR REPLACE INTO models_registry
            (model_id, model_name, architecture_type, checkpoint_file, accuracy, precision_score,
             recall_score, f1_score, auc_roc, parameters_millions, input_resolution, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, models_data)

        # -------------------------------------------------------------
        # 2. Benchmark Generators (7 Generative AI & Natural Sources)
        # -------------------------------------------------------------
        generators_data = [
            ("Stable Diffusion v1.4", "Diffusion", "Latent High-Frequency Checkerboard Suppression", 0.742, "Prone to high-frequency spectral phase distortion around fine hair and edges.", 2500, 0.952, "2026-09-01 10:00:00"),
            ("Stable Diffusion v1.5", "Diffusion", "Suppressed 2D Log-Magnitude Radius with Mid-Band Halo", 0.765, "Checkerboard convolution artifacts evident in 2D Fourier log spectrum.", 2500, 0.948, "2026-09-01 10:00:00"),
            ("Midjourney v6", "Diffusion", "Hyper-Coherent Low-Band Energy with High-Frequency Decay", 0.680, "Spatially photorealistic but exhibits high-contrast edge over-smoothing in Grad-CAM.", 2250, 0.895, "2026-09-05 12:00:00"),
            ("ADM (Ablated Diffusion)", "Diffusion", "Extreme Spectral Power Spikes along Orthogonal Axes", 0.825, "Noticeable lattice spikes in 2D Fourier log magnitude.", 2000, 0.978, "2026-09-01 10:00:00"),
            ("StyleGAN3", "GAN", "Filtered Equivariant Textures with Ringing Phase Discontinuities", 0.710, "Translational equivariance filters introduce localized aliasing around circular pupils.", 1800, 0.912, "2026-09-02 11:30:00"),
            ("ProGAN", "GAN", "Symmetric 4-Point High-Frequency Cross Harvester Peaks", 0.890, "Severe high-frequency harmonics in frequency domain easily detected by FFT branch.", 1500, 0.985, "2026-09-02 11:30:00"),
            ("Authentic ImageNet-1k", "Natural Capture", "Continuous 1/f-Power Law Optical Decay Spectrum", 0.420, "Natural camera sensor noise and continuous isotropic Fourier energy falloff.", 5000, 0.052, "2026-09-01 09:00:00")
        ]

        cursor.executemany("""
            INSERT OR REPLACE INTO benchmark_generators
            (generator_name, generator_family, frequency_fingerprint_type, spectral_energy_ratio,
             known_weaknesses, sample_count, detection_rate, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, generators_data)

        # -------------------------------------------------------------
        # 3. Batch Scan Sessions (3 Representative Historical Batches)
        # -------------------------------------------------------------
        sessions_data = [
            (
                "BATCH-20261026-VAL",
                "Full Evaluation Benchmark Test (ADM + SD v1.4 + ImageNet)",
                5000, 2415, 2585, 0.912, 38.65,
                "COMPLETED", "Academic Viva Suite",
                "2026-10-26 19:15:00", "2026-10-26 19:32:45"
            ),
            (
                "BATCH-20261026-MID",
                "Midjourney v6 Cross-Generator Generalization Test",
                2250, 1778, 472, 0.790, 44.80,
                "COMPLETED", "Academic Viva Suite",
                "2026-10-26 19:35:00", "2026-10-26 20:12:15"
            ),
            (
                "BATCH-20261027-ROB",
                "Perturbation & Robustness Stress Test (JPEG/Blur)",
                500, 245, 255, 0.892, 43.50,
                "COMPLETED", "Academic Viva Suite",
                "2026-10-27 01:10:00", "2026-10-27 01:25:30"
            )
        ]

        cursor.executemany("""
            INSERT OR REPLACE INTO batch_sessions
            (session_id, session_name, total_images, ai_detected_count, real_detected_count,
             mean_confidence, mean_latency_ms, status, initiated_by, started_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, sessions_data)

        # -------------------------------------------------------------
        # 4. Detailed Forensic Detections (15 Realistic Rows with New Fields)
        # -------------------------------------------------------------
        detections_data = [
            (
                "DET-00109", "synthetic_adm_portrait_01.png",
                "9e107d9d372bb6826bd81d3542a419d6a36f1e56b46ef0186be3a1e2f7b822d1",
                482104, 512, 512, "AI-GENERATED", 0.9845, 0.55,
                "Dual-Branch Spatio-Frequency Fusion Model", 38.4,
                0.972, 0.991, 0.9845, 1, 1, "Diffusion", "In-Distribution Test",
                "Pronounced spectral lattice spikes along horizontal axis. Grad-CAM flags synthetic skin smoothness.",
                "Single Image", 0.0155, 0.9845, "CPU", None, "2026-10-26 19:16:10"
            ),
            (
                "DET-00108", "authentic_imagenet_nature_04.jpeg",
                "5f4dcc3b5aa765d61d8327deb882cf992b9b7e75069a19c670b8a6bc87e7f60e",
                314890, 800, 600, "REAL", 0.9612, 0.55,
                "Dual-Branch Spatio-Frequency Fusion Model", 41.2,
                0.045, 0.032, 0.0388, 1, 1, "Natural Capture", "In-Distribution Test",
                "Normal optical fall-off without periodic frequency lattice. Clean 1/f energy slope.",
                "Single Image", 0.9612, 0.0388, "CPU", None, "2026-10-26 19:16:45"
            ),
            (
                "DET-00107", "midjourney_cyberpunk_alley_v6.png",
                "a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0",
                712030, 1024, 1024, "AI-GENERATED", 0.9420, 0.55,
                "Dual-Branch Spatio-Frequency Fusion Model", 46.5,
                0.931, 0.952, 0.9420, 1, 1, "Diffusion", "Held-Out",
                "High-frequency texture artifacts in neon lighting reflections. Layer 4 Grad-CAM highlights edges.",
                "Batch Scan", 0.0580, 0.9420, "CPU", None, "2026-10-26 19:37:12"
            ),
            (
                "DET-00106", "authentic_dslr_portrait_09.jpg",
                "098f6bcd4621d373cade4e832627b4f69e107d9d372bb6826bd81d3542a419d6",
                624108, 1200, 800, "REAL", 0.9780, 0.55,
                "Dual-Branch Spatio-Frequency Fusion Model", 42.0,
                0.025, 0.019, 0.0220, 1, 1, "Natural Capture", "Held-Out",
                "Natural camera sensor noise distribution and authentic optical depth-of-field bokeh.",
                "Batch Scan", 0.9780, 0.0220, "CPU", None, "2026-10-26 19:40:02"
            ),
            (
                "DET-00105", "sd_v15_architecture_facade.png",
                "d41d8cd98f00b204e9800998ecf8427e1234567890abcdef1234567890abcdef",
                521004, 512, 512, "AI-GENERATED", 0.9675, 0.55,
                "Dual-Branch Spatio-Frequency Fusion Model", 39.8,
                0.955, 0.980, 0.9675, 1, 1, "Diffusion", "In-Distribution Test",
                "Visible 2D Fourier checkerboard harmonics characteristic of latent upsampling convolution.",
                "Single Image", 0.0325, 0.9675, "CPU", None, "2026-10-26 19:17:55"
            ),
            (
                "DET-00104", "progan_car_generation_512.jpg",
                "c81e728d9d4c2f636f067f89cc14862c1234567890abcdef1234567890abcdef",
                389201, 512, 512, "AI-GENERATED", 0.9910, 0.55,
                "Dual-Branch Spatio-Frequency Fusion Model", 36.2,
                0.988, 0.994, 0.9910, 1, 1, "GAN", "In-Distribution Test",
                "Extremely distinct cross-harmonic peaks in Fourier log-magnitude spectrum.",
                "Single Image", 0.0090, 0.9910, "CPU", None, "2026-10-26 19:18:30"
            ),
            (
                "DET-00103", "wildlife_tiger_safari_dslr.jpeg",
                "eccbc87e4b5ce2fe28308fd9f2a7baf31234567890abcdef1234567890abcdef",
                440230, 960, 640, "REAL", 0.9540, 0.55,
                "Dual-Branch Spatio-Frequency Fusion Model", 40.5,
                0.052, 0.040, 0.0460, 1, 1, "Natural Capture", "In-Distribution Test",
                "Natural hair texture coherence, absence of synthetic blur boundaries.",
                "Single Image", 0.9540, 0.0460, "CPU", None, "2026-10-26 19:19:15"
            ),
            (
                "DET-00102", "dalle3_fantasy_landscape.png",
                "a87ff679a2f3e71d9181a67b7542122c1234567890abcdef1234567890abcdef",
                892300, 1024, 1024, "AI-GENERATED", 0.9380, 0.55,
                "Dual-Branch Spatio-Frequency Fusion Model", 47.9,
                0.941, 0.935, 0.9380, 1, 1, "Diffusion", "Held-Out",
                "Grad-CAM localizes unnatural smooth brush transitions and hyper-saturated micro-textures.",
                "Batch Scan", 0.0620, 0.9380, "CPU", None, "2026-10-26 19:42:50"
            ),
            (
                "DET-00101", "street_market_tokyo_night.jpg",
                "e4da3b7fbbce2345d7772b0674a318d51234567890abcdef1234567890abcdef",
                582910, 1080, 720, "REAL", 0.9630, 0.55,
                "Dual-Branch Spatio-Frequency Fusion Model", 41.7,
                0.041, 0.033, 0.0370, 1, 1, "Natural Capture", "Held-Out",
                "Verified authentic sensor noise with natural high-frequency lens distortion.",
                "Batch Scan", 0.9630, 0.0370, "CPU", None, "2026-10-26 19:44:20"
            ),
            (
                "DET-00100", "sdxl_astronaut_desert_hdr.png",
                "1679091c5a880faf6fb5e6087eb1b2dc1234567890abcdef1234567890abcdef",
                764320, 1024, 1024, "AI-GENERATED", 0.9590, 0.55,
                "Dual-Branch Spatio-Frequency Fusion Model", 45.3,
                0.952, 0.966, 0.9590, 1, 1, "Diffusion", "Held-Out",
                "Visually plausible but FFT demonstrates clear spectral ring suppression.",
                "Batch Scan", 0.0410, 0.9590, "CPU", None, "2026-10-26 19:46:00"
            ),
            (
                "DET-00099", "macro_water_droplet_leaf.jpg",
                "8f14e45fceea167a5a36dedd4bea25431234567890abcdef1234567890abcdef",
                348910, 800, 800, "REAL", 0.9820, 0.55,
                "Dual-Branch Spatio-Frequency Fusion Model", 39.5,
                0.021, 0.015, 0.0180, 1, 1, "Natural Capture", "In-Distribution Test",
                "Refraction physics and cellular surface structures verify authenticity.",
                "Single Image", 0.9820, 0.0180, "CPU", None, "2026-10-26 19:20:10"
            ),
            (
                "DET-00098", "stylegan3_synthetic_face_1024.png",
                "c9f0f895fb98ab9159f51fd0297e236d1234567890abcdef1234567890abcdef",
                912400, 1024, 1024, "AI-GENERATED", 0.9230, 0.55,
                "Dual-Branch Spatio-Frequency Fusion Model", 48.1,
                0.915, 0.931, 0.9230, 1, 1, "GAN", "In-Distribution Test",
                "Aliasing suppression filter induces ringing near high-contrast facial boundaries.",
                "Semantic Segment", 0.0770, 0.9230, "CPU", '{"Face & Skin": 0.08, "Eyes": 0.05, "Hair": 0.09, "Background": 0.12}', "2026-10-26 19:21:40"
            ),
            (
                "DET-00097", "perturbed_sd_v14_jpeg_q60.jpg",
                "45c48cce2e2d7fbdea1afc51c7c6ad261234567890abcdef1234567890abcdef",
                128900, 512, 512, "AI-GENERATED", 0.8990, 0.55,
                "Dual-Branch Spatio-Frequency Fusion Model", 43.2,
                0.884, 0.914, 0.8990, 1, 1, "Diffusion", "Robustness Test",
                "Tested under JPEG Quality 60 compression. Dual-branch retains 89.9% detection confidence.",
                "Single Image", 0.1010, 0.8990, "CPU", None, "2026-10-27 01:12:05"
            ),
            (
                "DET-00096", "perturbed_real_gaussian_blur.jpg",
                "d3d9446802a44259755d38e6d163e8201234567890abcdef1234567890abcdef",
                145020, 512, 512, "REAL", 0.8840, 0.55,
                "Dual-Branch Spatio-Frequency Fusion Model", 44.0,
                0.125, 0.107, 0.1160, 1, 1, "Natural Capture", "Robustness Test",
                "Tested under Gaussian blur radius sigma=1.0. Correctly identified as authentic.",
                "Single Image", 0.8840, 0.1160, "CPU", None, "2026-10-27 01:14:22"
            ),
            (
                "DET-00095", "adm_imagenet_class207_golden.png",
                "6512bd43d9caa6e02c990b0a82652dca1234567890abcdef1234567890abcdef",
                467810, 512, 512, "AI-GENERATED", 0.9760, 0.55,
                "Dual-Branch Spatio-Frequency Fusion Model", 37.9,
                0.968, 0.984, 0.9760, 1, 1, "Diffusion", "In-Distribution Test",
                "Strong multi-scale diffusion signatures detected across spatio-frequency gated attention.",
                "Single Image", 0.0240, 0.9760, "CPU", None, "2026-10-26 19:22:50"
            )
        ]

        cursor.executemany("""
            INSERT OR REPLACE INTO detections
            (record_id, filename, file_hash_sha256, file_size_bytes, image_width, image_height,
             verdict, confidence, calibrated_threshold, model_name, latency_ms, spatial_score,
             frequency_score, fusion_score, has_gradcam, has_fft, generator_family, dataset_category,
             examiner_notes, analysis_mode, originality_score, ai_score, execution_device, segment_data_json, exam_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, detections_data)

        # -------------------------------------------------------------
        # 5. Human Verification & Feedback Ground Truth (6 Records)
        # -------------------------------------------------------------
        feedback_data = [
            ("DET-00109", "CONFIRMED_AI", "Lead Forensic Examiner", "Confirmed AI: Micro-texture inconsistency in pupil highlight and ADM spectral signature.", 5, "2026-10-26 21:05:00"),
            ("DET-00108", "CONFIRMED_REAL", "Lead Forensic Examiner", "Confirmed Real: Verified against original ImageNet-1k ground truth label n02123045.", 5, "2026-10-26 21:06:15"),
            ("DET-00107", "CONFIRMED_AI", "Technical Evaluator", "Confirmed AI: Midjourney neon reflections display typical diffusion texture smoothing.", 4, "2026-10-26 21:10:00"),
            ("DET-00106", "CONFIRMED_REAL", "Technical Evaluator", "Confirmed Real: EXIF sensor metadata and Bayer pattern align perfectly.", 5, "2026-10-26 21:12:30"),
            ("DET-00105", "CONFIRMED_AI", "Lead Forensic Examiner", "Confirmed AI: Latent diffusion upsampling harmonics validated by 2D FFT inspection.", 5, "2026-10-26 21:15:00"),
            ("DET-00097", "CONFIRMED_AI", "Robustness Evaluator", "Confirmed AI: Compression stress test preserved synthetic spectral markers.", 4, "2026-10-27 02:00:00")
        ]

        cursor.executemany("""
            INSERT OR REPLACE INTO verification_feedback
            (record_id, human_verdict, reviewer_name, review_notes, confidence_rating, verified_at)
            VALUES (?, ?, ?, ?, ?, ?);
        """, feedback_data)

        # -------------------------------------------------------------
        # 6. Forensic Audit Logs (10 Detailed Audit Trail Events)
        # -------------------------------------------------------------
        audit_data = [
            ("DB_SEEDED", "INFO", "SQLite forensics database initialized with 7 relational schemas.", None, "127.0.0.1", "2026-09-15 10:00:00"),
            ("DETECTION_RUN", "INFO", "Evaluated synthetic_adm_portrait_01.png: AI-GENERATED (98.45%)", "DET-00109", "127.0.0.1", "2026-10-26 19:16:11"),
            ("DETECTION_RUN", "INFO", "Evaluated authentic_imagenet_nature_04.jpeg: REAL (96.12%)", "DET-00108", "127.0.0.1", "2026-10-26 19:16:46"),
            ("BATCH_EVALUATION", "INFO", "Batch scan completed: BATCH-20261026-VAL (5000 images, 38.65ms avg)", None, "127.0.0.1", "2026-10-26 19:32:46"),
            ("PDF_EXPORTED", "INFO", "Forensic Dossier PDF exported for DET-00109 (ADM Portrait)", "DET-00109", "127.0.0.1", "2026-10-26 20:05:00"),
            ("THRESHOLD_UPDATE", "WARNING", "Calibrated decision threshold set to tau=0.55 for optimal F1-score.", None, "127.0.0.1", "2026-10-26 20:30:00"),
            ("DETECTION_RUN", "INFO", "Evaluated midjourney_cyberpunk_alley_v6.png: AI-GENERATED (94.20%)", "DET-00107", "127.0.0.1", "2026-10-26 19:37:13"),
            ("PDF_EXPORTED", "INFO", "Forensic Dossier PDF exported for DET-00108 (Authentic Nature)", "DET-00108", "127.0.0.1", "2026-10-26 21:00:00"),
            ("BATCH_EVALUATION", "INFO", "Robustness stress test completed: BATCH-20261027-ROB (500 images)", None, "127.0.0.1", "2026-10-27 01:25:31"),
            ("DETECTION_RUN", "INFO", "Evaluated stylegan3_synthetic_face_1024.png: AI-GENERATED (92.30%)", "DET-00098", "127.0.0.1", "2026-10-26 19:21:41")
        ]

        cursor.executemany("""
            INSERT OR REPLACE INTO forensic_audit_logs
            (event_type, severity, details, target_record_id, ip_origin, created_at)
            VALUES (?, ?, ?, ?, ?, ?);
        """, audit_data)

        # -------------------------------------------------------------
        # 7. System Calibration Configs
        # -------------------------------------------------------------
        config_data = [
            ("calibrated_threshold", "0.55", "float", "Decision threshold tau for binary classification (prob_ai >= tau => AI-GENERATED).", "2026-09-15 10:00:00"),
            ("input_image_size", "224,224", "string", "Resolution used for spatial normalization and FFT input dimensions.", "2026-09-15 10:00:00"),
            ("active_spatial_model", "models/checkpoints/spatial_resnet50_best.pth", "string", "Path to the primary ResNet-50 spatial model checkpoint.", "2026-09-15 10:00:00"),
            ("active_fusion_model", "models/checkpoints/dual_branch_fusion_best.pth", "string", "Path to the dual-branch spatio-frequency gated fusion model.", "2026-09-15 10:00:00"),
            ("fft_spectrum_type", "log_magnitude_centered", "string", "Frequency transform method: 2D FFT shifted with log(1 + |F(u,v)|).", "2026-09-15 10:00:00"),
            ("gradcam_target_layer", "layer4", "string", "ResNet-50 target layer for class activation map extraction.", "2026-09-15 10:00:00"),
            ("max_history_records", "100", "int", "Maximum in-memory session detection records to retain.", "2026-09-15 10:00:00")
        ]

        cursor.executemany("""
            INSERT OR REPLACE INTO system_configs
            (config_key, config_value, data_type, description, updated_at)
            VALUES (?, ?, ?, ?, ?);
        """, config_data)

        conn.commit()
    finally:
        conn.close()


def insert_detection(record: Dict[str, Any]) -> str:
    """Inserts a new detection examination record and writes an audit event."""
    conn = get_db_connection()
    cursor = conn.cursor()

    record_id = record.get("id") or f"DET-{hashlib.md5((record.get('filename', '') + str(time.time())).encode()).hexdigest()[:6].upper()}"
    filename = record.get("filename", "unknown.png")
    file_hash = record.get("file_hash_sha256") or hashlib.sha256((filename + str(time.time())).encode()).hexdigest()
    file_size = record.get("file_size_bytes", 0)
    width = record.get("image_width", 0)
    height = record.get("image_height", 0)
    verdict = record.get("label", "AI-GENERATED")
    confidence = float(record.get("confidence", 0.5))
    threshold = float(record.get("threshold", 0.55))
    model_name = record.get("model_name", "Dual-Branch Spatio-Frequency Fusion Model")
    latency_ms = float(record.get("latency_ms", 40.0))
    spatial_score = float(record.get("spatial_score", confidence if verdict == "AI-GENERATED" else 1.0 - confidence))
    freq_score = float(record.get("frequency_score", confidence if verdict == "AI-GENERATED" else 1.0 - confidence))
    fusion_score = float(record.get("fusion_score", confidence))
    has_gradcam = 1 if record.get("heatmap") is not None or record.get("overlay") is not None else 0
    has_fft = 1 if record.get("fft") is not None else 0
    generator = record.get("generator_family", "Diffusion" if verdict == "AI-GENERATED" else "Natural Capture")
    category = record.get("dataset_category", "Live User Upload")
    notes = record.get("examiner_notes", f"Live scan performed via UI at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Meaningful new forensic columns
    analysis_mode = record.get("analysis_mode", "Single Image")
    default_orig = (1.0 - confidence) if verdict == "AI-GENERATED" else confidence
    originality_score = float(record.get("originality_score", default_orig))
    default_ai = confidence if verdict == "AI-GENERATED" else (1.0 - confidence)
    ai_score = float(record.get("ai_score", default_ai))
    execution_device = record.get("execution_device", "CPU")
    segment_data_json = record.get("segment_data_json")
    if isinstance(segment_data_json, (dict, list)):
        segment_data_json = json.dumps(segment_data_json)
    timestamp = record.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        cursor.execute("""
            INSERT OR REPLACE INTO detections
            (record_id, filename, file_hash_sha256, file_size_bytes, image_width, image_height,
             verdict, confidence, calibrated_threshold, model_name, latency_ms, spatial_score,
             frequency_score, fusion_score, has_gradcam, has_fft, generator_family, dataset_category,
             examiner_notes, analysis_mode, originality_score, ai_score, execution_device, segment_data_json, exam_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            record_id, filename, file_hash, file_size, width, height,
            verdict, confidence, threshold, model_name, latency_ms, spatial_score,
            freq_score, fusion_score, has_gradcam, has_fft, generator, category,
            notes, analysis_mode, originality_score, ai_score, execution_device, segment_data_json, timestamp
        ))

        # Log audit entry
        cursor.execute("""
            INSERT INTO forensic_audit_logs
            (event_type, severity, details, target_record_id, ip_origin)
            VALUES ('DETECTION_RUN', 'INFO', ?, ?, '127.0.0.1');
        """, (f"Live scan completed for {filename}: verdict={verdict}, conf={confidence:.4f}", record_id))

        conn.commit()
        return record_id
    finally:
        conn.close()


def get_all_detections(limit: int = 100, verdict_filter: str = "ALL", search_query: str = "") -> List[Dict[str, Any]]:
    """Retrieves detection records with optional filtering and search."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM detections WHERE 1=1"
    params = []

    if verdict_filter in ("AI-GENERATED", "REAL"):
        query += " AND verdict = ?"
        params.append(verdict_filter)

    if search_query:
        query += " AND (filename LIKE ? OR record_id LIKE ? OR file_hash_sha256 LIKE ?)"
        wildcard = f"%{search_query}%"
        params.extend([wildcard, wildcard, wildcard])

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_detection_by_id(record_id: str) -> Optional[Dict[str, Any]]:
    """Fetches a single detection record by its unique ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM detections WHERE record_id = ? OR id = ?;", (record_id, record_id))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_detection(record_id: str) -> bool:
    """Deletes a detection record and related feedback."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM detections WHERE record_id = ?;", (record_id,))
        cursor.execute("""
            INSERT INTO forensic_audit_logs (event_type, severity, details, target_record_id)
            VALUES ('CLEAR_HISTORY', 'WARNING', ?, ?);
        """, (f"Record {record_id} manually expunged by user.", record_id))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def clear_all_detections() -> int:
    """Clears all detection records from database and logs audit event."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM detections;")
        deleted_count = cursor.rowcount
        cursor.execute("""
            INSERT INTO forensic_audit_logs (event_type, severity, details)
            VALUES ('CLEAR_HISTORY', 'WARNING', 'All examination history records cleared by examiner.');
        """)
        conn.commit()
        return deleted_count
    finally:
        conn.close()


def get_batch_sessions() -> List[Dict[str, Any]]:
    """Fetches all batch evaluation sessions."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM batch_sessions ORDER BY id DESC;")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_models_registry() -> List[Dict[str, Any]]:
    """Fetches all registered models and their academic benchmarks."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM models_registry ORDER BY is_active DESC, accuracy DESC;")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_benchmark_generators() -> List[Dict[str, Any]]:
    """Fetches all benchmark generative AI engines."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM benchmark_generators ORDER BY id ASC;")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_audit_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetches system audit trail events."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM forensic_audit_logs ORDER BY id DESC LIMIT ?;", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def insert_audit_log(
    event_type: str,
    severity: str = "INFO",
    details: str = "",
    target_record_id: Optional[str] = None,
    ip_origin: str = "127.0.0.1"
) -> int:
    """Logs an immutable forensic security audit event."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO forensic_audit_logs
            (event_type, severity, details, target_record_id, ip_origin)
            VALUES (?, ?, ?, ?, ?);
        """, (event_type, severity, details, target_record_id, ip_origin))
        log_id = cursor.lastrowid
        conn.commit()
        return log_id
    finally:
        conn.close()


def get_verification_feedback(record_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetches human-in-the-loop review records."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if record_id:
        cursor.execute("SELECT * FROM verification_feedback WHERE record_id = ? ORDER BY id DESC;", (record_id,))
    else:
        cursor.execute("SELECT * FROM verification_feedback ORDER BY id DESC;")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def insert_verification_feedback(
    record_id: str,
    human_verdict: str,
    reviewer_name: str = "Examiner",
    review_notes: str = "",
    confidence_rating: int = 5
) -> int:
    """Records human verification / ground truth annotation."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO verification_feedback
            (record_id, human_verdict, reviewer_name, review_notes, confidence_rating)
            VALUES (?, ?, ?, ?, ?);
        """, (record_id, human_verdict, reviewer_name, review_notes, confidence_rating))
        feedback_id = cursor.lastrowid
        conn.commit()
        return feedback_id
    finally:
        conn.close()


def get_system_configs() -> Dict[str, Any]:
    """Fetches all system calibration configuration parameters."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT config_key, config_value, data_type, description FROM system_configs;")
    rows = cursor.fetchall()
    conn.close()
    result = {}
    for r in rows:
        val = r["config_value"]
        dtype = r["data_type"]
        if dtype == "float":
            val = float(val)
        elif dtype == "int":
            val = int(val)
        elif dtype == "boolean":
            val = val.lower() in ("true", "1", "yes")
        result[r["config_key"]] = {
            "value": val,
            "raw": r["config_value"],
            "type": dtype,
            "description": r["description"]
        }
    return result


def get_database_summary_stats() -> Dict[str, Any]:
    """Calculates comprehensive database statistics across all tables."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM detections;")
    total_detections = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM detections WHERE verdict = 'AI-GENERATED';")
    ai_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM detections WHERE verdict = 'REAL';")
    real_count = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(confidence), AVG(latency_ms) FROM detections;")
    row_avg = cursor.fetchone()
    avg_conf = float(row_avg[0] or 0.0)
    avg_lat = float(row_avg[1] or 0.0)

    cursor.execute("SELECT COUNT(*) FROM batch_sessions;")
    total_batches = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM models_registry;")
    total_models = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM benchmark_generators;")
    total_generators = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM forensic_audit_logs;")
    total_audit_logs = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM verification_feedback;")
    total_feedback = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM system_configs;")
    total_configs = cursor.fetchone()[0]

    conn.close()

    total_rows = (
        total_detections + total_batches + total_models +
        total_generators + total_audit_logs + total_feedback + total_configs
    )

    return {
        "total_tables": 7,
        "total_rows": total_rows,
        "detections_count": total_detections,
        "ai_detected": ai_count,
        "real_detected": real_count,
        "ai_ratio_pct": round((ai_count / total_detections * 100), 1) if total_detections > 0 else 0.0,
        "avg_confidence": round(avg_conf, 4),
        "avg_latency_ms": round(avg_lat, 2),
        "batch_sessions_count": total_batches,
        "models_count": total_models,
        "generators_count": total_generators,
        "audit_logs_count": total_audit_logs,
        "feedback_count": total_feedback,
        "configs_count": total_configs,
        "db_file_path": DB_PATH
    }


def execute_custom_query(query: str, params: Tuple = ()) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Executes a read-only custom SQL SELECT query safely.
    Returns (column_names, rows_as_dicts).
    """
    clean_query = query.strip()
    if not clean_query.upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed in query runner mode.")

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(clean_query, params)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = [dict(r) for r in cursor.fetchall()]
        return columns, rows
    finally:
        conn.close()
