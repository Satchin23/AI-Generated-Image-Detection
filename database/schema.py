"""
SQLite Relational Database Schema Definitions for AI Image Forensics.
Defines 7 relational tables with indexes, foreign keys, and default values.
"""

SCHEMA_SQL = """
-- 1. Examination & Detection Records
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    file_hash_sha256 TEXT NOT NULL,
    file_size_bytes INTEGER DEFAULT 0,
    image_width INTEGER DEFAULT 0,
    image_height INTEGER DEFAULT 0,
    verdict TEXT NOT NULL CHECK(verdict IN ('AI-GENERATED', 'REAL')),
    confidence REAL NOT NULL CHECK(confidence >= 0.0 AND confidence <= 1.0),
    calibrated_threshold REAL DEFAULT 0.55,
    model_name TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    spatial_score REAL DEFAULT 0.0,
    frequency_score REAL DEFAULT 0.0,
    fusion_score REAL DEFAULT 0.0,
    has_gradcam INTEGER DEFAULT 1 CHECK(has_gradcam IN (0, 1)),
    has_fft INTEGER DEFAULT 1 CHECK(has_fft IN (0, 1)),
    generator_family TEXT DEFAULT 'Unknown',
    dataset_category TEXT DEFAULT 'Held-Out',
    examiner_notes TEXT,
    analysis_mode TEXT DEFAULT 'Single Image',
    originality_score REAL DEFAULT 0.0,
    ai_score REAL DEFAULT 0.0,
    execution_device TEXT DEFAULT 'CPU',
    segment_data_json TEXT,
    exam_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 2. Batch Scan Sessions
CREATE TABLE IF NOT EXISTS batch_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL UNIQUE,
    session_name TEXT NOT NULL,
    total_images INTEGER NOT NULL DEFAULT 0,
    ai_detected_count INTEGER NOT NULL DEFAULT 0,
    real_detected_count INTEGER NOT NULL DEFAULT 0,
    mean_confidence REAL DEFAULT 0.0,
    mean_latency_ms REAL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'COMPLETED' CHECK(status IN ('COMPLETED', 'IN_PROGRESS', 'FAILED')),
    initiated_by TEXT DEFAULT 'Academic Viva Console',
    started_at DATETIME NOT NULL,
    completed_at DATETIME
);

-- 3. Models Registry & Benchmarks
CREATE TABLE IF NOT EXISTS models_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL UNIQUE,
    model_name TEXT NOT NULL,
    architecture_type TEXT NOT NULL,
    checkpoint_file TEXT NOT NULL,
    accuracy REAL NOT NULL,
    precision_score REAL NOT NULL,
    recall_score REAL NOT NULL,
    f1_score REAL NOT NULL,
    auc_roc REAL NOT NULL,
    parameters_millions REAL NOT NULL,
    input_resolution TEXT DEFAULT '224x224',
    is_active INTEGER NOT NULL DEFAULT 0 CHECK(is_active IN (0, 1)),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 4. Benchmark Generative Engines & Artifact Signatures
CREATE TABLE IF NOT EXISTS benchmark_generators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generator_name TEXT NOT NULL UNIQUE,
    generator_family TEXT NOT NULL CHECK(generator_family IN ('Diffusion', 'GAN', 'Autoregressive', 'Natural Capture')),
    frequency_fingerprint_type TEXT NOT NULL,
    spectral_energy_ratio REAL DEFAULT 0.0,
    known_weaknesses TEXT,
    sample_count INTEGER DEFAULT 0,
    detection_rate REAL DEFAULT 0.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 5. Forensic Audit & Tamper Log
CREATE TABLE IF NOT EXISTS forensic_audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL CHECK(event_type IN ('DETECTION_RUN', 'PDF_EXPORTED', 'BATCH_EVALUATION', 'DB_SEEDED', 'THRESHOLD_UPDATE', 'CLEAR_HISTORY')),
    severity TEXT NOT NULL DEFAULT 'INFO' CHECK(severity IN ('INFO', 'WARNING', 'CRITICAL')),
    details TEXT NOT NULL,
    target_record_id TEXT,
    ip_origin TEXT DEFAULT '127.0.0.1',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 6. Human-in-the-Loop Verification & Ground Truth Feedback
CREATE TABLE IF NOT EXISTS verification_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id TEXT NOT NULL,
    human_verdict TEXT NOT NULL CHECK(human_verdict IN ('CONFIRMED_AI', 'CONFIRMED_REAL', 'FALSE_POSITIVE', 'FALSE_NEGATIVE', 'DISPUTED')),
    reviewer_name TEXT NOT NULL DEFAULT 'Examiner',
    review_notes TEXT,
    confidence_rating INTEGER CHECK(confidence_rating BETWEEN 1 AND 5),
    verified_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (record_id) REFERENCES detections(record_id) ON DELETE CASCADE
);

-- 7. System Calibration & Global Configuration Settings
CREATE TABLE IF NOT EXISTS system_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_key TEXT NOT NULL UNIQUE,
    config_value TEXT NOT NULL,
    data_type TEXT NOT NULL DEFAULT 'string' CHECK(data_type IN ('string', 'float', 'int', 'boolean', 'json')),
    description TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for optimal lookup performance
CREATE INDEX IF NOT EXISTS idx_detections_record_id ON detections(record_id);
CREATE INDEX IF NOT EXISTS idx_detections_verdict ON detections(verdict);
CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(exam_timestamp);
CREATE INDEX IF NOT EXISTS idx_detections_mode ON detections(analysis_mode);
CREATE INDEX IF NOT EXISTS idx_audit_created_at ON forensic_audit_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_record_id ON verification_feedback(record_id);
"""
