"""
Database package for AI-Generated Image Detector Forensic Suite.
Provides SQLite database initialization, schema migration, ORM/query helpers, and seeding.
"""

from .db_manager import (
    get_db_connection,
    init_db,
    seed_database_if_empty,
    insert_detection,
    get_all_detections,
    get_detection_by_id,
    delete_detection,
    clear_all_detections,
    get_batch_sessions,
    get_models_registry,
    get_benchmark_generators,
    get_audit_logs,
    insert_audit_log,
    get_verification_feedback,
    insert_verification_feedback,
    get_system_configs,
    get_database_summary_stats,
    execute_custom_query
)

__all__ = [
    "get_db_connection",
    "init_db",
    "seed_database_if_empty",
    "insert_detection",
    "get_all_detections",
    "get_detection_by_id",
    "delete_detection",
    "clear_all_detections",
    "get_batch_sessions",
    "get_models_registry",
    "get_benchmark_generators",
    "get_audit_logs",
    "insert_audit_log",
    "get_verification_feedback",
    "insert_verification_feedback",
    "get_system_configs",
    "get_database_summary_stats",
    "execute_custom_query"
]

