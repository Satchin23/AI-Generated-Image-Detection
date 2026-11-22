import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

# Dataset Configuration
CLASSES = ["REAL", "AI-GENERATED"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 42

# We will define the held-out generator for evaluation.
# Held-out generator is completely isolated from training/validation splits.
HELD_OUT_GENERATOR = "midjourney"

# Validation/Test splits
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
