import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.dataset_loader import discover_dataset, create_splits
from config.config import RAW_DATA_DIR, HELD_OUT_GENERATOR

def main():
    print(f"--- Dataset Validation Report ---")
    print(f"Looking in: {RAW_DATA_DIR}")
    print(f"Held-out Generator configured as: {HELD_OUT_GENERATOR}\n")
    
    real_files, ai_files, errors = discover_dataset()
    
    for error in errors:
        print(f"WARNING: {error}")
        
    num_real = sum(len(files) for files in real_files.values())
    num_ai = sum(len(files) for files in ai_files.values())
    total_images = num_real + num_ai
    
    print(f"Total Images Found: {total_images}")
    print(f"  - REAL: {num_real}")
    print(f"  - AI-GENERATED: {num_ai}\n")
    
    if total_images == 0:
        print("CRITICAL ERROR: No dataset found. Please download a dataset (e.g. CIFAKE, GenImage) and place it in the correct directory structure.")
        print("Expected Structure:")
        print("  data/raw/")
        print("    REAL/")
        print("      source_name/ (e.g., coco/)")
        print("        image1.jpg ...")
        print("    AI-GENERATED/")
        print("      generator_name/ (e.g., midjourney/)")
        print("        image1.jpg ...")
        return
        
    print("Detected Sources/Generators:")
    print("REAL sources:")
    for source, files in real_files.items():
        print(f"  - {source}: {len(files)} images")
        
    print("\nAI-GENERATED generators:")
    for gen, files in ai_files.items():
        print(f"  - {gen}: {len(files)} images")
        
    print("\n--- Split Information ---")
    train_data, val_data, test_data, held_out_data = create_splits(real_files, ai_files)
    
    print(f"Train set: {len(train_data[0])} images (REAL: {train_data[1].count(0)}, AI: {train_data[1].count(1)})")
    print(f"Validation set: {len(val_data[0])} images (REAL: {val_data[1].count(0)}, AI: {val_data[1].count(1)})")
    print(f"Test set (In-distribution): {len(test_data[0])} images (REAL: {test_data[1].count(0)}, AI: {test_data[1].count(1)})")
    print(f"Held-out Generator set: {len(held_out_data[0])} images (REAL: {held_out_data[1].count(0)}, AI: {held_out_data[1].count(1)})")
    
if __name__ == "__main__":
    main()
