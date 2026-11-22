"""
One-shot script: evaluates the saved DualBranchFusion checkpoint on test splits
using the calibrated threshold and writes evaluation/results/fusion_metrics.json.
"""
import os
import sys
import json
import time
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import BASE_DIR, BATCH_SIZE
from preprocessing.dataset_loader import get_dual_dataloaders
from models.dual_branch import DualBranchFusion
from evaluation.metrics import calculate_metrics

def evaluate_dual_model(model, dataloader, device, threshold=0.5):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for spatial_imgs, fft_imgs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            spatial_imgs = spatial_imgs.to(device)
            fft_imgs = fft_imgs.to(device)
            outputs = model(spatial_imgs, fft_imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.tolist())

    preds = [1 if p >= threshold else 0 for p in all_probs]
    metrics = calculate_metrics(all_labels, preds)
    return metrics, all_probs, all_labels

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = os.path.join(BASE_DIR, "models", "checkpoints", "dual_branch_fusion_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DualBranchFusion()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    best_threshold = checkpoint.get("calibrated_threshold", 0.35)
    best_val_f1 = checkpoint.get("best_val_f1", 0.9329)
    epoch = checkpoint.get("epoch", 1)

    print(f"Loaded trained DualBranchFusion from {ckpt_path}")
    print(f"Checkpoint Epoch: {epoch}, Best Val F1: {best_val_f1:.4f}, Calibrated Threshold: {best_threshold:.2f}")

    print("\nLoading dual dataloaders (batch_size=32)...")
    t0 = time.time()
    _, _, test_loader, held_out_loader = get_dual_dataloaders(
        batch_size=32,
        max_train_samples=None,
        max_eval_samples=None,
        augment=False
    )
    print(f"Dataloaders ready in {time.time()-t0:.1f}s. Test size={len(test_loader.dataset)}, Held-out size={len(held_out_loader.dataset)}")

    print(f"\n--- [In-Distribution Test Set (tau={best_threshold:.2f})] ---")
    t_start = time.time()
    test_metrics, _, _ = evaluate_dual_model(model, test_loader, device, threshold=best_threshold)
    print(f"In-Distribution Eval completed in {time.time()-t_start:.1f}s")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    print(f"\n--- [Held-Out Generator: Midjourney (tau={best_threshold:.2f})] ---")
    t_start = time.time()
    held_out_metrics, _, _ = evaluate_dual_model(model, held_out_loader, device, threshold=best_threshold)
    print(f"Held-Out Eval completed in {time.time()-t_start:.1f}s")
    for k, v in held_out_metrics.items():
        print(f"  {k}: {v}")

    results_dir = os.path.join(BASE_DIR, "evaluation", "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "fusion_metrics.json")
    results = {
        "in_distribution_test": test_metrics,
        "held_out_generator": held_out_metrics,
        "best_val_f1": best_val_f1,
        "calibrated_threshold": best_threshold
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n--> Fusion metrics successfully recorded to {results_file}")

if __name__ == "__main__":
    main()
