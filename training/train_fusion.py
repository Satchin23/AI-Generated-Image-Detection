import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Project Root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().tolist() # P(AI-Generated)
            all_probs.extend(probs)
            all_labels.extend(labels.tolist())

    preds = [1 if p >= threshold else 0 for p in all_probs]
    metrics = calculate_metrics(all_labels, preds)
    return metrics, all_probs, all_labels

def calibrate_threshold(all_labels, all_probs):
    """
    Finds the optimal decision threshold tau on validation set to balance precision & recall.
    Guard: If the best achievable F1 < 0.3, the model is degenerate (predicts one class).
    In that case, fall back to tau=0.5 to avoid encoding a biased threshold in the checkpoint.
    """
    best_thresh = 0.5
    best_f1 = -1.0
    for t in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        preds = [1 if p >= t else 0 for p in all_probs]
        m = calculate_metrics(all_labels, preds)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = t

    # Degenerate guard: if no threshold achieves meaningful F1, default to 0.5
    if best_f1 < 0.30:
        print(f"  [WARNING] Calibration degenerate (best F1={best_f1:.3f} < 0.30). Forcing tau=0.50.")
        best_thresh = 0.5
    return best_thresh, best_f1

def train_fusion(epochs=3, batch_size=8, max_train_samples=0, max_eval_samples=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Gated Spatio-Frequency Fusion Training ===")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}")
    if max_train_samples:
        print(f"Subsampling: max_train_samples={max_train_samples}, max_eval_samples={max_eval_samples}")

    train_loader, val_loader, test_loader, held_out_loader = get_dual_dataloaders(
        batch_size=batch_size, max_train_samples=max_train_samples, max_eval_samples=max_eval_samples, augment=True
    )

    spatial_checkpoint = os.path.join(BASE_DIR, "models", "checkpoints", "spatial_resnet50_best.pth")
    model = DualBranchFusion(spatial_checkpoint=spatial_checkpoint if os.path.exists(spatial_checkpoint) else None)
    model = model.to(device)

    # Class balance weights
    train_labels = train_loader.dataset.labels
    n_real = sum(1 for l in train_labels if l == 0)
    n_ai = sum(1 for l in train_labels if l == 1)
    total_samples = len(train_labels)
    weight_real = total_samples / (2.0 * max(1, n_real))
    weight_ai = total_samples / (2.0 * max(1, n_ai))
    class_weights = torch.tensor([weight_real, weight_ai], dtype=torch.float).to(device)
    print(f"Training distribution: {n_real} REAL, {n_ai} AI. Weights: REAL={weight_real:.2f}, AI={weight_ai:.2f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Differential Learning Rates: fine-tune spatial backbone gently, train frequency & gate quickly
    spatial_params = list(model.spatial_branch.parameters())
    fusion_params = list(model.frequency_branch.parameters()) + list(model.gate.parameters()) + list(model.classifier.parameters())
    optimizer = optim.Adam([
        {"params": filter(lambda p: p.requires_grad, spatial_params), "lr": 1e-5},
        {"params": filter(lambda p: p.requires_grad, fusion_params), "lr": 2e-4}
    ], weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Train size={len(train_loader.dataset)} Val={len(val_loader.dataset)} Test={len(test_loader.dataset)} Held-out={len(held_out_loader.dataset)} AMP={use_amp}")

    best_val_f1 = -1.0
    best_threshold = 0.5
    checkpoints_dir = os.path.join(BASE_DIR, "models", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoints_dir, "dual_branch_fusion_best.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Gated Fusion]", leave=True)
        for spatial_imgs, fft_imgs, labels in pbar:
            spatial_imgs = spatial_imgs.to(device, non_blocking=True)
            fft_imgs = fft_imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(spatial_imgs, fft_imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct / total:.4f}"
            })

        epoch_loss = running_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        elapsed = time.time() - start_time
        print(f"\nEpoch {epoch}/{epochs} Finished in {elapsed:.1f}s - Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f}")

        # Validation & Threshold Calibration
        print("Evaluating and Calibrating on Validation Set...")
        val_metrics_raw, val_probs, val_labels_list = evaluate_dual_model(model, val_loader, device, threshold=0.5)
        calibrated_thresh, val_f1_calib = calibrate_threshold(val_labels_list, val_probs)
        val_metrics, _, _ = evaluate_dual_model(model, val_loader, device, threshold=calibrated_thresh)
        print(f"Validation (tau={calibrated_thresh:.2f}): Acc={val_metrics['accuracy']:.4f}, Prec={val_metrics['precision']:.4f}, Rec={val_metrics['recall']:.4f}, F1={val_metrics['f1']:.4f}")

        if val_metrics["f1"] >= best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_threshold = calibrated_thresh
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "best_val_f1": best_val_f1,
                "calibrated_threshold": best_threshold
            }, best_checkpoint_path)
            print(f"--> Saved new best Gated Fusion checkpoint to {best_checkpoint_path}")

    # Final In-Distribution & Out-of-Distribution Evaluation using calibrated threshold
    print("\n=======================================================")
    print(f"=== FINAL GATED FUSION EVALUATION (Threshold: {best_threshold:.2f}) ===")
    print("=======================================================")
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    calibrated_tau = checkpoint.get("calibrated_threshold", 0.5)

    print(f"\n--- [In-Distribution Test Set (tau={calibrated_tau:.2f})] ---")
    test_metrics, _, _ = evaluate_dual_model(model, test_loader, device, threshold=calibrated_tau)
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    print(f"\n--- [Held-Out Generator: Midjourney (tau={calibrated_tau:.2f})] ---")
    held_out_metrics, _, _ = evaluate_dual_model(model, held_out_loader, device, threshold=calibrated_tau)
    for k, v in held_out_metrics.items():
        print(f"  {k}: {v}")

    results_dir = os.path.join(BASE_DIR, "evaluation", "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "fusion_metrics.json")
    with open(results_file, "w") as f:
        json.dump({
            "in_distribution_test": test_metrics,
            "held_out_generator": held_out_metrics,
            "best_val_f1": best_val_f1,
            "calibrated_threshold": calibrated_tau
        }, f, indent=4)
    print(f"\n--> Fusion metrics successfully recorded to {results_file}")

    return test_metrics, held_out_metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_train_samples", type=int, default=0, help="Max train samples (0 = full set)")
    parser.add_argument("--max_eval_samples", type=int, default=0, help="Max eval samples (0 = full set)")
    args = parser.parse_args()

    train_fusion(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples
    )
