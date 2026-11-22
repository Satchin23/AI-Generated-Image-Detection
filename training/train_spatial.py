import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

# Project Root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import BASE_DIR, BATCH_SIZE, RANDOM_SEED
from preprocessing.dataset_loader import get_dataloaders
from evaluation.evaluate import evaluate_model

def build_spatial_resnet50(pretrained=True, fine_tune_layer3=True):
    """
    Constructs the Spatial Branch ResNet50 baseline:
    - Pretrained ResNet50 on ImageNet
    - Early layers (conv1, bn1, layer1, layer2) frozen
    - Deep layers (layer3 optional, layer4, fc) trainable
    - Classification head replaced: Linear(2048, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 2)
    """
    if pretrained:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)

    # Freeze early layers
    for name, param in model.named_parameters():
        if "fc" in name or "layer4" in name or (fine_tune_layer3 and "layer3" in name):
            param.requires_grad = True
        else:
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    return model

def train(epochs=2, lr=1e-4, batch_size=BATCH_SIZE, max_train_samples=None, max_eval_samples=None, fine_tune_layer3=False):
    import json
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Spatial ResNet-50 Baseline Training ===")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {lr}")
    if max_train_samples:
        print(f"Subsampling: max_train_samples={max_train_samples}, max_eval_samples={max_eval_samples}")

    train_loader, val_loader, test_loader, held_out_loader = get_dataloaders(
        batch_size=batch_size, max_train_samples=max_train_samples, max_eval_samples=max_eval_samples
    )

    model = build_spatial_resnet50(pretrained=True, fine_tune_layer3=fine_tune_layer3)
    model = model.to(device)

    # Calculate dynamic class weights for training set balance
    train_labels = train_loader.dataset.labels
    n_real = sum(1 for l in train_labels if l == 0)
    n_ai = sum(1 for l in train_labels if l == 1)
    total_samples = len(train_labels)
    weight_real = total_samples / (2.0 * max(1, n_real))
    weight_ai = total_samples / (2.0 * max(1, n_ai))
    class_weights = torch.tensor([weight_real, weight_ai], dtype=torch.float).to(device)
    print(f"Training distribution: {n_real} REAL, {n_ai} AI. Class weights: REAL={weight_real:.2f}, AI={weight_ai:.2f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Train size={len(train_loader.dataset)} Val={len(val_loader.dataset)} Test={len(test_loader.dataset)} Held-out={len(held_out_loader.dataset)} AMP={use_amp}")

    best_val_f1 = -1.0
    checkpoints_dir = os.path.join(BASE_DIR, "models", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoints_dir, "spatial_resnet50_best.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=True)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
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

        # Validation
        print("Evaluating on Validation Set...")
        val_metrics = evaluate_model(model, val_loader, device)
        print(f"Validation Metrics: Acc={val_metrics['accuracy']:.4f}, Prec={val_metrics['precision']:.4f}, Rec={val_metrics['recall']:.4f}, F1={val_metrics['f1']:.4f}")
        print(f"Validation Confusion Matrix [TN, FP; FN, TP]: {val_metrics['confusion_matrix']}")

        if val_metrics["f1"] >= best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "best_val_f1": best_val_f1
            }, best_checkpoint_path)
            print(f"--> Saved new best checkpoint to {best_checkpoint_path}")

    # Final In-Distribution and Out-of-Distribution (Held-out) Evaluation
    print("\n==========================================")
    print("=== FINAL BASELINE EVALUATION (BEST MODEL) ===")
    print("==========================================")
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\n--- [In-Distribution Test Set] ---")
    test_metrics = evaluate_model(model, test_loader, device)
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    print("\n--- [Held-Out Generator: Midjourney] ---")
    held_out_metrics = evaluate_model(model, held_out_loader, device)
    for k, v in held_out_metrics.items():
        print(f"  {k}: {v}")

    # Save metrics to results file
    results_dir = os.path.join(BASE_DIR, "evaluation", "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "spatial_baseline_metrics.json")
    with open(results_file, "w") as f:
        json.dump({
            "in_distribution_test": test_metrics,
            "held_out_generator": held_out_metrics,
            "best_val_f1": best_val_f1
        }, f, indent=4)
    print(f"\n--> Metrics recorded to {results_file}")

    return test_metrics, held_out_metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Max train samples")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Max eval samples")
    parser.add_argument("--fine_tune_layer3", action="store_true", help="Also fine tune layer3")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        fine_tune_layer3=args.fine_tune_layer3
    )
