"""
One-shot script: evaluates the saved spatial checkpoint on all splits
and writes spatial_baseline_metrics.json. Run this after training.
"""
import os, sys, json, torch
from torchvision import models
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import BASE_DIR, BATCH_SIZE
from preprocessing.dataset_loader import get_dataloaders
from evaluation.evaluate import evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Build model
model = models.resnet50(weights=None)
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2)
)

ckpt_path = os.path.join(BASE_DIR, "models", "checkpoints", "spatial_resnet50_best.pth")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(ckpt.get("model_state_dict", ckpt))
model = model.to(device)
model.eval()
print(f"Loaded checkpoint from {ckpt_path}")
print(f"Checkpoint best_val_f1: {ckpt.get('best_val_f1', 'N/A')}")

# Use balanced 200 samples for swift evaluation on CPU
_, _, test_loader, held_out_loader = get_dataloaders(batch_size=32, max_train_samples=None, max_eval_samples=200)

print("\n--- In-Distribution Test Set ---")
test_metrics = evaluate_model(model, test_loader, device)
for k, v in test_metrics.items():
    print(f"  {k}: {v}")

print("\n--- Held-Out Generator: Midjourney ---")
held_out_metrics = evaluate_model(model, held_out_loader, device)
for k, v in held_out_metrics.items():
    print(f"  {k}: {v}")

results = {
    "in_distribution_test": test_metrics,
    "held_out_generator": held_out_metrics,
    "best_val_f1": ckpt.get("best_val_f1", 0.0)
}

out_path = os.path.join(BASE_DIR, "evaluation", "results", "spatial_baseline_metrics.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"\n--> Saved to {out_path}")
