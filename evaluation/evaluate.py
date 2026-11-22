import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any

# Project Root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import calculate_metrics

def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, Any]:
    """
    Evaluates a model over a dataloader and returns accuracy, precision, recall, f1, and confusion matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    metrics = calculate_metrics(all_labels, all_preds)
    return metrics

if __name__ == "__main__":
    import argparse
    from torchvision import models
    from preprocessing.dataset_loader import get_dataloaders
    from config.config import BATCH_SIZE

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build spatial baseline architecture
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, 2)
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)

    _, val_loader, test_loader, held_out_loader = get_dataloaders(batch_size=BATCH_SIZE)

    print("\n--- In-Distribution Test Set Evaluation ---")
    test_metrics = evaluate_model(model, test_loader, device)
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    print("\n--- Held-Out Generator Evaluation (Midjourney) ---")
    held_out_metrics = evaluate_model(model, held_out_loader, device)
    for k, v in held_out_metrics.items():
        print(f"  {k}: {v}")
