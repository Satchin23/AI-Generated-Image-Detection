"""
Fresh Verification Evaluation and Comparative Analysis Graph Generator
Project: Grad-CAM Enabled Detection of AI-Generated Images Using Spatio-Frequency Features
Saves outputs to record_evidence/ without touching historical metrics.
"""

import os
import sys
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config.config import BASE_DIR
from preprocessing.dataset_loader import get_dataloaders, get_dual_dataloaders
from models.dual_branch import DualBranchFusion
from evaluation.metrics import calculate_metrics
from evaluation.evaluate import evaluate_model
from torchvision import models
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 1. Load historical verified metrics
hist_spatial_path = os.path.join(BASE_DIR, "evaluation", "results", "spatial_baseline_metrics.json")
hist_fusion_path = os.path.join(BASE_DIR, "evaluation", "results", "fusion_metrics.json")

with open(hist_spatial_path, "r") as f:
    hist_spatial = json.load(f)
with open(hist_fusion_path, "r") as f:
    hist_fusion = json.load(f)

# 2. Fresh sample-level verification run
print("\n--- Running Fresh Verification Evaluation on 60 Balanced Test Samples ---")
t0 = time.time()
# Load 30 real + 30 AI for in-distribution test
_, _, test_loader_sp, held_out_loader_sp = get_dataloaders(batch_size=16, max_eval_samples=30)
_, _, test_loader_du, held_out_loader_du = get_dual_dataloaders(batch_size=16, max_eval_samples=30, augment=False)

# Load Spatial Model
sp_model = models.resnet50(weights=None)
in_feat = sp_model.fc.in_features
sp_model.fc = nn.Sequential(
    nn.Linear(in_feat, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2)
)
sp_ckpt_path = os.path.join(BASE_DIR, "models", "checkpoints", "spatial_resnet50_best.pth")
sp_ckpt = torch.load(sp_ckpt_path, map_location=device, weights_only=True)
sp_model.load_state_dict(sp_ckpt.get("model_state_dict", sp_ckpt))
sp_model = sp_model.to(device)
sp_model.eval()

# Load Fusion Model
du_model = DualBranchFusion()
du_ckpt_path = os.path.join(BASE_DIR, "models", "checkpoints", "dual_branch_fusion_best.pth")
du_ckpt = torch.load(du_ckpt_path, map_location=device, weights_only=False)
du_model.load_state_dict(du_ckpt.get("model_state_dict", du_ckpt))
du_model = du_model.to(device)
du_model.eval()
threshold = du_ckpt.get("calibrated_threshold", 0.55)

# Evaluate fresh spatial
fresh_sp_test = evaluate_model(sp_model, test_loader_sp, device)
fresh_sp_held = evaluate_model(sp_model, held_out_loader_sp, device)

# Evaluate fresh fusion
def eval_fresh_dual(model, dataloader, th):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for sp_img, fft_img, lbl in dataloader:
            sp_img = sp_img.to(device)
            fft_img = fft_img.to(device)
            out = model(sp_img, fft_img)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().tolist()
            preds = [1 if p >= th else 0 for p in probs]
            all_preds.extend(preds)
            all_labels.extend(lbl.tolist())
    return calculate_metrics(all_labels, all_preds)

fresh_du_test = eval_fresh_dual(du_model, test_loader_du, threshold)
fresh_du_held = eval_fresh_dual(du_model, held_out_loader_du, threshold)
elapsed_fresh = time.time() - t0

fresh_results = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "evaluation_mode": "Fresh Verification on CPU Sample Batch",
    "elapsed_seconds": elapsed_fresh,
    "spatial_baseline": {
        "checkpoint": "models/checkpoints/spatial_resnet50_best.pth",
        "in_distribution_test_fresh": fresh_sp_test,
        "held_out_generator_fresh": fresh_sp_held,
        "historical_benchmark": hist_spatial
    },
    "dual_branch_fusion": {
        "checkpoint": "models/checkpoints/dual_branch_fusion_best.pth",
        "calibrated_threshold": threshold,
        "in_distribution_test_fresh": fresh_du_test,
        "held_out_generator_fresh": fresh_du_held,
        "historical_benchmark": hist_fusion
    }
}

fresh_out_file = os.path.join(BASE_DIR, "record_evidence", "fresh_evaluation_verification.json")
with open(fresh_out_file, "w", encoding="utf-8") as f:
    json.dump(fresh_results, f, indent=4)
print(f"Fresh evaluation results successfully saved to: {fresh_out_file}")

# 3. Generate Comparative Graph based on Verified Numbers
# Using the full benchmark numbers from the verified historical benchmark (evaluated on test splits)
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
sp_indist = [
    hist_spatial["in_distribution_test"]["accuracy"] * 100,
    hist_spatial["in_distribution_test"]["precision"] * 100,
    hist_spatial["in_distribution_test"]["recall"] * 100,
    hist_spatial["in_distribution_test"]["f1"] * 100
]
fu_indist = [
    hist_fusion["in_distribution_test"]["accuracy"] * 100,
    hist_fusion["in_distribution_test"]["precision"] * 100,
    hist_fusion["in_distribution_test"]["recall"] * 100,
    hist_fusion["in_distribution_test"]["f1"] * 100
]

sp_heldout = [
    hist_spatial["held_out_generator"]["accuracy"] * 100,
    hist_spatial["held_out_generator"]["precision"] * 100,
    hist_spatial["held_out_generator"]["recall"] * 100,
    hist_spatial["held_out_generator"]["f1"] * 100
]
fu_heldout = [
    hist_fusion["held_out_generator"]["accuracy"] * 100,
    hist_fusion["held_out_generator"]["precision"] * 100,
    hist_fusion["held_out_generator"]["recall"] * 100,
    hist_fusion["held_out_generator"]["f1"] * 100
]

plt.rcParams.update({'font.sans-serif': 'Times New Roman', 'font.family': 'sans-serif'})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

x = np.arange(len(metrics))
width = 0.35

# Color Palette: Deep Navy for Spatial Baseline, Teal/Cyan for Dual-Branch Fusion
c_spatial = "#1F4E79"
c_fusion = "#2E8B57"

# Panel 1: In-Distribution Evaluation
rects1 = ax1.bar(x - width/2, sp_indist, width, label='Spatial Baseline (ResNet-50)', color=c_spatial, edgecolor='black', linewidth=0.8)
rects2 = ax1.bar(x + width/2, fu_indist, width, label='Dual-Branch Spatio-Frequency Fusion', color=c_fusion, edgecolor='black', linewidth=0.8)

ax1.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
ax1.set_title('(a) In-Distribution Test Set (ADM, SD v1.4, SD v1.5)', fontsize=12, fontweight='bold', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=10, fontweight='bold')
ax1.set_ylim(0, 105)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.legend(loc='lower right', framealpha=0.9, fontsize=9)

for rect in rects1:
    h = rect.get_height()
    ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
for rect in rects2:
    h = rect.get_height()
    ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

# Panel 2: Held-Out Generator Evaluation
rects3 = ax2.bar(x - width/2, sp_heldout, width, label='Spatial Baseline (ResNet-50)', color=c_spatial, edgecolor='black', linewidth=0.8)
rects4 = ax2.bar(x + width/2, fu_heldout, width, label='Dual-Branch Spatio-Frequency Fusion', color=c_fusion, edgecolor='black', linewidth=0.8)

ax2.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Out-of-Distribution Held-Out Generator (Midjourney)', fontsize=12, fontweight='bold', pad=12)
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=10, fontweight='bold')
ax2.set_ylim(0, 105)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.legend(loc='lower right', framealpha=0.9, fontsize=9)

for rect in rects3:
    h = rect.get_height()
    ax2.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
for rect in rects4:
    h = rect.get_height()
    ax2.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.suptitle('Figure 7.1: Comparative Performance Analysis — Spatial ResNet-50 Baseline vs. Dual-Branch Fusion',
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])

graph_output_path = os.path.join(BASE_DIR, "record_evidence", "comparative_analysis_graph.png")
plt.savefig(graph_output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Comparative graph successfully created at: {graph_output_path}")
