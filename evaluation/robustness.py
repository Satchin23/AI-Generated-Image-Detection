import os
import sys
import io
import json
import torch
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as transforms
from tqdm import tqdm

# Project Root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import BASE_DIR, IMAGE_SIZE
from preprocessing.dataset_loader import discover_dataset, create_splits
from evaluation.metrics import calculate_metrics
from models.dual_branch import DualBranchFusion
from preprocessing.frequency_transform import compute_multi_band_fft, fft_spectrum_to_tensor

def apply_jpeg_compression(image: Image.Image, quality=60) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')

def apply_gaussian_blur(image: Image.Image, radius=2) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_downsampling(image: Image.Image, scale=0.5) -> Image.Image:
    w, h = image.size
    small = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)

def apply_contrast(image: Image.Image, factor=1.4) -> Image.Image:
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

DISTORTIONS = {
    "clean": lambda img: img,
    "jpeg_q60": lambda img: apply_jpeg_compression(img, quality=60),
    "gaussian_blur": lambda img: apply_gaussian_blur(img, radius=2),
    "downsampling_0.5x": lambda img: apply_downsampling(img, scale=0.5),
    "contrast_1.4x": lambda img: apply_contrast(img, factor=1.4)
}

def evaluate_fusion_robustness(checkpoint_path, test_samples, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DualBranchFusion()
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    tau = ckpt.get("calibrated_threshold", 0.55)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Using saved decision threshold: tau={tau:.2f}")

    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    results = {}
    print(f"Evaluating {len(test_samples)} test samples per perturbation condition across {len(DISTORTIONS)} conditions...")

    for dist_name, dist_fn in DISTORTIONS.items():
        all_preds = []
        all_labels = []
        
        pbar = tqdm(range(0, len(test_samples), batch_size), desc=f"Fusion [{dist_name}]", leave=False)
        for i in pbar:
            batch = test_samples[i:i + batch_size]
            s_tensors = []
            f_tensors = []
            b_labels = []

            for path, label in batch:
                try:
                    img = Image.open(path).convert('RGB')
                    dist_img = dist_fn(img)
                    s_tensor = transform(dist_img)
                    m_band = compute_multi_band_fft(dist_img, size=IMAGE_SIZE)
                    f_tensor = fft_spectrum_to_tensor(m_band)
                    s_tensors.append(s_tensor)
                    f_tensors.append(f_tensor)
                    b_labels.append(label)
                except Exception:
                    continue

            if not s_tensors:
                continue

            s_batch = torch.stack(s_tensors).to(device)
            f_batch = torch.stack(f_tensors).to(device)

            with torch.no_grad():
                out = model(s_batch, f_batch)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().tolist()
                preds = [1 if p >= tau else 0 for p in probs]

            all_preds.extend(preds)
            all_labels.extend(b_labels)

        metrics = calculate_metrics(all_labels, all_preds)
        metrics["predicted_ai_rate"] = float(sum(all_preds) / max(1, len(all_preds)))
        results[dist_name] = metrics

    return results, tau

def main():
    real_files, ai_files, _ = discover_dataset()
    # Use exact 400-sample in-distribution test split matching evaluation/evaluate.py
    _, _, (test_paths, test_labels), _ = create_splits(real_files, ai_files, max_eval_samples=400)
    test_samples = list(zip(test_paths, test_labels))

    print(f"Total in-distribution test samples for robustness: {len(test_samples)}")
    fusion_ckpt = os.path.join(BASE_DIR, "models", "checkpoints", "dual_branch_fusion_best.pth")

    if not os.path.exists(fusion_ckpt):
        print(f"[ERROR] Dual-branch fusion checkpoint not found: {fusion_ckpt}")
        return

    print("\n=== Running Robustness Evaluation on Dual-Branch Spatio-Frequency Fusion ===")
    fusion_rob, tau = evaluate_fusion_robustness(fusion_ckpt, test_samples, batch_size=32)

    print("\n" + "="*80)
    print(f"{'Perturbation':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Pred-AI Rate':<12}")
    print("="*80)
    for name in DISTORTIONS.keys():
        m = fusion_rob.get(name, {})
        acc = f"{m.get('accuracy', 0)*100:.1f}%"
        prec = f"{m.get('precision', 0)*100:.1f}%"
        rec = f"{m.get('recall', 0)*100:.1f}%"
        f1 = f"{m.get('f1', 0)*100:.1f}%"
        pair = f"{m.get('predicted_ai_rate', 0)*100:.1f}%"
        print(f"{name:<20} | {acc:<10} | {prec:<10} | {rec:<10} | {f1:<10} | {pair:<12}")
    print("="*80)

    out_dir = os.path.join(BASE_DIR, "evaluation", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "robustness_results.json")
    with open(out_file, "w") as f:
        json.dump(fusion_rob, f, indent=4)
    print(f"\nRobustness results written to {out_file}")

if __name__ == "__main__":
    main()
