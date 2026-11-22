# Initial Project Inspection Report

**Project Title:** Grad-CAM Enabled Detection of AI-Generated Images Using Spatio-Frequency Features  
**Inspection Date:** October 27, 2026  
**Environment:** Windows OS, Python 3.13.5, PyTorch 2.14.0+cpu (CUDA: False), Torchvision 0.29.0+cpu, Streamlit 1.63.0, Pytest 9.1.1  

---

## 1. Discovered Components and Directory Structure

1. **Application Layer:**
   - Entry point: `app.py` (Streamlit-based forensic dashboard with top horizontal navigation).
   - Views: `views/home.py`, `views/detect.py`, `views/analysis.py`, `views/performance.py`, `views/reports.py`, `views/about.py`.
   - UI utilities: `ui/styles.py` (Forensic Cyber-Lab Dark CSS Theme), `ui/pipeline.py` (state management and model caches), `ui/components.py`, `ui/gauge.py`, `ui/pdf_generator.py`.

2. **Model Architecture and Checkpoints:**
   - Model definitions: `models/dual_branch.py` implementing `SpatialBranch` (ResNet-50 backbone with 256-d bottleneck embedding), `FrequencyBranch` (4-stage ConvNet for 3-channel 2D FFT spectral maps), and `DualBranchFusion` (Gated attention fusion blending spatial and frequency vectors with sigmoid gating $\alpha$).
   - Checkpoints on disk (`models/checkpoints/`):
     - `spatial_resnet50_best.pth` (220.4 MB, verified valid PyTorch state dict).
     - `dual_branch_fusion_best.pth` (227.9 MB, verified valid PyTorch state dict with calibrated threshold $\tau = 0.55$).
     - `dual_branch_fusion_aug_retrain.pth` (227.9 MB, auxiliary retrained checkpoint).

3. **Preprocessing and Spectral Analysis:**
   - Spatial preprocessing: `preprocessing/image_preprocessing.py` (dummy and normalization pipelines).
   - Frequency transformation: `preprocessing/frequency_transform.py` implementing `compute_fft_spectrum` (2D FFT centered log-magnitude with percentile normalization) and `compute_multi_band_fft` (3-channel representation: full spectrum, high-pass filter $r > 15$, and bandpass filter $15 < r < 60$).
   - Dataset loader: `preprocessing/dataset_loader.py` implementing standard and dual-branch PyTorch DataLoaders with stratified splits and held-out generator isolation.

4. **Explainability Engine:**
   - Grad-CAM: `explainability/gradcam.py` implementing forward/backward hooks attached to `model.layer4[-1]` of ResNet-50, generating normalized heatmaps and colormap overlays.

5. **Inference Interface:**
   - `inference/predictor.py` exposing `DummyPredictor`, `SpatialPredictor`, and `DualBranchPredictor`.

6. **Training and Evaluation Scripts:**
   - Training: `training/train_spatial.py`, `training/train_fusion.py`.
   - Evaluation: `eval_and_save_spatial.py`, `eval_and_save_fusion.py`, `evaluation/evaluate.py`, `evaluation/metrics.py`, `evaluation/robustness.py`, `evaluation/final_report.py`.
   - Historical results: `evaluation/results/spatial_baseline_metrics.json`, `evaluation/results/fusion_metrics.json`, `evaluation/results/robustness_results.json`.

7. **Dataset Availability:**
   - Root: `data/raw/`
   - REAL images: 2,500 samples in `data/raw/REAL/imagenet/`.
   - AI-GENERATED images: 8,436 total samples across:
     - ADM: 2,000 samples in `data/raw/AI-GENERATED/adm/`
     - Stable Diffusion v1.4: 2,436 samples in `data/raw/AI-GENERATED/sd_v1_4/`
     - Stable Diffusion v1.5: 2,000 samples in `data/raw/AI-GENERATED/sd_v1_5/`
     - Midjourney: 2,000 samples in `data/raw/AI-GENERATED/midjourney/` (held-out generator).

8. **Existing Test Suite:**
   - `tests/test_inference.py`: 4 unit tests covering preprocessing, dummy predictor, spatial predictor, and dual-branch predictor. All 4 tests PASS.

---

## 2. Identified Limitations and Missing Items

1. **Hardware Acceleration:** CUDA is unavailable on this host system (`torch.cuda.is_available() == False`). All model inference, unit tests, and evaluation scripts execute in CPU mode.
2. **College Record Headings for 6.1.2 & 6.1.3:** The existing project repository and reference documents (`Overview/SRS.docx`, `Overview/gaps-and-flaws.docx`, `Overview/implementation-guide.docx`) provide project specifications but do not define custom subheadings for sections 6.1.2 and 6.1.3. Following academic testing engineering practices, standard, project-aligned headings will be adopted:
   - 6.1.2 Frequency-Domain Multi-Band Spectral Extraction and Tensor Conversion
   - 6.1.3 Interpretability Verification and Grad-CAM Hook Invariant Testing
3. **Empty Demo Directory:** The `demo/sample_images/` directory is empty, but representative test images are directly available in `data/raw/REAL/imagenet/` and `data/raw/AI-GENERATED/`.
4. **Historical File Integrity:** Historical JSON files in `evaluation/results/` are intact and verified. No files were overwritten or corrupted during inspection.
