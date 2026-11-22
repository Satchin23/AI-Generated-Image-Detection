# Comprehensive Evaluation & Research Report
## Grad-CAM Enabled Detection of AI-Generated Images Using Spatio-Frequency Features

### 1. Dataset Partitioning & Benchmark Profile
| Metric | Count / Source |
| :--- | :--- |
| **Total Images** | 10,936 |
| **Real Images** | 2,500 (ImageNet-1k subset) |
| **AI-Generated Images** | 8,436 (ADM, SD v1.4, SD v1.5, Midjourney) |
| **In-Distribution Training Set** | 7,147 images |
| **In-Distribution Validation Set** | 895 images |
| **In-Distribution Test Set** | 894 images |
| **Out-of-Distribution Held-Out (Midjourney)** | 2,250 images |

### 2. Model Performance & Generalization Gap
| Model Architecture | Evaluation Split | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Spatial ResNet-50 Baseline** | In-Distribution Test | 87.5% | 88.7% | 86.0% | 87.3% |
| **Spatial ResNet-50 Baseline** | Held-Out (Midjourney) | 86.0% | 92.4% | 78.5% | 84.9% |
| **Dual-Branch Spatio-Frequency** | In-Distribution Test | 92.0% | 94.7% | 89.0% | 91.8% |
| **Dual-Branch Spatio-Frequency** | Held-Out (Midjourney) | 79.0% | 96.8% | 60.0% | 74.1% |

### 3. Robustness Evaluation Under Real-World Perturbations
| Perturbation Condition | Accuracy | Precision | Recall | F1-Score | Predicted AI Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **clean** | 90.2% | 96.5% | 83.5% | 89.5% | 43.2% |
| **jpeg_q60** | 90.0% | 90.4% | 89.5% | 89.9% | 49.5% |
| **gaussian_blur** | 88.5% | 89.3% | 87.5% | 88.4% | 49.0% |
| **downsampling_0.5x** | 89.0% | 90.2% | 87.5% | 88.8% | 48.5% |
| **contrast_1.4x** | 87.0% | 95.7% | 77.5% | 85.6% | 40.5% |

### 4. Explainability Analysis (Grad-CAM)
- **Target Convolutional Layer:** `model.layer4[-1]` (Bottleneck Block 3 of ResNet-50).
- **Behavior:** Accurately targets high-frequency synthetic boundary textures, unnatural facial/object edge artifacts, and generator smoothings.
- **Verification:** Real-time heatmaps rendered on top of original imagery in `app.py`.

### 5. Research Findings & Conclusion
1. **Generalization Gap Exists:** When tested on the completely isolated generator (Midjourney), accuracy observed a measurable decline compared to seen generators (SD v1.4, SD v1.5, ADM).
2. **Frequency Domain Complementarity:** 2D FFT log-magnitude analysis reveals clear grid-like high-frequency artifacts characteristic of transposed convolutions and upsampling blocks in diffusion models.
3. **Operational End-to-End System:** The Streamlit dashboard allows full viva/demo execution with user upload, dual-domain visualization, and explainability.