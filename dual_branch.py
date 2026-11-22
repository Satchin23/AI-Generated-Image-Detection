import os
import torch
import torch.nn as nn
from torchvision import models

class SpatialBranch(nn.Module):
    """
    Spatial CNN branch using fine-tuned ResNet-50 backbone.
    Extracts deep semantic and spatial texture representations.
    Preserves pre-trained 256-d bottleneck embedding from Phase 3.
    """
    def __init__(self, feature_dim=256, pretrained_checkpoint=None):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained_checkpoint is None else None)

        in_features = self.backbone.fc.in_features
        # 256-d penultimate feature representation
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(feature_dim, 2)

        if pretrained_checkpoint and os.path.exists(pretrained_checkpoint):
            ckpt = torch.load(pretrained_checkpoint, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            
            # Map fc weights from spatial baseline checkpoint
            backbone_dict = {}
            fc_dict = {}
            for k, v in state_dict.items():
                if k.startswith("fc."):
                    fc_dict[k.replace("fc.", "")] = v
                else:
                    backbone_dict[k] = v
            self.backbone.load_state_dict(backbone_dict, strict=False)
            
            # Load feature_extractor and classifier if available
            if "0.weight" in fc_dict:
                self.feature_extractor[0].weight.data.copy_(fc_dict["0.weight"])
                self.feature_extractor[0].bias.data.copy_(fc_dict["0.bias"])
            if "3.weight" in fc_dict:
                self.classifier.weight.data.copy_(fc_dict["3.weight"])
                self.classifier.bias.data.copy_(fc_dict["3.bias"])
            print(f"SpatialBranch initialized from {pretrained_checkpoint}")

        # Replace default ResNet fc with identity
        self.backbone.fc = nn.Identity()

        # Freeze early layers, keep layer4 and feature_extractor active
        for name, param in self.backbone.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x, return_logits=False):
        raw_feat = self.backbone(x)
        feat_256 = self.feature_extractor(raw_feat)
        if return_logits:
            return feat_256, self.classifier(feat_256)
        return feat_256

class FrequencyBranch(nn.Module):
    """
    Frequency CNN branch designed for 2D FFT multi-band spectrums.
    Captures high-frequency grid artifacts, Fourier anomalies, and generator periodicities.
    """
    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # 28x28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        feat = self.encoder(x)
        return self.projection(feat)

class DualBranchFusion(nn.Module):
    """
    Gated Spatio-Frequency Fusion Architecture:
    1. Spatial Branch (ResNet-50) -> Spatial Embedding (256-d)
    2. Frequency Branch (FFT ConvNet) -> Frequency Embedding (256-d)
    3. Gated Attention Fusion:
       alpha = Sigmoid(Linear([feat_s, feat_f]))
       feat_fused = alpha * feat_s + (1 - alpha) * feat_f
    4. Classifier Head -> REAL / AI-GENERATED
    Guarantees that the network dynamically balances spatial visual clues
    and frequency domain anomalies without one degrading the other.
    """
    def __init__(self, spatial_checkpoint=None, feature_dim=256, num_classes=2):
        super().__init__()
        self.spatial_branch = SpatialBranch(feature_dim=feature_dim, pretrained_checkpoint=spatial_checkpoint)
        self.frequency_branch = FrequencyBranch(in_channels=3, feature_dim=feature_dim)

        # Cross-modal Gating Mechanism
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        # Initialize gate bias to slightly favor spatial branch initially (alpha ~ 0.7)
        nn.init.constant_(self.gate[0].bias, 0.8)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, spatial_img, frequency_img, return_gate=False):
        feat_s = self.spatial_branch(spatial_img)
        feat_f = self.frequency_branch(frequency_img)

        # Compute dynamic gating weights alpha in [0, 1]
        concat_feat = torch.cat([feat_s, feat_f], dim=1)
        alpha = self.gate(concat_feat)

        # Gated blending
        fused = alpha * feat_s + (1.0 - alpha) * feat_f
        logits = self.classifier(fused)

        if return_gate:
            return logits, alpha
        return logits
