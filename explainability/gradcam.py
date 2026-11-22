import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
    for PyTorch convolutional neural networks (e.g., ResNet-50 layer4).
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        Generates Grad-CAM heatmap for an input image tensor (shape: [1, C, H, W]).
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        try:
            score = output[0, target_class]
            score.backward(retain_graph=True)

            # Global average pooling of gradients
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            cam = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))


        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        probabilities = F.softmax(output, dim=1).detach().cpu().numpy()[0]
        return cam, target_class, probabilities

    def overlay(self, original_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlays heatmap on top of PIL Image.
        """
        img_np = np.array(original_img)
        h, w = img_np.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        overlayed = np.uint8(alpha * heatmap_color + (1 - alpha) * img_np)
        return overlayed, heatmap_color

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
