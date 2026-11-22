"""
Semantic region segmentation module for meaningful image decomposition.
Extracts anatomical & portrait regions: Face & Skin, Eyes, Hair, Clothing, and Background
using DeepLabV3/LRASPP semantic parsing and color-chrominance space filtering.
"""

import os
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.models.segmentation as seg
import cv2

# Regional Color Palette for Segmentation Mask Visualization
REGION_COLORS = {
    "Face & Skin": [245, 158, 11],     # Amber / Warm Gold
    "Eyes": [0, 240, 255],             # Electric Cyan
    "Hair": [168, 85, 247],            # Neon Violet / Purple
    "Clothing": [16, 185, 129],        # Emerald Green
    "Background": [100, 116, 139],     # Slate Blue
    "Foreground Region": [245, 158, 11],
    "Salient Core": [0, 240, 255],
    "Upper Structure": [168, 85, 247],
    "Base Structure": [16, 185, 129],
}

_SEG_MODEL = None

def get_segmentation_model():
    """Loads and caches the pretrained LRASPP MobileNet-V3 segmentation backbone."""
    global _SEG_MODEL
    if _SEG_MODEL is None:
        try:
            model = seg.lraspp_mobilenet_v3_large(weights=seg.LRASPP_MobileNet_V3_Large_Weights.DEFAULT)
            model.eval()
            _SEG_MODEL = model
        except Exception:
            model = seg.lraspp_mobilenet_v3_large(weights="DEFAULT")
            model.eval()
            _SEG_MODEL = model
    return _SEG_MODEL


def segment_portrait_regions(pil_img: Image.Image):
    """
    Decomposes an input portrait image into 5 meaningful regions:
    1. Face & Skin
    2. Eyes
    3. Hair
    4. Clothing
    5. Background
    
    Returns a dictionary of region masks, bounding box crops, pixel metrics,
    and a blended color segmentation overlay.
    """
    w, h = pil_img.size
    img_np = np.array(pil_img)
    total_pixels = w * h

    seg_model = get_segmentation_model()
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    inp = transform(pil_img).unsqueeze(0)
    
    with torch.no_grad():
        out = seg_model(inp)["out"]
    pred = torch.argmax(out.squeeze(), dim=0).byte().cpu().numpy()

    # In Pascal VOC, class 15 is 'person'
    person_mask = (pred == 15)
    bg_mask = ~person_mask

    masks = {}

    if np.sum(person_mask) > (total_pixels * 0.02):
        # Portrait / Person detected
        rows = np.any(person_mask, axis=1)
        cols = np.any(person_mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        person_h = max(10, ymax - ymin)
        person_w = max(10, xmax - xmin)

        # Skin Chrominance extraction via YCrCb color space
        img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        cr = img_yuv[:, :, 1]
        cb = img_yuv[:, :, 2]
        # Calibrated human skin threshold in YCrCb
        skin_raw = (cr >= 130) & (cr <= 175) & (cb >= 75) & (cb <= 130)

        # 1. Face & Skin: skin inside the head & neck region
        head_y_end = int(ymin + 0.60 * person_h)
        face_mask = person_mask.copy()
        face_mask[head_y_end:, :] = False
        face_mask = face_mask & skin_raw

        # Morphological clean up to connect contiguous facial skin
        kernel = np.ones((3, 3), np.uint8)
        face_mask = cv2.dilate(face_mask.astype(np.uint8), kernel, iterations=1).astype(bool) & person_mask

        # 2. Eyes region: ocular band in upper-middle portion of head
        eye_y_start = int(ymin + 0.22 * person_h)
        eye_y_end = int(ymin + 0.44 * person_h)
        eye_x_start = int(xmin + 0.18 * person_w)
        eye_x_end = int(xmin + 0.82 * person_w)
        eyes_mask = np.zeros((h, w), dtype=bool)
        eyes_mask[eye_y_start:eye_y_end, eye_x_start:eye_x_end] = person_mask[eye_y_start:eye_y_end, eye_x_start:eye_x_end]

        # 3. Hair region: cranial crown and top/sides of head (excluding skin & eyes)
        hair_mask = person_mask.copy()
        hair_mask[int(ymin + 0.52 * person_h):, :] = False
        hair_mask = hair_mask & (~face_mask) & (~eyes_mask)

        # 4. Clothing region: torso and body lower section, excluding exposed skin
        clothing_mask = person_mask.copy()
        clothing_mask[:int(ymin + 0.50 * person_h), :] = False
        clothing_mask = clothing_mask & (~face_mask)

        # Ensure background takes all remaining non-person pixels
        masks = {
            "Face & Skin": face_mask,
            "Eyes": eyes_mask,
            "Hair": hair_mask,
            "Clothing": clothing_mask,
            "Background": bg_mask
        }
    else:
        # Fallback for non-portrait / general images to prevent failure
        cy, cx = h // 2, w // 2
        central_h, central_w = int(h * 0.4), int(w * 0.4)
        
        salient_mask = np.zeros((h, w), dtype=bool)
        salient_mask[cy - central_h//2 : cy + central_h//2, cx - central_w//2 : cx + central_w//2] = True
        
        upper_mask = np.zeros((h, w), dtype=bool)
        upper_mask[:cy, :] = True
        upper_mask = upper_mask & (~salient_mask)
        
        lower_mask = np.zeros((h, w), dtype=bool)
        lower_mask[cy:, :] = True
        lower_mask = lower_mask & (~salient_mask)
        
        masks = {
            "Salient Core": salient_mask,
            "Upper Structure": upper_mask,
            "Base Structure": lower_mask,
            "Background": bg_mask
        }

    # Build region details dictionary and crops
    regions_data = []
    overlay_np = img_np.copy().astype(float)
    alpha = 0.42

    for name, mask in masks.items():
        pixel_count = int(np.sum(mask))
        if pixel_count < 10:
            continue

        # Extract tight bounding box crop for classifier input
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Minimal padding
        pad = 4
        crop_ymin = max(0, rmin - pad)
        crop_ymax = min(h, rmax + pad + 1)
        crop_xmin = max(0, cmin - pad)
        crop_xmax = min(w, cmax + pad + 1)

        crop_arr = img_np[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        crop_pil = Image.fromarray(crop_arr)

        # Apply tint to overlay
        color = np.array(REGION_COLORS.get(name, [128, 128, 128]), dtype=float)
        overlay_np[mask] = (1 - alpha) * overlay_np[mask] + alpha * color

        # Draw contour border on mask edge for clean visual delineation
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_np, contours, -1, color, 1)

        regions_data.append({
            "name": name,
            "mask": mask,
            "crop": crop_pil,
            "pixel_count": pixel_count,
            "area_pct": (pixel_count / total_pixels) * 100.0,
            "color": REGION_COLORS.get(name, [128, 128, 128]),
            "box": (crop_xmin, crop_ymin, crop_xmax, crop_ymax)
        })

    overlay_img = Image.fromarray(np.clip(overlay_np, 0, 255).astype(np.uint8))

    return {
        "regions": regions_data,
        "overlay": overlay_img,
        "total_pixels": total_pixels
    }
