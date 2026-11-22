import torch
import numpy as np
from PIL import Image

def compute_fft_spectrum(image: Image.Image, size=(224, 224)) -> np.ndarray:
    """
    Computes a 2D FFT log-magnitude spectrum with DC attenuation and robust scaling.
    Preserves subtle high-frequency grid anomalies from generative diffusion/GAN models.
    """
    img_resized = image.resize(size).convert('L')
    img_arr = np.array(img_resized, dtype=np.float32)

    # 2D FFT & shift low frequencies to center
    fft2 = np.fft.fft2(img_arr)
    fft_shifted = np.fft.fftshift(fft2)

    # Resolution-normalized magnitude
    h, w = img_arr.shape
    magnitude = np.abs(fft_shifted) / np.sqrt(h * w)
    log_spectrum = np.log1p(magnitude)

    # Robust scaling using percentiles to prevent DC spike from flattening high frequencies
    p5 = np.percentile(log_spectrum, 5)
    p99 = np.percentile(log_spectrum, 99.5)
    if p99 > p5:
        norm_spectrum = np.clip((log_spectrum - p5) / (p99 - p5), 0.0, 1.0)
    else:
        norm_spectrum = np.zeros_like(log_spectrum)

    return norm_spectrum.astype(np.float32)

def compute_multi_band_fft(image: Image.Image, size=(224, 224)) -> np.ndarray:
    """
    Constructs a 3-channel frequency feature representation:
    Channel 0: Full centered log-magnitude spectrum
    Channel 1: High-pass filtered spectrum (DC center masked out to highlight generator grid noise)
    Channel 2: Bandpass spectrum (captures mid-frequency synthetic textures)
    """
    spec = compute_fft_spectrum(image, size)
    h, w = spec.shape
    cy, cx = h // 2, w // 2

    # Create coordinate grid distance from center
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Channel 1: High-pass (attenuate center low frequencies, r < 15)
    hp_mask = np.clip((dist - 8) / 15.0, 0.0, 1.0)
    high_pass = spec * hp_mask

    # Channel 2: Bandpass (peak around mid frequencies, r between 15 and 60)
    bp_mask = np.exp(-((dist - 35)**2) / (2 * (15**2)))
    band_pass = spec * bp_mask

    multi_band = np.stack([spec, high_pass, band_pass], axis=0) # [3, H, W]
    return multi_band.astype(np.float32)

def fft_spectrum_to_tensor(spectrum: np.ndarray, channels: int = 3) -> torch.Tensor:
    """
    Converts spectrum array to a PyTorch tensor (shape: [C, H, W]).
    If spectrum is 2D [H, W], replicates to requested channels.
    If spectrum is 3D [3, H, W], returns directly.
    """
    if spectrum.ndim == 2:
        tensor_1c = torch.from_numpy(spectrum).unsqueeze(0)
        if channels == 3:
            return tensor_1c.repeat(3, 1, 1)
        return tensor_1c
    elif spectrum.ndim == 3:
        return torch.from_numpy(spectrum)
    else:
        raise ValueError(f"Unsupported spectrum shape: {spectrum.shape}")
