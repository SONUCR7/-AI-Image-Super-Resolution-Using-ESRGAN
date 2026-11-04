"""
Image Quality Assessment Metrics Module
Computes Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM)
following standard Super-Resolution benchmark protocols (DIV2K / Set5 / Set14).
"""

from typing import Tuple, Dict, Optional
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from src.image_processing import ensure_rgb, align_images


def rgb2ycbcr(image: np.ndarray) -> np.ndarray:
    """
    Converts RGB image [0, 255] to standard ITU-R BT.601 YCbCr.
    Extracts the Y (luminance) channel used in traditional CVIP SR literature.
    """
    image_f = image.astype(np.float32)
    # ITU-R BT.601 conversion coefficients
    y = 0.257 * image_f[:, :, 0] + 0.504 * image_f[:, :, 1] + 0.098 * image_f[:, :, 2] + 16.0
    return y


def crop_borders(image: np.ndarray, crop_border: int = 4) -> np.ndarray:
    """
    Crops boundary pixels to remove border artifacts introduced by
    convolution padding and bicubic kernel edge extrapolation.
    """
    if crop_border <= 0:
        return image
    h, w = image.shape[:2]
    if h <= 2 * crop_border or w <= 2 * crop_border:
        return image
    return image[crop_border:h - crop_border, crop_border:w - crop_border]


def calculate_psnr(
    img1: np.ndarray,
    img2: np.ndarray,
    crop_border: int = 4,
    test_y_channel: bool = True
) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) between two images in decibels (dB).

    Formula:
        PSNR = 10 * log10(MAX_I^2 / MSE) = 20 * log10(255 / sqrt(MSE))

    Args:
        img1: First RGB image (H, W, 3) uint8.
        img2: Second RGB image (H, W, 3) uint8.
        crop_border: Border pixels to crop (standard is scale factor = 4).
        test_y_channel: If True, computes PSNR on the Y (luminance) channel
                        following official super-resolution benchmark literature.

    Returns:
        PSNR value in decibels (dB). If images are identical, returns 100.0 dB.
    """
    img1_rgb, img2_rgb = align_images(ensure_rgb(img1), ensure_rgb(img2))
    img1_c = crop_borders(img1_rgb, crop_border)
    img2_c = crop_borders(img2_rgb, crop_border)

    if test_y_channel:
        y1 = rgb2ycbcr(img1_c)
        y2 = rgb2ycbcr(img2_c)
        mse = np.mean((y1 - y2) ** 2)
        if mse < 1e-10:
            return 100.0
        return float(10.0 * np.log10((255.0 ** 2) / mse))
    else:
        # Standard RGB PSNR
        mse = np.mean((img1_c.astype(np.float64) - img2_c.astype(np.float64)) ** 2)
        if mse < 1e-10:
            return 100.0
        return float(10.0 * np.log10((255.0 ** 2) / mse))


def calculate_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    crop_border: int = 4,
    test_y_channel: bool = True
) -> float:
    """
    Computes Structural Similarity Index (SSIM) between two images.
    Quantifies degradation in structural information, luminance, and contrast.

    Args:
        img1: First RGB image (H, W, 3) uint8.
        img2: Second RGB image (H, W, 3) uint8.
        crop_border: Border pixels to crop (standard is scale factor = 4).
        test_y_channel: If True, evaluates on luminance Y channel.

    Returns:
        SSIM value between -1.0 and +1.0 (typically 0.6 to 1.0 for image SR).
    """
    img1_rgb, img2_rgb = align_images(ensure_rgb(img1), ensure_rgb(img2))
    img1_c = crop_borders(img1_rgb, crop_border)
    img2_c = crop_borders(img2_rgb, crop_border)

    if test_y_channel:
        y1 = np.clip(rgb2ycbcr(img1_c), 0, 255).astype(np.float64)
        y2 = np.clip(rgb2ycbcr(img2_c), 0, 255).astype(np.float64)
        val = structural_similarity(y1, y2, data_range=255.0)
        return float(val)
    else:
        val = structural_similarity(img1_c, img2_c, channel_axis=2, data_range=255)
        return float(val)


def is_lpips_available() -> bool:
    """Checks whether the external `lpips` perceptual assessment package is installed."""
    try:
        import lpips
        return True
    except ImportError:
        return False


def calculate_lpips(
    img1: np.ndarray,
    img2: np.ndarray,
    crop_border: int = 4,
    device: str = "cpu"
) -> Optional[float]:
    """
    Computes Learned Perceptual Image Patch Similarity (LPIPS) if `lpips` is installed.
    Lower values indicate higher perceptual similarity (0.0 = identical).
    Returns None gracefully if `lpips` or `torchvision` is not installed in the environment.
    """
    if not is_lpips_available():
        return None

    try:
        import lpips
        import torch

        loss_fn = lpips.LPIPS(net='alex', verbose=False).to(device)
        img1_rgb, img2_rgb = align_images(ensure_rgb(img1), ensure_rgb(img2))
        img1_c = crop_borders(img1_rgb, crop_border)
        img2_c = crop_borders(img2_rgb, crop_border)

        # Scale to [-1, 1] tensor
        t1 = torch.from_numpy(img1_c.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)
        t2 = torch.from_numpy(img2_c.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            dist = loss_fn(t1, t2).item()
        return round(float(dist), 4)
    except Exception:
        return None


def evaluate_pair(
    sr_image: np.ndarray,
    hr_reference: np.ndarray,
    crop_border: int = 4
) -> Dict[str, float]:
    """
    Evaluates both PSNR and SSIM across both Y-channel and RGB representations.

    Returns:
        Dict with keys: 'psnr_y', 'ssim_y', 'psnr_rgb', 'ssim_rgb'
    """
    metrics = {
        "psnr_y": round(calculate_psnr(sr_image, hr_reference, crop_border=crop_border, test_y_channel=True), 2),
        "ssim_y": round(calculate_ssim(sr_image, hr_reference, crop_border=crop_border, test_y_channel=True), 4),
        "psnr_rgb": round(calculate_psnr(sr_image, hr_reference, crop_border=crop_border, test_y_channel=False), 2),
        "ssim_rgb": round(calculate_ssim(sr_image, hr_reference, crop_border=crop_border, test_y_channel=False), 4),
    }
    lpips_val = calculate_lpips(sr_image, hr_reference, crop_border=crop_border)
    if lpips_val is not None:
        metrics["lpips"] = lpips_val
    return metrics


# Alias for backward-compatibility
calculate_metrics = evaluate_pair
