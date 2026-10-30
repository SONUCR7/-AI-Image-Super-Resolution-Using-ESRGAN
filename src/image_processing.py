"""
Image Processing Module
Traditional CVIP operations: Bicubic interpolation, RGB standardization,
alpha channel handling, synthetic downsampling, and spatial alignment.
"""

import time
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import cv2
from PIL import Image


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """
    Standardizes any input image array into an 8-bit 3-channel RGB image.
    - Converts 2D grayscale (H, W) -> (H, W, 3) by channel replication.
    - Converts RGBA (H, W, 4) -> (H, W, 3) via compositing over a solid white background.
    - Clamps values and enforces uint8 dtype.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    if image.ndim == 2:
        # Grayscale single-channel
        return np.stack([image, image, image], axis=-1).astype(np.uint8)

    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return np.repeat(image, 3, axis=2).astype(np.uint8)
        elif channels == 3:
            return image.astype(np.uint8)
        elif channels == 4:
            # Alpha composite onto white background
            rgb = image[:, :, :3].astype(np.float32)
            alpha = (image[:, :, 3:4].astype(np.float32)) / 255.0
            composite = rgb * alpha + 255.0 * (1.0 - alpha)
            return np.clip(composite, 0, 255).astype(np.uint8)

    raise ValueError(f"Unsupported image shape for RGB conversion: {image.shape}")


def bicubic_upscale(image: np.ndarray, scale: int = 4) -> Tuple[np.ndarray, float]:
    """
    Upscales an input image using classical Bicubic (cubic spline) interpolation.

    Bicubic interpolation computes new pixel values from the weighted average
    of the nearest 4x4 (16) pixel neighborhood using a cubic polynomial kernel.
    Uses standard anti-aliased bicubic interpolation (matching DIV2K / Set5 literature).

    Args:
        image: Input RGB numpy array (H, W, 3), uint8.
        scale: Upscaling factor (default 4).

    Returns:
        Tuple of (upscaled_image: np.ndarray, elapsed_time_seconds: float)
    """
    rgb_img = ensure_rgb(image)
    h, w = rgb_img.shape[:2]
    target_w = w * scale
    target_h = h * scale

    start_time = time.perf_counter()
    pil_img = Image.fromarray(rgb_img)
    upscaled_pil = pil_img.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
    upscaled = np.array(upscaled_pil, dtype=np.uint8)
    elapsed = time.perf_counter() - start_time

    return upscaled, elapsed


def degrade_image(hr_image: np.ndarray, scale: int = 4) -> np.ndarray:
    """
    Generates a low-resolution (LR) counterpart from a high-resolution (HR) reference image.
    Applies the standard super-resolution benchmark degradation protocol (matching DIV2K / NTIRE / Set5 / Set14):
    continuous bicubic downsampling with an anti-aliasing low-pass filter by factor `scale`.

    Note: OpenCV's cv2.resize lacks an anti-aliasing filter during downsampling, introducing
    artificial aliasing and sub-pixel phase shifts. Pillow's Image.Resampling.BICUBIC applies
    the standard continuous anti-aliasing kernel expected by pretrained super-resolution models.

    Args:
        hr_image: High-resolution RGB numpy array (H, W, 3).
        scale: Downsampling factor (default 4).

    Returns:
        Downsampled low-resolution RGB image.
    """
    rgb_img = ensure_rgb(hr_image)
    h, w = rgb_img.shape[:2]
    lr_w = max(1, w // scale)
    lr_h = max(1, h // scale)

    pil_hr = Image.fromarray(rgb_img)
    pil_lr = pil_hr.resize((lr_w, lr_h), resample=Image.Resampling.BICUBIC)
    return np.array(pil_lr, dtype=np.uint8)


def align_images(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aligns two images to identical spatial dimensions by center-cropping or trimming
    any minor pixel difference caused by fractional dimensions during downsampling/upsampling.

    Args:
        img1: First image array (H1, W1, C).
        img2: Second image array (H2, W2, C).

    Returns:
        Tuple of (aligned_img1, aligned_img2) with identical (H, W, C).
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    min_h = min(h1, h2)
    min_w = min(w1, w2)

    return img1[:min_h, :min_w], img2[:min_h, :min_w]


def load_image(file_source: Union[str, Path, bytes]) -> np.ndarray:
    """
    Safely loads an image file or byte stream into an RGB uint8 numpy array.
    """
    if isinstance(file_source, (str, Path)):
        img = Image.open(file_source)
    else:
        import io
        img = Image.open(io.BytesIO(file_source))

    # Convert to RGB (handles palette, grayscale, RGBA)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)


def save_image(image: np.ndarray, save_path: Union[str, Path]) -> None:
    """
    Saves an RGB numpy array to disk in PNG format.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(ensure_rgb(image), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_bgr)
