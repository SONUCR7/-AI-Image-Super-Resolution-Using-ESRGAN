"""
ESRGAN Inference Pipeline
Handles tensor formatting, device execution (CPU/CUDA), latency timing,
tiled inference for high-resolution images, and live progress reporting.
"""

import time
import math
from typing import Tuple, Optional, Callable
import numpy as np
import torch
import torch.nn as nn

import config
from src.image_processing import ensure_rgb
from src.model_loader import get_cached_model


def estimate_tiles(
    height: int,
    width: int,
    tile_size: int = 256,
    overlap: int = 24
) -> int:
    """Calculates the total number of sub-tiles required for inference."""
    stride = tile_size - overlap
    rows = max(1, math.ceil((max(0, height - overlap)) / stride)) if height > tile_size else 1
    cols = max(1, math.ceil((max(0, width - overlap)) / stride)) if width > tile_size else 1
    return rows * cols


def _direct_inference(img_tensor: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Runs a single forward pass without gradient tracking using inference_mode."""
    with torch.inference_mode():
        output = model(img_tensor)
    return output


def _tiled_inference(
    img_tensor: torch.Tensor,
    model: nn.Module,
    scale: int = 4,
    tile_size: int = 256,
    overlap: int = 24,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> torch.Tensor:
    """
    Performs memory-efficient tiled inference with linear feathered blending across seams.
    Prevents Out-Of-Memory (OOM) exceptions when processing large images on CPU or limited VRAM.
    """
    batch, channels, height, width = img_tensor.shape
    out_h, out_w = height * scale, width * scale

    device = img_tensor.device
    output_tensor = torch.zeros((batch, channels, out_h, out_w), device=device, dtype=torch.float32)
    weight_tensor = torch.zeros((batch, 1, out_h, out_w), device=device, dtype=torch.float32)

    stride = tile_size - overlap

    # Precompute tile boundaries
    tiles = []
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            tiles.append((y_start, y_end, x_start, x_end))

    total_tiles = len(tiles)

    for idx, (y_start, y_end, x_start, x_end) in enumerate(tiles, 1):
        if progress_callback:
            progress_callback(
                idx,
                total_tiles,
                f"Processing tile {idx}/{total_tiles} ({int(idx / total_tiles * 100)}%)..."
            )

        tile_in = img_tensor[:, :, y_start:y_end, x_start:x_end]

        with torch.inference_mode():
            tile_out = model(tile_in)

        out_y_start = y_start * scale
        out_y_end = y_end * scale
        out_x_start = x_start * scale
        out_x_end = x_end * scale

        t_h = out_y_end - out_y_start
        t_w = out_x_end - out_x_start

        # Linear weight mask for smooth feathering
        y_weights = torch.ones(t_h, device=device)
        x_weights = torch.ones(t_w, device=device)

        ramp = overlap * scale
        if y_start > 0 and ramp > 0:
            y_weights[:ramp] = torch.linspace(0.01, 1.0, ramp, device=device)
        if y_end < height and ramp > 0:
            y_weights[-ramp:] = torch.linspace(1.0, 0.01, ramp, device=device)

        if x_start > 0 and ramp > 0:
            x_weights[:ramp] = torch.linspace(0.01, 1.0, ramp, device=device)
        if x_end < width and ramp > 0:
            x_weights[-ramp:] = torch.linspace(1.0, 0.01, ramp, device=device)

        tile_weight = (y_weights[:, None] * x_weights[None, :]).unsqueeze(0).unsqueeze(0)

        output_tensor[:, :, out_y_start:out_y_end, out_x_start:out_x_end] += tile_out * tile_weight
        weight_tensor[:, :, out_y_start:out_y_end, out_x_start:out_x_end] += tile_weight

    # Normalize overlapping regions
    output_tensor /= torch.clamp(weight_tensor, min=1e-6)
    return output_tensor


def super_resolve(
    image: np.ndarray,
    model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    use_tiling: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Tuple[np.ndarray, float]:
    """
    Applies 4x deep learning super-resolution using the ESRGAN RRDBNet model.

    Args:
        image: RGB input image as numpy array (H, W, 3), dtype uint8.
        model: Loaded RRDBNet model. If None, uses cached singleton.
        device: Device to run inference on ('cpu' or 'cuda'). If None, uses config.DEVICE.
        use_tiling: Whether to force tiled inference. If False, tiling is used automatically
                    only if image dimensions exceed memory thresholds on CPU.
        progress_callback: Optional callback(current_tile, total_tiles, message) for UI progress.

    Returns:
        Tuple of (sr_image: np.ndarray [H*4, W*4, 3] uint8, elapsed_time_seconds: float)
    """
    rgb_img = ensure_rgb(image)
    h, w = rgb_img.shape[:2]

    # Model and device resolution
    target_device = device if device is not None else config.get_device()
    sr_model = model if model is not None else get_cached_model(str(target_device))
    sr_model.eval()

    # Preprocessing: Normalization to [0.0, 1.0], transpose to (1, C, H, W)
    img_norm = rgb_img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(np.transpose(img_norm, (2, 0, 1))).unsqueeze(0).to(target_device)

    # Determine execution strategy (direct vs tiled)
    is_cpu = target_device.type == "cpu"
    requires_tiling = use_tiling or (is_cpu and (h > 384 or w > 384))

    start_time = time.perf_counter()

    if requires_tiling:
        out_tensor = _tiled_inference(
            img_tensor,
            sr_model,
            scale=config.SCALE_FACTOR,
            tile_size=256,
            overlap=20,
            progress_callback=progress_callback
        )
    else:
        if progress_callback:
            progress_callback(1, 1, "Executing ESRGAN forward pass...")
        out_tensor = _direct_inference(img_tensor, sr_model)

    elapsed_time = time.perf_counter() - start_time

    # Postprocessing: Clamp, transfer to CPU, transpose to (H, W, C), scale to [0, 255]
    out_np = out_tensor.squeeze(0).clamp(0.0, 1.0).cpu().numpy()
    out_np = np.transpose(out_np, (1, 2, 0))
    sr_image = np.round(out_np * 255.0).astype(np.uint8)

    return sr_image, elapsed_time
