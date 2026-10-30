"""
Model Loader for ESRGAN
Handles checkpoint verification, downloading, state dict adaptation,
device placement (CPU/CUDA), and singleton in-memory caching.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

import config
from src.rrdb_net import RRDBNet
from models.download_weights import ensure_weights

# Global cache for the loaded model singleton
_MODEL_CACHE: Dict[str, nn.Module] = {}


def clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleans and standardizes state_dict keys from various ESRGAN checkpoints.
    Handles 'params', 'params_ema', 'module.', or 'model.' prefixes.
    """
    if "params" in state_dict:
        state_dict = state_dict["params"]
    elif "params_ema" in state_dict:
        state_dict = state_dict["params_ema"]

    cleaned_dict = {}
    for key, value in state_dict.items():
        # Remove common wrapper prefixes
        clean_key = key
        if clean_key.startswith("module."):
            clean_key = clean_key[7:]
        if clean_key.startswith("model."):
            clean_key = clean_key[6:]
        cleaned_dict[clean_key] = value

    return cleaned_dict


def load_esrgan_model(
    weights_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    download_if_missing: bool = True
) -> nn.Module:
    """
    Loads and initializes the RRDBNet model with pretrained ESRGAN weights.

    Args:
        weights_path: Path to .pth weights file. Defaults to config.MODEL_WEIGHTS_PATH.
        device: PyTorch device ('cpu' or 'cuda'). Defaults to config.DEVICE.
        download_if_missing: If True, attempts automatic download if weights are absent.

    Returns:
        Loaded, evaluated, and frozen RRDBNet PyTorch model.

    Raises:
        FileNotFoundError: If weights file is not found and cannot be downloaded.
        RuntimeError: If checkpoint cannot be loaded or architecture doesn't match.
    """
    path = Path(weights_path) if weights_path else config.MODEL_WEIGHTS_PATH
    target_device = device if device is not None else config.get_device()

    # Ensure weights are present
    if not path.exists() or path.stat().st_size < 10 * 1024 * 1024:
        if download_if_missing:
            success = ensure_weights(destination=path)
            if not success or not path.exists():
                raise FileNotFoundError(
                    f"Model weights file not found at '{path}'. "
                    "Automatic download was unsuccessful. Please check your network connection "
                    "or run 'python models/download_weights.py'."
                )
        else:
            raise FileNotFoundError(
                f"Model weights file not found at '{path}'. "
                "Download it by running 'python models/download_weights.py'."
            )

    # Initialize RRDBNet architecture
    model = RRDBNet(
        in_nc=config.RRDB_CONFIG["in_nc"],
        out_nc=config.RRDB_CONFIG["out_nc"],
        nf=config.RRDB_CONFIG["nf"],
        nb=config.RRDB_CONFIG["nb"],
        gc=config.RRDB_CONFIG["gc"]
    )

    try:
        # Load weights with map_location
        checkpoint = torch.load(path, map_location=target_device, weights_only=True)
    except Exception:
        # Fallback for older PyTorch versions or checkpoints with non-tensor metadata
        checkpoint = torch.load(path, map_location=target_device, weights_only=False)

    cleaned_dict = clean_state_dict(checkpoint)

    # Load weights into model
    try:
        model.load_state_dict(cleaned_dict, strict=True)
    except RuntimeError as e:
        # Attempt non-strict loading if minor keys differ, but verify majority loaded
        missing, unexpected = model.load_state_dict(cleaned_dict, strict=False)
        if len(missing) > 5:
            raise RuntimeError(
                f"Checkpoint at {path} could not be matched to RRDBNet architecture: {e}"
            )

    model.to(target_device)
    model.eval()

    # Freeze all parameters for inference efficiency
    for param in model.parameters():
        param.requires_grad = False

    return model


def get_cached_model(device_str: Optional[str] = None) -> nn.Module:
    """
    Retrieves or caches the ESRGAN model singleton in memory.
    Ensures the model is only loaded from disk once during application lifecycle.
    """
    selected_device = torch.device(device_str) if device_str else config.get_device()
    cache_key = str(selected_device)

    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = load_esrgan_model(device=selected_device)

    return _MODEL_CACHE[cache_key]
