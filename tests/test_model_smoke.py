"""
Unit and Integration Tests for ESRGAN Model Architecture & Inference
"""

import pytest
import numpy as np
import torch

import config
from src.rrdb_net import RRDBNet
from src.model_loader import load_esrgan_model
from src.inference import super_resolve


def test_rrdbnet_architecture_layers_and_params():
    """Verify RRDBNet structure and parameter count (~16.7M for 23 blocks)."""
    model = RRDBNet(
        in_nc=config.RRDB_CONFIG["in_nc"],
        out_nc=config.RRDB_CONFIG["out_nc"],
        nf=config.RRDB_CONFIG["nf"],
        nb=config.RRDB_CONFIG["nb"],
        gc=config.RRDB_CONFIG["gc"]
    )
    total_params = sum(p.numel() for p in model.parameters())
    # Standard ESRGAN has ~16,697,987 parameters
    assert 16_000_000 < total_params < 17_500_000

    # Test forward pass with small random dummy tensor
    dummy_input = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (1, 3, 64, 64)


def test_model_loading_and_inference():
    """Verify loading real weights checkpoint and running inference on an RGB image."""
    weights_path = config.MODEL_WEIGHTS_PATH
    if not weights_path.exists():
        pytest.skip("Pretrained weights not available on disk.")

    model = load_esrgan_model(weights_path=weights_path, download_if_missing=False)
    assert isinstance(model, torch.nn.Module)

    # Run super_resolve on 32x32 image
    test_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    sr_img, elapsed = super_resolve(test_img, model=model)

    assert sr_img.shape == (128, 128, 3)
    assert sr_img.dtype == np.uint8
    assert elapsed > 0.0
