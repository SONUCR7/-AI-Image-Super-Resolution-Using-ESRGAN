"""
Unit Tests for Quantitative Metrics Module (PSNR & SSIM)
"""

import pytest
import numpy as np

from src.metrics import (
    calculate_psnr,
    calculate_ssim,
    crop_borders,
    evaluate_pair
)


def test_crop_borders():
    """Verify border cropping shaves edges properly."""
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    cropped = crop_borders(img, crop_border=4)
    assert cropped.shape == (12, 12, 3)

    # If image is smaller than 2*crop_border, should return unchanged
    small = np.zeros((6, 6, 3), dtype=np.uint8)
    assert crop_borders(small, crop_border=4).shape == (6, 6, 3)


def test_psnr_identical_images():
    """Identical images must yield a maximum capped PSNR (100.0 dB)."""
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    psnr_y = calculate_psnr(img, img, crop_border=0, test_y_channel=True)
    psnr_rgb = calculate_psnr(img, img, crop_border=0, test_y_channel=False)

    assert psnr_y >= 99.9
    assert psnr_rgb >= 99.9


def test_ssim_identical_images():
    """Identical images must yield an SSIM of exactly 1.0."""
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    ssim_y = calculate_ssim(img, img, crop_border=0, test_y_channel=True)
    ssim_rgb = calculate_ssim(img, img, crop_border=0, test_y_channel=False)

    assert abs(ssim_y - 1.0) < 1e-4
    assert abs(ssim_rgb - 1.0) < 1e-4


def test_psnr_decreases_with_increasing_noise():
    """Verify PSNR decreases monotonically with increased degradation."""
    clean = np.full((64, 64, 3), 128, dtype=np.uint8)
    slight_diff = np.full((64, 64, 3), 130, dtype=np.uint8)  # Delta = 2
    large_diff = np.full((64, 64, 3), 160, dtype=np.uint8)   # Delta = 32

    psnr_slight = calculate_psnr(clean, slight_diff, crop_border=0)
    psnr_large = calculate_psnr(clean, large_diff, crop_border=0)

    assert psnr_slight > psnr_large
    assert psnr_large > 0.0


def test_evaluate_pair_dictionary_format():
    """Verify evaluate_pair returns dictionary with expected keys and rounded floats."""
    img1 = np.full((64, 64, 3), 100, dtype=np.uint8)
    img2 = np.full((64, 64, 3), 110, dtype=np.uint8)

    results = evaluate_pair(img1, img2, crop_border=2)
    assert "psnr_y" in results
    assert "ssim_y" in results
    assert "psnr_rgb" in results
    assert "ssim_rgb" in results
    assert isinstance(results["psnr_y"], float)
    assert isinstance(results["ssim_y"], float)
