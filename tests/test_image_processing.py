"""
Unit Tests for Image Processing Module
Tests bicubic interpolation, RGB standardization, degradation, and alignment.
"""

import pytest
import numpy as np

import config
from src.image_processing import (
    ensure_rgb,
    bicubic_upscale,
    degrade_image,
    align_images
)


def test_ensure_rgb_from_grayscale():
    """Verify 2D grayscale array converts to (H, W, 3)."""
    gray = np.full((64, 64), 128, dtype=np.uint8)
    rgb = ensure_rgb(gray)
    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == np.uint8
    assert np.all(rgb[:, :, 0] == 128)
    assert np.all(rgb[:, :, 1] == 128)
    assert np.all(rgb[:, :, 2] == 128)


def test_ensure_rgb_from_rgba():
    """Verify 4-channel RGBA converts cleanly to RGB with alpha compositing."""
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    # Red with 50% opacity
    rgba[:, :, 0] = 200
    rgba[:, :, 3] = 128  # ~0.5 alpha over white (255)
    rgb = ensure_rgb(rgba)
    assert rgb.shape == (32, 32, 3)
    assert rgb.dtype == np.uint8
    # Composite should be ~ 200 * 0.5 + 255 * 0.5 = 227
    assert 220 <= rgb[0, 0, 0] <= 235


def test_ensure_rgb_preserves_standard_rgb():
    """Verify 3-channel RGB image remains unaltered."""
    img = np.random.randint(0, 256, (40, 50, 3), dtype=np.uint8)
    rgb = ensure_rgb(img)
    assert rgb.shape == (40, 50, 3)
    assert np.array_equal(img, rgb)


def test_bicubic_upscale_dimensions_and_aspect_ratio():
    """Verify 4x scaling correctly multiplies dimensions and preserves aspect ratio."""
    h, w = 35, 70
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    upscaled, elapsed = bicubic_upscale(img, scale=4)

    assert upscaled.shape == (h * 4, w * 4, 3)
    assert upscaled.dtype == np.uint8
    assert elapsed >= 0.0
    original_ar = w / h
    upscaled_ar = upscaled.shape[1] / upscaled.shape[0]
    assert abs(original_ar - upscaled_ar) < 1e-5


def test_degrade_image_downsamples_by_scale():
    """Verify degrade_image produces accurate 4x downscaled counterpart."""
    hr = np.random.randint(0, 256, (128, 256, 3), dtype=np.uint8)
    lr = degrade_image(hr, scale=4)

    assert lr.shape == (32, 64, 3)
    assert lr.dtype == np.uint8


def test_align_images():
    """Verify alignment crops images with minor pixel difference to matching dimensions."""
    img1 = np.ones((100, 102, 3), dtype=np.uint8)
    img2 = np.ones((99, 100, 3), dtype=np.uint8)

    a1, a2 = align_images(img1, img2)
    assert a1.shape == (99, 100, 3)
    assert a2.shape == (99, 100, 3)
