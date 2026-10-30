"""
Unit Tests for Input Validation and Image Verification
"""

import io
import pytest
import numpy as np
from PIL import Image

import config
from src.validation import validate_image_file


def create_test_image_bytes(width: int = 100, height: int = 100, fmt: str = "PNG", mode: str = "RGB") -> bytes:
    """Helper to generate in-memory image bytes."""
    if mode == "RGB":
        arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    elif mode == "L":
        arr = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    elif mode == "RGBA":
        arr = np.random.randint(0, 255, (height, width, 4), dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_valid_png_image():
    """Test validation with a standard valid PNG image."""
    img_bytes = create_test_image_bytes(128, 96, "PNG", "RGB")
    is_valid, err, arr, meta = validate_image_file(img_bytes, filename="test.png")

    assert is_valid is True
    assert err is None
    assert arr is not None
    assert arr.shape == (96, 128, 3)
    assert meta["width"] == 128
    assert meta["height"] == 96
    assert meta["target_width_4x"] == 512
    assert meta["target_height_4x"] == 384


def test_valid_jpeg_and_webp_images():
    """Test validation with JPEG and WebP formats."""
    jpg_bytes = create_test_image_bytes(80, 80, "JPEG", "RGB")
    is_valid_jpg, err_jpg, arr_jpg, _ = validate_image_file(jpg_bytes, filename="sample.jpg")
    assert is_valid_jpg is True
    assert err_jpg is None
    assert arr_jpg.shape == (80, 80, 3)

    webp_bytes = create_test_image_bytes(64, 64, "WEBP", "RGB")
    is_valid_webp, err_webp, arr_webp, _ = validate_image_file(webp_bytes, filename="sample.webp")
    assert is_valid_webp is True
    assert err_webp is None
    assert arr_webp.shape == (64, 64, 3)


def test_empty_file():
    """Test rejection of empty 0-byte file."""
    is_valid, err, arr, meta = validate_image_file(b"", filename="empty.png")
    assert is_valid is False
    assert "empty" in err.lower()
    assert arr is None


def test_corrupted_image_stream():
    """Test detection and graceful rejection of corrupt byte stream."""
    corrupted_bytes = b"GIF89a\x00\x00\x00\x00CORRUPT_PAYLOAD_GARBAGE"
    is_valid, err, arr, meta = validate_image_file(corrupted_bytes, filename="corrupt.gif")
    assert is_valid is False
    assert err is not None
    assert arr is None


def test_unsupported_extension():
    """Test rejection of unsupported file extensions."""
    img_bytes = create_test_image_bytes(50, 50, "PNG")
    is_valid, err, arr, meta = validate_image_file(img_bytes, filename="document.pdf")
    assert is_valid is False
    assert "unsupported" in err.lower()


def test_dimension_lower_bound():
    """Test rejection of images smaller than MIN_IMAGE_DIMENSION."""
    tiny_bytes = create_test_image_bytes(8, 8, "PNG")
    is_valid, err, arr, meta = validate_image_file(tiny_bytes, filename="tiny.png", min_dimension=16)
    assert is_valid is False
    assert "too small" in err.lower()


def test_dimension_upper_bound():
    """Test rejection of images exceeding MAX_IMAGE_DIMENSION."""
    img_bytes = create_test_image_bytes(100, 100, "PNG")
    is_valid, err, arr, meta = validate_image_file(img_bytes, filename="large.png", max_dimension=50)
    assert is_valid is False
    assert "exceeds maximum" in err.lower()


def test_oversized_file():
    """Test rejection when file size exceeds MAX_IMAGE_SIZE_MB."""
    img_bytes = create_test_image_bytes(100, 100, "PNG")
    # Simulate a tiny max_size threshold
    is_valid, err, arr, meta = validate_image_file(img_bytes, filename="big.png", max_size_mb=0.001)
    assert is_valid is False
    assert "exceeds maximum allowed limit" in err.lower()
