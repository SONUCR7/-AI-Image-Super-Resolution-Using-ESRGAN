"""
Input Validation and Verification Module
Ensures secure, robust image handling, file format checks, dimension bounds,
and detection of corrupted or unsupported image streams.
"""

import io
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
from PIL import Image, UnidentifiedImageError

import config


def validate_image_file(
    file_source: Union[bytes, io.BytesIO, str, Path, Any],
    filename: Optional[str] = None,
    max_size_mb: float = config.MAX_IMAGE_SIZE_MB,
    max_dimension: int = 4096,
    min_dimension: int = 8
) -> Tuple[bool, Optional[str], Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Validates uploaded file data for image integrity, extension, size, and dimensions.
    """
    # 1. Read raw bytes safely
    try:
        if isinstance(file_source, (str, Path)):
            path = Path(file_source)
            if not path.exists():
                return False, f"File not found: {path}", None, None
            raw_bytes = path.read_bytes()
            if filename is None:
                filename = path.name
        elif hasattr(file_source, "getvalue"):
            # Streamlit UploadedFile or BytesIO
            raw_bytes = file_source.getvalue()
            if hasattr(file_source, "name") and filename is None:
                filename = file_source.name
        elif hasattr(file_source, "read"):
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            raw_bytes = file_source.read()
            if hasattr(file_source, "name") and filename is None:
                filename = file_source.name
        elif isinstance(file_source, bytes):
            raw_bytes = file_source
        else:
            return False, "Unsupported file source type.", None, None
    except Exception as e:
        return False, f"Failed to read file stream: {str(e)}", None, None

    # 2. File size verification
    size_bytes = len(raw_bytes)
    size_mb = size_bytes / (1024 * 1024)
    if size_bytes == 0:
        return False, "Uploaded file is empty (0 bytes).", None, None

    if size_mb > max_size_mb:
        return (
            False,
            f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit of {max_size_mb:.1f} MB.",
            None,
            None
        )

    # 3. Filename extension verification
    if filename:
        ext = Path(filename).suffix.lower()
        if ext and ext not in config.SUPPORTED_EXTENSIONS:
            allowed = ", ".join(config.SUPPORTED_EXTENSIONS)
            return (
                False,
                f"Unsupported file format '{ext}'. Allowed formats: {allowed}",
                None,
                None
            )

    # 4. Image integrity and full decode
    try:
        decode_stream = io.BytesIO(raw_bytes)
        with Image.open(decode_stream) as pil_img:
            width, height = pil_img.size
            mode = pil_img.mode

            if mode != "RGB":
                pil_img = pil_img.convert("RGB")
            img_array = np.array(pil_img, dtype=np.uint8)

    except UnidentifiedImageError:
        return False, "Uploaded file is not a valid or recognizable image.", None, None
    except Exception as e:
        return False, f"Image decoding error: {str(e)}", None, None

    # 5. Dimension checks
    if width < min_dimension or height < min_dimension:
        return (
            False,
            f"Image resolution ({width}x{height}) is too small. Minimum dimension is {min_dimension}px.",
            None,
            None
        )

    if width > max_dimension or height > max_dimension:
        return (
            False,
            f"Image resolution ({width}x{height}) exceeds maximum supported dimension of {max_dimension}x{max_dimension} px.",
            None,
            None
        )

    aspect_ratio = width / max(1, height)

    metadata = {
        "filename": filename or "uploaded_image",
        "format": "PNG",
        "original_mode": mode,
        "width": width,
        "height": height,
        "aspect_ratio": round(aspect_ratio, 3),
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 2),
        "target_width_4x": width * config.SCALE_FACTOR,
        "target_height_4x": height * config.SCALE_FACTOR,
    }

    return True, None, img_array, metadata
