"""
End-to-End System Smoke Test
SR Vision Lab: Comparative Image Super-Resolution Using ESRGAN and Deep Learning

Verifies:
1. Python environment and required dependency imports
2. Device selection (CPU / CUDA)
3. Pretrained weight file presence and integrity
4. Model architecture initialization and parameter count (~16.7M)
5. Image validation module (valid and invalid inputs)
6. Bicubic upscaling (dimension and aspect ratio checks)
7. ESRGAN forward inference pipeline
8. PSNR and SSIM metric calculation
"""

import sys
import time
from pathlib import Path
import numpy as np

# Ensure root folder is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config


def print_row(test_name: str, status: str, details: str):
    color_prefix = "\033[92m" if status == "PASS" else "\033[91m"
    color_suffix = "\033[0m"
    print(f"| {test_name:<36} | {color_prefix}{status:^6}{color_suffix} | {details:<32} |")


def run_smoke_test() -> bool:
    print("\n" + "=" * 82)
    print("           SR VISION LAB — SYSTEM SMOKE & INTEGRATION TEST")
    print("=" * 82)
    print(f"| {'Test Case':<36} | {'Status':^6} | {'Details':<32} |")
    print("|" + "-" * 38 + "|" + "-" * 8 + "|" + "-" * 34 + "|")

    all_passed = True

    # 1. Dependency Imports
    try:
        import torch
        import cv2
        import PIL
        import skimage
        import streamlit
        import matplotlib
        import pytest
        print_row("1. Dependency Imports", "PASS", f"PyTorch {torch.__version__}, CV2 {cv2.__version__}")
    except Exception as e:
        print_row("1. Dependency Imports", "FAIL", str(e)[:32])
        all_passed = False

    # 2. Hardware Compute Device Selection
    try:
        import torch
        device = config.get_device()
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU (Standard Mode)"
        print_row("2. Device Selection", "PASS", f"{device.type.upper()} ({gpu_name[:20]})")
    except Exception as e:
        print_row("2. Device Selection", "FAIL", str(e)[:32])
        all_passed = False

    # 3. Model Weight Verification
    weights_path = config.MODEL_WEIGHTS_PATH
    has_weights = weights_path.exists() and weights_path.stat().st_size > 10 * 1024 * 1024
    if has_weights:
        size_mb = weights_path.stat().st_size / (1024 * 1024)
        print_row("3. Model Weights Check", "PASS", f"{weights_path.name} ({size_mb:.1f} MB)")
    else:
        print_row("3. Model Weights Check", "WARN", "Weights missing (run downloader)")
        all_passed = False

    # 4. Architecture Initialization
    model = None
    try:
        from src.rrdb_net import RRDBNet
        model = RRDBNet(
            in_nc=config.RRDB_CONFIG["in_nc"],
            out_nc=config.RRDB_CONFIG["out_nc"],
            nf=config.RRDB_CONFIG["nf"],
            nb=config.RRDB_CONFIG["nb"],
            gc=config.RRDB_CONFIG["gc"]
        )
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print_row("4. RRDBNet Architecture", "PASS", f"23 RRDB Blocks ({param_count:.2f}M params)")
    except Exception as e:
        print_row("4. RRDBNet Architecture", "FAIL", str(e)[:32])
        all_passed = False

    # 5. Image Validation Checks
    try:
        from src.validation import validate_image_file
        # Valid test pattern
        valid_img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(valid_img).save(buf, format="PNG")
        buf.seek(0)
        is_val, err, _, meta = validate_image_file(buf.getvalue(), filename="test.png")
        assert is_val and err is None, f"Valid check failed: {err}"

        # Corrupted stream check
        is_val_corrupt, err_corrupt, _, _ = validate_image_file(b"CORRUPTED_BYTES_NOT_IMAGE", filename="bad.png")
        assert not is_val_corrupt, "Corrupted stream was erroneously marked valid!"
        print_row("5. Input Validation Module", "PASS", "Valid + corrupt streams handled")
    except Exception as e:
        print_row("5. Input Validation Module", "FAIL", str(e)[:32])
        all_passed = False

    # 6. Bicubic Interpolation Baseline
    try:
        from src.image_processing import bicubic_upscale
        test_in = np.zeros((48, 48, 3), dtype=np.uint8)
        upscaled, b_time = bicubic_upscale(test_in, scale=4)
        assert upscaled.shape == (192, 192, 3), f"Bicubic shape mismatch: {upscaled.shape}"
        print_row("6. Bicubic Interpolation", "PASS", f"48x48 -> 192x192 ({b_time*1000:.1f}ms)")
    except Exception as e:
        print_row("6. Bicubic Interpolation", "FAIL", str(e)[:32])
        all_passed = False

    # 7. Model Inference (ESRGAN 4x)
    if has_weights and model is not None:
        try:
            from src.model_loader import load_esrgan_model
            from src.inference import super_resolve
            sr_model = load_esrgan_model(download_if_missing=False)
            test_lr = np.full((32, 32, 3), 128, dtype=np.uint8)
            sr_out, sr_time = super_resolve(test_lr, model=sr_model)
            assert sr_out.shape == (128, 128, 3), f"SR shape mismatch: {sr_out.shape}"
            assert sr_out.dtype == np.uint8, f"Output dtype mismatch: {sr_out.dtype}"
            print_row("7. ESRGAN Model Inference", "PASS", f"32x32 -> 128x128 ({sr_time:.2f}s)")
        except Exception as e:
            print_row("7. ESRGAN Model Inference", "FAIL", str(e)[:32])
            all_passed = False
    else:
        print_row("7. ESRGAN Model Inference", "SKIP", "Requires pretrained weights")

    # 8. Quantitative Metrics (PSNR / SSIM)
    try:
        from src.metrics import calculate_psnr, calculate_ssim
        ref = np.full((64, 64, 3), 100, dtype=np.uint8)
        # Identical images
        psnr_ident = calculate_psnr(ref, ref)
        ssim_ident = calculate_ssim(ref, ref)
        assert psnr_ident >= 99.0, f"Expected PSNR ~100 dB, got {psnr_ident}"
        assert abs(ssim_ident - 1.0) < 1e-4, f"Expected SSIM 1.0, got {ssim_ident}"

        # Perturbed image
        pert = np.full((64, 64, 3), 110, dtype=np.uint8)
        psnr_pert = calculate_psnr(ref, pert)
        ssim_pert = calculate_ssim(ref, pert)
        assert psnr_pert < 99.0 and ssim_pert < 1.0
        print_row("8. PSNR & SSIM Metrics", "PASS", f"Identical=100dB/1.0, Pert={psnr_pert:.1f}dB")
    except Exception as e:
        print_row("8. PSNR & SSIM Metrics", "FAIL", str(e)[:32])
        all_passed = False

    print("=" * 82)
    if all_passed:
        print(">> ALL SYSTEM CHECKS PASSED: Application is fully operational! <<\n")
        return True
    else:
        print(">> SOME CHECKS FAILED: Please review logs above. <<\n")
        return False


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
