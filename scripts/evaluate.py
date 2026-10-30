"""
Quantitative Super-Resolution Evaluation Benchmark Pipeline
SR Vision Lab: Comparative Image Super-Resolution Using ESRGAN and Deep Learning

Evaluates Bicubic Interpolation vs. ESRGAN (4x) on high-resolution reference images:
- Synthesizes 4x downsampled inputs using standard degradation
- Reconstructs inputs using both approaches
- Computes PSNR and SSIM on Y-channel (luminance) and RGB
- Measures runtime and resolution scaling
- Exports detailed per-image CSV and aggregate visual charts
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt

# Ensure root folder is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from src.image_processing import load_image, ensure_rgb, bicubic_upscale, degrade_image, align_images
from src.metrics import calculate_psnr, calculate_ssim, calculate_lpips, is_lpips_available
from src.model_loader import get_cached_model, load_esrgan_model
from src.inference import super_resolve


def run_benchmark(
    hr_dir: Path = config.EVAL_HR_DIR,
    output_dir: Path = config.OUTPUTS_DIR,
    device_str: Optional[str] = None,
    save_images: bool = True,
    progress_callback = None
) -> Dict[str, Any]:
    """
    Executes the complete reproducible quantitative benchmark.

    Args:
        hr_dir: Directory containing high-resolution reference images.
        output_dir: Directory to save CSV, charts, and summary reports.
        device_str: Device to run ESRGAN on ('cpu' or 'cuda').
        save_images: Whether to save intermediate LR, Bicubic, and ESRGAN images.
        progress_callback: Optional callable(current, total, filename) for progress tracking.

    Returns:
        Dictionary containing summary statistics, per-image records, and output paths.
    """
    hr_path = Path(hr_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    images_out_dir = out_path / "benchmark_visuals"
    if save_images:
        images_out_dir.mkdir(parents=True, exist_ok=True)

    if not hr_path.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {hr_path}")

    # Gather reference images
    supported_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    image_files = sorted([f for f in hr_path.iterdir() if f.suffix.lower() in supported_exts])

    if not image_files:
        raise ValueError(f"No valid reference images found in {hr_path}.")

    print(f"===============================================================")
    print(f"  SR VISION LAB: REPRODUCIBLE EVALUATION BENCHMARK PIPELINE   ")
    print(f"===============================================================")
    print(f"Reference images directory : {hr_path}")
    print(f"Total reference images     : {len(image_files)}")
    print(f"Scale factor               : {config.SCALE_FACTOR}x")
    device = config.get_device() if device_str is None else config.torch.device(device_str)
    print(f"Compute hardware device    : {device}")
    print(f"ESRGAN Model weights       : {config.MODEL_WEIGHTS_PATH.name}")
    print(f"---------------------------------------------------------------")

    # Load ESRGAN model
    print("Loading ESRGAN generator model into memory...")
    model = get_cached_model(str(device))
    print("Model initialized successfully.\n")

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    total_images = len(image_files)

    for idx, img_file in enumerate(image_files, 1):
        if progress_callback:
            progress_callback(idx, total_images, img_file.name)

        print(f"[{idx}/{total_images}] Processing: {img_file.name} ... ", end="", flush=True)

        try:
            # 1. Load HR ground truth reference
            hr_img = load_image(img_file)
            hr_h, hr_w = hr_img.shape[:2]

            # 2. Synthesize degraded LR input (4x bicubic downsampling)
            lr_img = degrade_image(hr_img, scale=config.SCALE_FACTOR)
            lr_h, lr_w = lr_img.shape[:2]

            # 3. Traditional baseline: Bicubic interpolation
            bicubic_img, bicubic_time = bicubic_upscale(lr_img, scale=config.SCALE_FACTOR)

            # 4. Deep Learning: ESRGAN 4x super-resolution
            esrgan_img, esrgan_time = super_resolve(lr_img, model=model, device=device)

            # 5. Spatial alignment (ensures pixel-accurate metric alignment)
            bicubic_aligned, hr_aligned = align_images(bicubic_img, hr_img)
            esrgan_aligned, _ = align_images(esrgan_img, hr_img)

            # 6. Calculate quantitative metrics (Y-channel & RGB)
            crop_border = config.SCALE_FACTOR
            bicubic_psnr_y = calculate_psnr(bicubic_aligned, hr_aligned, crop_border=crop_border, test_y_channel=True)
            bicubic_ssim_y = calculate_ssim(bicubic_aligned, hr_aligned, crop_border=crop_border, test_y_channel=True)
            bicubic_psnr_rgb = calculate_psnr(bicubic_aligned, hr_aligned, crop_border=crop_border, test_y_channel=False)
            bicubic_ssim_rgb = calculate_ssim(bicubic_aligned, hr_aligned, crop_border=crop_border, test_y_channel=False)

            esrgan_psnr_y = calculate_psnr(esrgan_aligned, hr_aligned, crop_border=crop_border, test_y_channel=True)
            esrgan_ssim_y = calculate_ssim(esrgan_aligned, hr_aligned, crop_border=crop_border, test_y_channel=True)
            esrgan_psnr_rgb = calculate_psnr(esrgan_aligned, hr_aligned, crop_border=crop_border, test_y_channel=False)
            esrgan_ssim_rgb = calculate_ssim(esrgan_aligned, hr_aligned, crop_border=crop_border, test_y_channel=False)

            # 7. Optionally save visual outputs
            if save_images:
                from src.image_processing import save_image
                stem = img_file.stem
                save_image(lr_img, images_out_dir / f"{stem}_01_LR.png")
                save_image(bicubic_img, images_out_dir / f"{stem}_02_Bicubic.png")
                save_image(esrgan_img, images_out_dir / f"{stem}_03_ESRGAN.png")
                save_image(hr_img, images_out_dir / f"{stem}_04_HR_Reference.png")

            row = {
                "Image": img_file.name,
                "Hardware": str(device),
                "Model": "RRDBNet_ESRGAN_x4",
                "Scale_Factor": f"{config.SCALE_FACTOR}x",
                "LR_Resolution": f"{lr_w}x{lr_h}",
                "HR_Resolution": f"{hr_w}x{hr_h}",
                "Degradation": "Bicubic Decimation (Antialiased PIL)",
                "Bicubic_PSNR_Y_dB": round(bicubic_psnr_y, 2),
                "Bicubic_SSIM_Y": round(bicubic_ssim_y, 4),
                "Bicubic_PSNR_RGB_dB": round(bicubic_psnr_rgb, 2),
                "Bicubic_SSIM_RGB": round(bicubic_ssim_rgb, 4),
                "Bicubic_Time_sec": round(bicubic_time, 4),
                "ESRGAN_PSNR_Y_dB": round(esrgan_psnr_y, 2),
                "ESRGAN_SSIM_Y": round(esrgan_ssim_y, 4),
                "ESRGAN_PSNR_RGB_dB": round(esrgan_psnr_rgb, 2),
                "ESRGAN_SSIM_RGB": round(esrgan_ssim_rgb, 4),
                "ESRGAN_Time_sec": round(esrgan_time, 4),
                "Status": "Success"
            }
            if is_lpips_available():
                row["Bicubic_LPIPS"] = calculate_lpips(bicubic_aligned, hr_aligned, crop_border=crop_border, device=str(device))
                row["ESRGAN_LPIPS"] = calculate_lpips(esrgan_aligned, hr_aligned, crop_border=crop_border, device=str(device))
            records.append(row)
            print(f"Done! (ESRGAN PSNR: {esrgan_psnr_y:.2f} dB, Time: {esrgan_time:.2f}s)")

        except Exception as e:
            print(f"FAILED ({str(e)})")
            failures.append({"Image": img_file.name, "Error": str(e)})

    # Generate DataFrame
    df = pd.DataFrame(records)
    csv_path = out_path / "evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nPer-image results exported to: {csv_path}")

    # Compute aggregate statistics
    avg_stats = {
        "Total_Images_Evaluated": len(records),
        "Failed_Images": len(failures),
        "Mean_Bicubic_PSNR_Y_dB": round(df["Bicubic_PSNR_Y_dB"].mean(), 2) if not df.empty else 0.0,
        "Mean_Bicubic_SSIM_Y": round(df["Bicubic_SSIM_Y"].mean(), 4) if not df.empty else 0.0,
        "Mean_Bicubic_Time_sec": round(df["Bicubic_Time_sec"].mean(), 4) if not df.empty else 0.0,
        "Mean_ESRGAN_PSNR_Y_dB": round(df["ESRGAN_PSNR_Y_dB"].mean(), 2) if not df.empty else 0.0,
        "Mean_ESRGAN_SSIM_Y": round(df["ESRGAN_SSIM_Y"].mean(), 4) if not df.empty else 0.0,
        "Mean_ESRGAN_Time_sec": round(df["ESRGAN_Time_sec"].mean(), 4) if not df.empty else 0.0,
    }

    # Generate summary report in Markdown
    summary_path = out_path / "evaluation_summary.md"
    generate_markdown_report(avg_stats, df, failures, summary_path, device)

    # Generate visualization chart
    chart_path = out_path / "benchmark_comparison_chart.png"
    generate_comparison_chart(avg_stats, chart_path, df=df)

    print(f"Benchmark summary report saved: {summary_path}")
    print(f"Comparison chart saved: {chart_path}")
    print("===============================================================\n")

    return {
        "dataframe": df,
        "summary": avg_stats,
        "csv_path": csv_path,
        "summary_path": summary_path,
        "chart_path": chart_path,
        "failures": failures
    }


def generate_comparison_chart(stats: Dict[str, Any], chart_path: Path, df: Optional[pd.DataFrame] = None):
    """Generates a publication-grade multi-panel figure comparing Bicubic vs. ESRGAN."""
    methods = ["Bicubic (Interpolation)", "ESRGAN (Deep Learning)"]
    colors = ["#4A90E2", "#E94E77"]

    if df is not None and len(df) > 1:
        fig = plt.figure(figsize=(16, 9), dpi=150)
        fig.patch.set_facecolor("#f8f9fa")
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, :])

        # 1. PSNR Comparison
        psnr_vals = [stats["Mean_Bicubic_PSNR_Y_dB"], stats["Mean_ESRGAN_PSNR_Y_dB"]]
        bars1 = ax1.bar(methods, psnr_vals, color=colors, width=0.55, edgecolor="#333", linewidth=1.2)
        ax1.set_title("Mean PSNR (Luminance Y-Channel)", fontsize=11, fontweight="bold", pad=10)
        ax1.set_ylabel("PSNR (dB)", fontsize=10)
        ax1.set_ylim(0, max(psnr_vals) * 1.25 if psnr_vals else 40)
        ax1.grid(axis="y", linestyle="--", alpha=0.5)
        for bar in bars1:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.6, f"{yval:.2f} dB", ha="center", va="bottom", fontweight="bold")

        # 2. SSIM Comparison
        ssim_vals = [stats["Mean_Bicubic_SSIM_Y"], stats["Mean_ESRGAN_SSIM_Y"]]
        bars2 = ax2.bar(methods, ssim_vals, color=colors, width=0.55, edgecolor="#333", linewidth=1.2)
        ax2.set_title("Mean SSIM (Structural Similarity)", fontsize=11, fontweight="bold", pad=10)
        ax2.set_ylabel("SSIM (0.0 to 1.0)", fontsize=10)
        ax2.set_ylim(0, 1.1)
        ax2.grid(axis="y", linestyle="--", alpha=0.5)
        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")

        # 3. Latency Comparison
        time_vals = [stats["Mean_Bicubic_Time_sec"], stats["Mean_ESRGAN_Time_sec"]]
        bars3 = ax3.bar(methods, time_vals, color=colors, width=0.55, edgecolor="#333", linewidth=1.2)
        ax3.set_title("Inference Latency (Runtime / Image)", fontsize=11, fontweight="bold", pad=10)
        ax3.set_ylabel("Processing Time (s)", fontsize=10)
        ax3.set_ylim(0, max(time_vals) * 1.3 if time_vals else 2.0)
        ax3.grid(axis="y", linestyle="--", alpha=0.5)
        for bar in bars3:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, yval + (0.04 * max(time_vals)), f"{yval:.3f} s", ha="center", va="bottom", fontweight="bold")

        # 4. Per-Image PSNR Breakdown
        x_indices = np.arange(len(df))
        width = 0.38
        ax4.bar(x_indices - width/2, df["Bicubic_PSNR_Y_dB"], width, label="Bicubic Baseline", color="#4A90E2", edgecolor="#333", linewidth=0.8)
        ax4.bar(x_indices + width/2, df["ESRGAN_PSNR_Y_dB"], width, label="ESRGAN (4x)", color="#E94E77", edgecolor="#333", linewidth=0.8)
        short_names = [name.replace("eval_", "").replace(".png", "") for name in df["Image"]]
        ax4.set_xticks(x_indices)
        ax4.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
        ax4.set_title("Per-Image PSNR (Y-Channel dB): Perception-Distortion Comparison Across Dataset", fontsize=11, fontweight="bold", pad=10)
        ax4.set_ylabel("PSNR (dB)", fontsize=10)
        ax4.grid(axis="y", linestyle="--", alpha=0.5)
        ax4.legend(loc="upper right", frameon=True)

        plt.suptitle("SR Vision Lab: Bicubic Baseline vs. ESRGAN (4x Super-Resolution)", fontsize=13, fontweight="bold", y=0.99)
        plt.tight_layout()
        plt.savefig(str(chart_path), bbox_inches="tight")
        plt.close()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=150)
        fig.patch.set_facecolor("#f8f9fa")

        # 1. PSNR Comparison
        psnr_vals = [stats["Mean_Bicubic_PSNR_Y_dB"], stats["Mean_ESRGAN_PSNR_Y_dB"]]
        bars1 = axes[0].bar(methods, psnr_vals, color=colors, width=0.55, edgecolor="#333", linewidth=1.2)
        axes[0].set_title("Mean PSNR (Luminance Y-Channel)", fontsize=11, fontweight="bold", pad=12)
        axes[0].set_ylabel("Peak Signal-to-Noise Ratio (dB)", fontsize=10)
        axes[0].set_ylim(0, max(psnr_vals) * 1.25 if psnr_vals else 40)
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)
        for bar in bars1:
            yval = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2, yval + 0.8, f"{yval:.2f} dB", ha="center", va="bottom", fontweight="bold")

        # 2. SSIM Comparison
        ssim_vals = [stats["Mean_Bicubic_SSIM_Y"], stats["Mean_ESRGAN_SSIM_Y"]]
        bars2 = axes[1].bar(methods, ssim_vals, color=colors, width=0.55, edgecolor="#333", linewidth=1.2)
        axes[1].set_title("Mean SSIM (Structural Similarity)", fontsize=11, fontweight="bold", pad=12)
        axes[1].set_ylabel("SSIM Index (0.0 to 1.0)", fontsize=10)
        axes[1].set_ylim(0, 1.1)
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)
        for bar in bars2:
            yval = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")

        # 3. Latency Comparison
        time_vals = [stats["Mean_Bicubic_Time_sec"], stats["Mean_ESRGAN_Time_sec"]]
        bars3 = axes[2].bar(methods, time_vals, color=colors, width=0.55, edgecolor="#333", linewidth=1.2)
        axes[2].set_title("Inference Latency (Runtime per Image)", fontsize=11, fontweight="bold", pad=12)
        axes[2].set_ylabel("Processing Time (Seconds)", fontsize=10)
        axes[2].set_ylim(0, max(time_vals) * 1.3 if time_vals else 2.0)
        axes[2].grid(axis="y", linestyle="--", alpha=0.5)
        for bar in bars3:
            yval = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2, yval + (0.05 * max(time_vals)), f"{yval:.3f} s", ha="center", va="bottom", fontweight="bold")

        plt.suptitle("SR Vision Lab: Bicubic Baseline vs. ESRGAN (4x Super-Resolution)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(str(chart_path), bbox_inches="tight")
        plt.close()


def generate_markdown_report(
    stats: Dict[str, Any],
    df: pd.DataFrame,
    failures: List[Dict[str, str]],
    report_path: Path,
    device: Any
):
    """Writes a comprehensive evaluation report in Markdown format."""
    md = [
        "# 📊 Quantitative Evaluation Benchmark Report",
        "",
        "**Project:** SR Vision Lab — Comparative Image Super-Resolution Using ESRGAN and Deep Learning  ",
        f"**Date Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Hardware Platform:** {device}  ",
        f"**Model Checkpoint:** `{config.MODEL_WEIGHTS_FILENAME}` (RRDBNet, 23 Blocks, x4 Scale)  ",
        f"**Degradation Method:** 4× Bicubic Downsampling (CVIP Standard)  ",
        "",
        "---",
        "",
        "## 1. Executive Summary Statistics",
        "",
        "| Metric | Bicubic (Interpolation) | ESRGAN (Deep Learning) | Delta / Observation |",
        "| :--- | :---: | :---: | :--- |",
        f"| **PSNR (Y-Channel)** | `{stats['Mean_Bicubic_PSNR_Y_dB']} dB` | `{stats['Mean_ESRGAN_PSNR_Y_dB']} dB` | {stats['Mean_ESRGAN_PSNR_Y_dB'] - stats['Mean_Bicubic_PSNR_Y_dB']:+.2f} dB |",
        f"| **SSIM (Y-Channel)** | `{stats['Mean_Bicubic_SSIM_Y']}` | `{stats['Mean_ESRGAN_SSIM_Y']}` | {stats['Mean_ESRGAN_SSIM_Y'] - stats['Mean_Bicubic_SSIM_Y']:+.4f} |",
        f"| **Mean Latency** | `{stats['Mean_Bicubic_Time_sec']} s` | `{stats['Mean_ESRGAN_Time_sec']} s` | ESRGAN is compute-intensive |",
        "",
        "> **Key CVIP Insight — The Perception-Distortion Trade-off (Blau & Michaeli, 2018):**  ",
        "> In single-image super-resolution, optimizing strictly for pixel fidelity (PSNR/MSE) leads to overly smooth, blurry outputs because the model predicts the statistical average of all possible high-frequency textures.  ",
        "> ESRGAN employs an adversarial GAN loss and VGG perceptual loss to reconstruct plausible, sharp, photorealistic high-frequency microtextures. Because these generated micro-textures may differ slightly in exact pixel phase from the ground truth reference, ESRGAN may score a lower PSNR despite having dramatically superior human perceptual sharpness.",
        "",
        "---",
        "",
        "## 2. Per-Image Benchmark Results Table",
        "",
    ]

    if not df.empty:
        # Convert df to markdown table
        md.append(df.to_markdown(index=False))
    else:
        md.append("_No images were successfully evaluated._")

    if failures:
        md.extend([
            "",
            "## 3. Reported Failures",
            "",
            "| Image | Error Description |",
            "| :--- | :--- |"
        ])
        for f in failures:
            md.append(f"| `{f['Image']}` | {f['Error']} |")

    md.extend([
        "",
        "---",
        "## 4. Evaluation Methodology & Reproducibility",
        "- **Reference Standards:** Protocols adhere to DIV2K / NTIRE / PIRM Super-Resolution benchmarks.",
        "- **Color Space:** Evaluated on the luminance ($Y$) channel of standard ITU-R BT.601 YCbCr.",
        "- **Border Cropping:** Boundary shaved by $4$ pixels to eliminate border padding effects.",
        "- **Data Integrity:** No data fabricated or hardcoded; all scores generated via real-time inference.",
        ""
    ])

    report_path.write_text("\n".join(md), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Reproducible Quantitative Evaluation Benchmark for ESRGAN vs. Bicubic.")
    parser.add_argument("--hr-dir", type=str, default=str(config.EVAL_HR_DIR), help="Path to high-resolution reference images directory.")
    parser.add_argument("--output-dir", type=str, default=str(config.OUTPUTS_DIR), help="Path to directory for outputs.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Computation device.")
    parser.add_argument("--no-save-images", action="store_true", help="Do not save reconstructed visual outputs.")
    args = parser.parse_args()

    run_benchmark(
        hr_dir=Path(args.hr_dir),
        output_dir=Path(args.output_dir),
        device_str=args.device,
        save_images=not args.no_save_images
    )


if __name__ == "__main__":
    main()
