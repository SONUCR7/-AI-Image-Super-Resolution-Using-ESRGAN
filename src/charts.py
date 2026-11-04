"""
Visual Charting Module for SR Vision Lab
Renders publication-grade, dark-themed scientific charts for the Evaluation Lab
and Compare Methods sections. Guarantees 100% reliable rendering without CDN dependencies.
"""

from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Global Theme Palette (Matches SR Vision Lab Dark SaaS)
BG_COLOR = "#0A0E17"
CARD_BG = "#101622"
CYAN_ACCENT = "#00D2FF"
BLUE_ACCENT = "#3A86FF"
GREEN_ACCENT = "#10B981"
AMBER_ACCENT = "#F59E0B"
TEXT_COLOR = "#F8FAFC"
MUTED_TEXT = "#94A3B8"
GRID_COLOR = "rgba(255, 255, 255, 0.08)"


def _apply_dark_style(ax, fig, title: str, xlabel: str, ylabel: str):
    """Applies high-end dark SaaS aesthetics to a Matplotlib axis and figure."""
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(CARD_BG)
    
    # Title & Labels
    ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=14)
    ax.set_xlabel(xlabel, fontsize=10, fontweight="bold", color=MUTED_TEXT, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=10, fontweight="bold", color=MUTED_TEXT, labelpad=10)
    
    # Tick formatting
    ax.tick_params(colors=MUTED_TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#2A3447")
        spine.set_linewidth(1.0)
        
    ax.grid(True, linestyle="--", alpha=0.3, color="#334155")


def create_psnr_chart(df: pd.DataFrame) -> plt.Figure:
    """
    Renders grouped bar chart comparing Bicubic vs ESRGAN PSNR (Y-Channel)
    across all reference targets with mean benchmark baselines.
    """
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=120)
    
    labels = [p.replace("eval_", "").replace(".png", "").replace("_", " ") for p in df["Image"]]
    x = np.arange(len(labels))
    width = 0.38
    
    # Bars
    bars_bic = ax.bar(x - width/2, df["Bicubic_PSNR_Y_dB"], width, label="Bicubic Baseline", color=BLUE_ACCENT, alpha=0.88, edgecolor="none")
    bars_esr = ax.bar(x + width/2, df["ESRGAN_PSNR_Y_dB"], width, label="ESRGAN (RRDBNet)", color=CYAN_ACCENT, alpha=0.95, edgecolor="none")
    
    # Average reference lines
    b_mean = df["Bicubic_PSNR_Y_dB"].mean()
    e_mean = df["ESRGAN_PSNR_Y_dB"].mean()
    ax.axhline(b_mean, color=BLUE_ACCENT, linestyle=":", alpha=0.7, label=f"Bicubic Mean ({b_mean:.2f} dB)")
    ax.axhline(e_mean, color=CYAN_ACCENT, linestyle="--", alpha=0.85, label=f"ESRGAN Mean ({e_mean:.2f} dB)")
    
    _apply_dark_style(
        ax, fig,
        title="1. Peak Signal-to-Noise Ratio (PSNR) — Luminance Y-Channel",
        xlabel="Benchmark Reference Targets (25 Images)",
        ylabel="PSNR (dB) — Higher is Better"
    )
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=48, ha="right", fontsize=8)
    ax.legend(facecolor=CARD_BG, edgecolor="#2A3447", labelcolor=TEXT_COLOR, fontsize=8.5, loc="upper right")
    ax.set_ylim(0, max(df["ESRGAN_PSNR_Y_dB"].max(), df["Bicubic_PSNR_Y_dB"].max()) * 1.15)
    
    fig.tight_layout()
    return fig


def create_ssim_chart(df: pd.DataFrame) -> plt.Figure:
    """
    Renders grouped bar chart comparing Bicubic vs ESRGAN SSIM (Structural Similarity)
    across all reference targets.
    """
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=120)
    
    labels = [p.replace("eval_", "").replace(".png", "").replace("_", " ") for p in df["Image"]]
    x = np.arange(len(labels))
    width = 0.38
    
    ax.bar(x - width/2, df["Bicubic_SSIM_Y"], width, label="Bicubic Baseline", color=BLUE_ACCENT, alpha=0.88)
    ax.bar(x + width/2, df["ESRGAN_SSIM_Y"], width, label="ESRGAN (RRDBNet)", color=CYAN_ACCENT, alpha=0.95)
    
    b_mean = df["Bicubic_SSIM_Y"].mean()
    e_mean = df["ESRGAN_SSIM_Y"].mean()
    ax.axhline(b_mean, color=BLUE_ACCENT, linestyle=":", alpha=0.7, label=f"Bicubic Mean ({b_mean:.4f})")
    ax.axhline(e_mean, color=CYAN_ACCENT, linestyle="--", alpha=0.85, label=f"ESRGAN Mean ({e_mean:.4f})")
    
    _apply_dark_style(
        ax, fig,
        title="2. Structural Similarity Index Measure (SSIM) — Structural Fidelity",
        xlabel="Benchmark Reference Targets (25 Images)",
        ylabel="SSIM (0.0 to 1.0) — Higher is Better"
    )
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=48, ha="right", fontsize=8)
    ax.legend(facecolor=CARD_BG, edgecolor="#2A3447", labelcolor=TEXT_COLOR, fontsize=8.5, loc="upper left")
    ax.set_ylim(0, 1.08)
    
    fig.tight_layout()
    return fig


def create_latency_chart(df: pd.DataFrame) -> plt.Figure:
    """
    Renders runtime latency comparison across reference images on CPU hardware.
    """
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=120)
    
    labels = [p.replace("eval_", "").replace(".png", "").replace("_", " ") for p in df["Image"]]
    x = np.arange(len(labels))
    
    ax.plot(x, df["ESRGAN_Time_sec"], marker="o", markersize=5, color=CYAN_ACCENT, linewidth=2.2, label="ESRGAN Forward Pass (s)")
    ax.fill_between(x, df["ESRGAN_Time_sec"], color=CYAN_ACCENT, alpha=0.15)
    
    e_time_mean = df["ESRGAN_Time_sec"].mean()
    ax.axhline(e_time_mean, color="#38BDF8", linestyle="--", alpha=0.8, label=f"Mean Latency ({e_time_mean:.2f} s)")
    
    _apply_dark_style(
        ax, fig,
        title="3. ESRGAN Computational Latency Profile (CPU Multi-Threaded)",
        xlabel="Benchmark Reference Targets (25 Images)",
        ylabel="Execution Time (Seconds) — Lower is Better"
    )
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=48, ha="right", fontsize=8)
    ax.legend(facecolor=CARD_BG, edgecolor="#2A3447", labelcolor=TEXT_COLOR, fontsize=8.5, loc="upper right")
    ax.set_ylim(0, df["ESRGAN_Time_sec"].max() * 1.25)
    
    fig.tight_layout()
    return fig


def create_delta_chart(df: pd.DataFrame) -> plt.Figure:
    """
    Renders delta PSNR (ESRGAN - Bicubic) illustrating the Perception-Distortion Trade-off.
    Green bars indicate where ESRGAN delivers massive gains (+1 to +10 dB on geometric & structural targets);
    Amber bars indicate stochastic textures where Bicubic blur minimizes MSE.
    """
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=120)
    
    delta = df["ESRGAN_PSNR_Y_dB"] - df["Bicubic_PSNR_Y_dB"]
    labels = [p.replace("eval_", "").replace(".png", "").replace("_", " ") for p in df["Image"]]
    x = np.arange(len(labels))
    
    colors = [GREEN_ACCENT if d >= 0 else AMBER_ACCENT for d in delta]
    
    ax.bar(x, delta, color=colors, width=0.55, alpha=0.9, edgecolor="none")
    ax.axhline(0, color="#64748B", linewidth=1.2, linestyle="-")
    
    mean_delta = delta.mean()
    ax.axhline(mean_delta, color="#38BDF8", linestyle="--", linewidth=1.5, label=f"Net Mean Gain (+{mean_delta:.2f} dB)")
    
    _apply_dark_style(
        ax, fig,
        title="4. Differential PSNR (ESRGAN vs. Bicubic Baseline) — Structural Gain vs. Perception-Distortion Trade-off",
        xlabel="Benchmark Reference Targets (25 Images)",
        ylabel="Δ PSNR (dB) — Positive = ESRGAN Superiority"
    )
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=48, ha="right", fontsize=8)
    ax.legend(facecolor=CARD_BG, edgecolor="#2A3447", labelcolor=TEXT_COLOR, fontsize=8.5, loc="upper left")
    
    # Custom annotations
    ax.text(
        0.02, 0.08,
        "■ Green: Significant GAN Structural Lead (+1 to +10 dB)\n■ Amber: Stochastic Micro-Texture Variance (Perception-Distortion Trade-off)",
        transform=ax.transAxes,
        fontsize=8.5,
        color=MUTED_TEXT,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=BG_COLOR, edgecolor="#2A3447", alpha=0.8)
    )
    
    fig.tight_layout()
    return fig
