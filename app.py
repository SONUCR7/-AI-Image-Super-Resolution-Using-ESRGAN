"""
SR Vision Lab: Professional AI Image Super-Resolution Platform
Comparative Image Super-Resolution Using ESRGAN and Deep Learning
Enterprise-Grade Computer Vision & Image Processing (CVIP) Research Suite

Tagline: Restore More. See Better.
Architecture: RRDBNet (23 RRDB Blocks, 64 Features, 32 Growth Channels, 16.7M Parameters)
Scale Factor: 4x Super-Resolution
"""

import io
import time
import math
import json
import base64
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

import config
from src.image_processing import (
    ensure_rgb,
    bicubic_upscale,
    align_images,
    load_image,
    save_image
)
from src.inference import super_resolve, estimate_tiles
from src.metrics import calculate_psnr, calculate_ssim
from src.model_loader import get_cached_model, load_esrgan_model
from scripts.evaluate import run_benchmark, generate_comparison_chart


# -----------------------------------------------------------------------------
# 1. Page Configuration & Professional AI SaaS Dark Design System
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SR Vision Lab | AI Image Super-Resolution",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SaaS Dark CSS System
st.markdown("""
<style>
    /* Global Background & Typography */
    .stApp {
        background-color: #0B0F17;
        color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Top Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #111827;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #1E293B;
        margin-bottom: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 500;
        font-size: 0.92rem;
        padding: 0 20px;
        background-color: transparent;
        border: none;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #F8FAFC;
        background-color: rgba(255, 255, 255, 0.04);
    }

    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
        font-weight: 600;
        border-bottom: 2px solid #06B6D4 !important;
    }

    /* Metric & Telemetry Cards */
    .saas-card {
        background-color: #111827;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .saas-card:hover {
        border-color: #334155;
    }
    .saas-card-label {
        font-size: 0.76rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .saas-card-val {
        font-size: 1.45rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 5px;
    }
    .saas-card-sub {
        font-size: 0.75rem;
        color: #64748B;
        margin-top: 3px;
    }

    /* Highlight Banner Cards */
    .hero-card {
        background: linear-gradient(135deg, #0F172A 0%, #111827 100%);
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    .hero-title {
        font-size: 1.75rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        line-height: 1.5;
    }

    /* Resolution Pipeline Cards */
    .res-flow-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 12px 14px;
        text-align: center;
    }
    .res-flow-title {
        font-size: 0.72rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
    }
    .res-flow-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: #38BDF8;
        margin-top: 2px;
    }
    .res-flow-tag {
        font-size: 0.70rem;
        color: #64748B;
    }

    /* Clean Subtle Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 500;
        border-radius: 9999px;
        background: rgba(6, 182, 212, 0.1);
        color: #38BDF8;
        border: 1px solid rgba(6, 182, 212, 0.3);
    }
    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #06B6D4;
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284C7 0%, #06B6D4 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        box-shadow: 0 2px 12px rgba(6, 182, 212, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 18px rgba(6, 182, 212, 0.45) !important;
        transform: translateY(-1px);
    }
    .stButton > button[kind="secondary"] {
        background: #1E293B !important;
        color: #E2E8F0 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }

    /* Sidebar Refinements */
    [data-testid="stSidebar"] {
        background-color: #0D121D;
        border-right: 1px solid #1E293B;
    }
    [data-testid="stSidebar"] hr {
        border-color: #1E293B;
    }

    /* Image Containers */
    img {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Helper Utilities & Interactive Components
# -----------------------------------------------------------------------------
def get_image_download_bytes(img_rgb: np.ndarray, format_type: str = "PNG") -> bytes:
    """Converts an RGB numpy array to in-memory bytes for downloading."""
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format=format_type)
    return buf.getvalue()


def numpy_to_base64_jpeg(img_rgb: np.ndarray, quality: int = 90, max_dim: int = 1400) -> str:
    """Converts numpy array into a compact base64 JPEG string for high-speed slider rendering."""
    pil_img = Image.fromarray(img_rgb)
    w, h = pil_img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64_str}"


def render_before_after_slider(
    img_before: np.ndarray,
    img_after: np.ndarray,
    label_before: str = "Original Input",
    label_after: str = "ESRGAN (4×)",
    height: int = 560
) -> None:
    """
    Renders a responsive, client-side HTML5/JavaScript draggable comparison slider
    with zoom, pan, reset, and native fullscreen controls.
    """
    # Resize before image to match after image dimensions for 1-to-1 visual alignment
    target_h, target_w = img_after.shape[:2]
    if img_before.shape[:2] != (target_h, target_w):
        before_aligned = cv2.resize(img_before, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    else:
        before_aligned = img_before

    b64_before = numpy_to_base64_jpeg(before_aligned)
    b64_after = numpy_to_base64_jpeg(img_after)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{ background: transparent; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; overflow: hidden; }}
      .slider-box {{
        position: relative;
        width: 100%;
        height: {height}px;
        background: #0B0F17;
        border: 1px solid #1E293B;
        border-radius: 12px;
        overflow: hidden;
        user-select: none;
      }}
      .img-viewport {{
        position: relative;
        width: 100%;
        height: 100%;
        transform-origin: center center;
        transition: transform 0.12s ease-out;
      }}
      .layer {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: contain;
        pointer-events: none;
      }}
      .before-layer {{
        clip-path: polygon(0 0, var(--pos, 50%) 0, var(--pos, 50%) 100%, 0 100%);
      }}
      .divider {{
        position: absolute;
        top: 0;
        bottom: 0;
        left: var(--pos, 50%);
        width: 2px;
        background: #06B6D4;
        box-shadow: 0 0 10px #06B6D4;
        pointer-events: none;
        z-index: 10;
      }}
      .divider-knob {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 36px;
        height: 36px;
        background: #0F172A;
        border: 2px solid #06B6D4;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #F8FAFC;
        font-size: 12px;
        box-shadow: 0 2px 10px rgba(6, 182, 212, 0.4);
      }}
      .range-bar {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        opacity: 0;
        cursor: ew-resize;
        z-index: 20;
        margin: 0;
      }}
      .badge {{
        position: absolute;
        top: 14px;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        z-index: 15;
        pointer-events: none;
        backdrop-filter: blur(8px);
      }}
      .badge-left {{
        left: 14px;
        background: rgba(15, 23, 42, 0.85);
        color: #94A3B8;
        border: 1px solid rgba(148, 163, 184, 0.3);
      }}
      .badge-right {{
        right: 14px;
        background: rgba(6, 182, 212, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.4);
      }}
      .ctrl-toolbar {{
        position: absolute;
        bottom: 12px;
        right: 12px;
        display: flex;
        gap: 6px;
        z-index: 25;
      }}
      .ctrl-btn {{
        background: rgba(15, 23, 42, 0.85);
        color: #F8FAFC;
        border: 1px solid #334155;
        border-radius: 6px;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
        cursor: pointer;
        backdrop-filter: blur(4px);
        transition: all 0.15s ease;
      }}
      .ctrl-btn:hover {{
        background: #1E293B;
        border-color: #06B6D4;
        color: #06B6D4;
      }}
    </style>
    </head>
    <body>
    <div class="slider-box" id="sliderBox">
      <div class="badge badge-left">{label_before}</div>
      <div class="badge badge-right">{label_after}</div>
      
      <div class="img-viewport" id="imgViewport" style="--pos: 50%;">
        <img class="layer after-layer" src="{b64_after}" alt="After" />
        <img class="layer before-layer" src="{b64_before}" alt="Before" />
        <div class="divider">
          <div class="divider-knob">&#9664;&#9654;</div>
        </div>
      </div>

      <input type="range" min="0" max="100" value="50" class="range-bar" id="rangeBar" />

      <div class="ctrl-toolbar">
        <button class="ctrl-btn" id="btnZoomIn" title="Zoom In">+</button>
        <button class="ctrl-btn" id="btnZoomOut" title="Zoom Out">-</button>
        <button class="ctrl-btn" id="btnReset" title="Reset">&#8635;</button>
        <button class="ctrl-btn" id="btnFullscreen" title="Fullscreen">&#x26F6;</button>
      </div>
    </div>

    <script>
      const slider = document.getElementById('rangeBar');
      const viewport = document.getElementById('imgViewport');
      const box = document.getElementById('sliderBox');
      let zoom = 1.0;

      slider.addEventListener('input', (e) => {{
        viewport.style.setProperty('--pos', e.target.value + '%');
      }});

      document.getElementById('btnZoomIn').addEventListener('click', (e) => {{
        e.stopPropagation();
        zoom = Math.min(zoom + 0.35, 3.5);
        viewport.style.transform = 'scale(' + zoom + ')';
      }});

      document.getElementById('btnZoomOut').addEventListener('click', (e) => {{
        e.stopPropagation();
        zoom = Math.max(zoom - 0.35, 1.0);
        viewport.style.transform = 'scale(' + zoom + ')';
      }});

      document.getElementById('btnReset').addEventListener('click', (e) => {{
        e.stopPropagation();
        zoom = 1.0;
        viewport.style.transform = 'scale(1)';
        slider.value = 50;
        viewport.style.setProperty('--pos', '50%');
      }});

      document.getElementById('btnFullscreen').addEventListener('click', (e) => {{
        e.stopPropagation();
        if (!document.fullscreenElement) {{
          box.requestFullscreen().catch(err => alert(err.message));
        }} else {{
          document.exitFullscreen();
        }}
      }});
    </script>
    </body>
    </html>
    """
    components.html(html_code, height=height + 25)


def create_results_zip(
    esrgan_img: np.ndarray,
    bicubic_img: np.ndarray,
    original_img: np.ndarray,
    metadata: Dict[str, Any]
) -> bytes:
    """Packages enhanced outputs, baseline, input, and telemetry metadata into a downloadable ZIP archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("01_ESRGAN_4x_enhanced.png", get_image_download_bytes(esrgan_img, "PNG"))
        zf.writestr("02_Bicubic_4x_baseline.png", get_image_download_bytes(bicubic_img, "PNG"))
        zf.writestr("03_Input_reference.png", get_image_download_bytes(original_img, "PNG"))
        report_str = json.dumps(metadata, indent=4)
        zf.writestr("super_resolution_metadata.json", report_str)
    return buf.getvalue()


def compute_difference_map(img1: np.ndarray, img2: np.ndarray, amplification: float = 4.0) -> np.ndarray:
    """Computes amplified high-frequency residual difference |ESRGAN - Bicubic| for CVIP analysis."""
    a1, a2 = align_images(ensure_rgb(img1), ensure_rgb(img2))
    diff = np.abs(a1.astype(np.float32) - a2.astype(np.float32))
    diff_gray = np.mean(diff, axis=2)
    diff_amplified = np.clip(diff_gray * amplification, 0, 255).astype(np.uint8)
    diff_heatmap = cv2.applyColorMap(diff_amplified, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(diff_heatmap, cv2.COLOR_BGR2RGB)


# -----------------------------------------------------------------------------
# 3. Streamlined Sidebar (Clean, Compact, Non-Intrusive)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔬 **SR Vision Lab**")
    st.caption("Restore More. See Better.")

    # Status Indicator
    avail_cuda = config.torch.cuda.is_available()
    device_label = "CUDA (GPU Acceleration)" if avail_cuda else "CPU (Multi-Threaded)"
    st.markdown(f"""
    <div style="margin: 8px 0 16px 0;">
        <span class="status-badge">
            <span class="status-dot"></span> Model Ready: RRDBNet (4×)
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"Compute Hardware: **{device_label}**")

    st.markdown("---")
    st.markdown("#### 📁 Quick Test Library")
    sample_options = ["None (Upload Custom Image)"]
    if config.SAMPLES_DIR.exists():
        sample_files = sorted([f.name for f in config.SAMPLES_DIR.glob("*.png")])
        sample_options.extend(sample_files)

    selected_sample = st.selectbox(
        "Select preloaded benchmark sample:",
        sample_options,
        index=0,
        help="Quickly evaluate the super-resolution pipeline with pre-generated reference images."
    )

    # Collapsible Model Information (Keeps sidebar clean)
    st.markdown("---")
    with st.expander("ℹ️ Model Architecture & Specs", expanded=False):
        st.markdown(f"""
        - **Generator Network:** `RRDBNet` (ESRGAN)
        - **Number of Blocks:** `23 RRDB` (69 Dense Layers)
        - **Growth Channels:** `32` | **Feature Maps:** `64`
        - **Verified Parameters:** `16,697,987` (~16.7M)
        - **Scale Factor:** `4×` (16× Pixel Density)
        - **Weights Checkpoint:** `{config.MODEL_WEIGHTS_FILENAME}`
        - **Checkpoint Size:** `63.8 MB`
        - **Official Architecture:** *Wang et al. (ECCV 2018)*
        - [View ESRGAN Paper](https://arxiv.org/abs/1809.00219)
        """)

    st.markdown("---")
    st.caption("CVIP Image Processing & Deep Learning Platform")


# -----------------------------------------------------------------------------
# 4. Main Application Navigation (4 Clean Sections)
# -----------------------------------------------------------------------------
tab_enhance, tab_compare, tab_eval, tab_about = st.tabs([
    "Enhance Image",
    "Compare Methods",
    "Evaluation Lab",
    "About"
])


# =============================================================================
# SECTION 1: ENHANCE IMAGE (MAIN WORKSPACE)
# =============================================================================
with tab_enhance:
    # Hero Section
    st.markdown("""
    <div class="hero-card">
        <div class="hero-title">AI-Powered Image Super-Resolution</div>
        <div class="hero-subtitle">
            Reconstruct high-frequency micro-textures and sharp contours from low-resolution imagery 
            using the authentic 23-block <strong>ESRGAN (Enhanced Super-Resolution GAN)</strong> deep convolutional architecture.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_settings = st.columns([1.2, 1.0])

    input_image_array: Optional[np.ndarray] = None
    input_metadata: Dict[str, Any] = {}

    with col_upload:
        st.markdown("#### 📤 Image Upload")
        uploaded_file = st.file_uploader(
            "Upload an image (JPG, JPEG, PNG, WEBP):",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload an image to super-resolve at 4× resolution."
        )

    # Decode uploaded file or selected sample
    if uploaded_file is not None:
        try:
            raw_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            input_image_array = np.array(pil_img, dtype=np.uint8)
            orig_w, orig_h = pil_img.size

            input_metadata = {
                "filename": getattr(uploaded_file, "name", "uploaded_image.png"),
                "orig_w": orig_w,
                "orig_h": orig_h,
                "format": pil_img.format or "PNG",
                "aspect_ratio": round(orig_w / max(1, orig_h), 2),
                "size_mb": round(len(raw_bytes) / (1024 * 1024), 2),
            }
        except Exception as e:
            st.error(f"Failed to decode uploaded image: {e}")
            input_image_array = None

    elif selected_sample != "None (Upload Custom Image)":
        sample_path = config.SAMPLES_DIR / selected_sample
        if sample_path.exists():
            pil_img = Image.open(sample_path).convert("RGB")
            input_image_array = np.array(pil_img, dtype=np.uint8)
            orig_w, orig_h = pil_img.size
            input_metadata = {
                "filename": selected_sample,
                "orig_w": orig_w,
                "orig_h": orig_h,
                "format": "PNG",
                "aspect_ratio": round(orig_w / max(1, orig_h), 2),
                "size_mb": round(sample_path.stat().st_size / (1024 * 1024), 2),
            }

    # Enhancement Settings & Resolution Pipeline Preview
    processing_mode = "Fast Mode"
    proc_w, proc_h = 0, 0
    out_w, out_h = 0, 0

    with col_settings:
        st.markdown("#### ⚙️ Enhancement Settings")
        if input_metadata and input_image_array is not None:
            w_in, h_in = input_metadata["orig_w"], input_metadata["orig_h"]
            is_large = max(w_in, h_in) > 400

            # Mode selection
            mode_choice = st.radio(
                "Processing Engine Mode:",
                [
                    "⚡ Fast Mode (~5–8s on CPU - Recommended)",
                    "🔬 Full Resolution Mode (Tiled Inference)"
                ],
                index=0,
                help=(
                    "Fast Mode scales large images to a ~384px proxy for rapid CPU turnaround.\n"
                    "Full Resolution Mode processes the full image via memory-safe overlapping tiles."
                )
            )

            if "Fast" in mode_choice and is_large and not avail_cuda:
                processing_mode = "Fast Mode"
                scale_down = 384.0 / max(w_in, h_in)
                proc_w = max(16, int(w_in * scale_down))
                proc_h = max(16, int(h_in * scale_down))
                out_w = proc_w * config.SCALE_FACTOR
                out_h = proc_h * config.SCALE_FACTOR
            else:
                processing_mode = "Full Resolution Mode"
                proc_w = w_in
                proc_h = h_in
                out_w = proc_w * config.SCALE_FACTOR
                out_h = proc_h * config.SCALE_FACTOR

            # Thumbnail preview and info
            st.image(input_image_array, caption=f"Thumbnail: {input_metadata['filename']} ({input_metadata['size_mb']} MB)", width=140)

        else:
            st.info("Upload an image or select a sample from the sidebar to configure enhancement settings.")

    # 3-STAGE RESOLUTION PIPELINE REPORTING (Fixes Resolution Confusion)
    if input_metadata and input_image_array is not None:
        st.markdown("---")
        st.markdown("##### 📏 Resolution Pipeline Specifications")
        r_c1, r_c2, r_c3 = st.columns(3)

        with r_c1:
            st.markdown(f"""
            <div class="res-flow-card">
                <div class="res-flow-title">A. Original Uploaded</div>
                <div class="res-flow-val">{input_metadata['orig_w']} × {input_metadata['orig_h']} px</div>
                <div class="res-flow-tag">Native Unscaled Source</div>
            </div>
            """, unsafe_allow_html=True)

        with r_c2:
            tag_text = "Downscaled Proxy" if processing_mode == "Fast Mode" and max(input_metadata['orig_w'], input_metadata['orig_h']) > 400 else "Full Native Dimensions"
            st.markdown(f"""
            <div class="res-flow-card">
                <div class="res-flow-title">B. Model Processing Resolution</div>
                <div class="res-flow-val">{proc_w} × {proc_h} px</div>
                <div class="res-flow-tag">{processing_mode} ({tag_text})</div>
            </div>
            """, unsafe_allow_html=True)

        with r_c3:
            st.markdown(f"""
            <div class="res-flow-card">
                <div class="res-flow-title">C. Generated Output</div>
                <div class="res-flow-val">{out_w} × {out_h} px</div>
                <div class="res-flow-tag">4× Super-Resolved (16× Pixels)</div>
            </div>
            """, unsafe_allow_html=True)

        if processing_mode == "Fast Mode" and max(input_metadata['orig_w'], input_metadata['orig_h']) > 400:
            st.caption(
                f"ℹ️ **Resolution Reporting Note:** Fast Mode optimizes CPU turnaround (~5–8s) by super-resolving a "
                f"**{proc_w}×{proc_h} px** proxy copy, yielding a **{out_w}×{out_h} px** output. "
                f"To process the full {input_metadata['orig_w']}×{input_metadata['orig_h']} px image directly, select Full Resolution Mode above."
            )
        elif processing_mode == "Full Resolution Mode" and not avail_cuda:
            est_t = estimate_tiles(proc_h, proc_w, tile_size=256, overlap=20)
            if est_t > 1:
                st.warning(
                    f"⚠️ **Full Resolution Mode:** Image will be divided into **{est_t} feathered tiles** (256×256 px). "
                    f"Processing on CPU may take ~{est_t * 20}s. Feathered linear blending eliminates seam artifacts."
                )

        # Enhance Action Button
        btn_col, _ = st.columns([1.5, 2.5])
        with btn_col:
            run_btn = st.button("✨ Enhance Image (4× Super-Resolution)", type="primary", use_container_width=True)

        # Inference Execution
        if run_btn:
            prog_bar = st.progress(0.0)
            status_box = st.empty()

            def on_inference_progress(current, total, msg):
                prog_bar.progress(min(1.0, current / total))
                status_box.markdown(f"⏳ **{msg}**")

            try:
                # Prepare input according to chosen mode
                if processing_mode == "Fast Mode" and max(input_image_array.shape[:2]) > 400 and not avail_cuda:
                    exec_input = cv2.resize(input_image_array, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
                else:
                    exec_input = input_image_array.copy()

                status_box.markdown("⏳ Stage 1/2: Computing traditional Bicubic baseline...")
                bicubic_res, bicubic_t = bicubic_upscale(exec_input, scale=4)

                status_box.markdown("⏳ Stage 2/2: Executing deep ESRGAN generator pass...")
                esrgan_res, esrgan_t = super_resolve(
                    exec_input,
                    use_tiling=(processing_mode == "Full Resolution Mode" and not avail_cuda and max(proc_w, proc_h) > 384),
                    progress_callback=on_inference_progress
                )

                prog_bar.progress(1.0)
                status_box.empty()
                prog_bar.empty()

                # Store cleanly in Streamlit session state
                st.session_state["results"] = {
                    "orig_uploaded": input_image_array,
                    "exec_input": exec_input,
                    "bicubic_out": bicubic_res,
                    "esrgan_out": esrgan_res,
                    "bicubic_time": bicubic_t,
                    "esrgan_time": esrgan_t,
                    "mode": processing_mode,
                    "orig_w": input_metadata["orig_w"],
                    "orig_h": input_metadata["orig_h"],
                    "proc_w": exec_input.shape[1],
                    "proc_h": exec_input.shape[0],
                    "out_w": esrgan_res.shape[1],
                    "out_h": esrgan_res.shape[0],
                    "filename": input_metadata["filename"],
                    "device": str(config.get_device())
                }
                st.success(f"✅ Enhancement complete! 4× Super-resolution finished in {esrgan_t:.2f} seconds.")

            except Exception as e:
                status_box.empty()
                prog_bar.empty()
                st.error(f"❌ Error during enhancement: {str(e)}")

    # RESULTS WORKSPACE
    if "results" in st.session_state and st.session_state["results"] is not None:
        res = st.session_state["results"]

        st.markdown("---")
        st.markdown("### 📊 Enhancement Telemetry")

        # Compact Information Cards
        t1, t2, t3, t4, t5, t6 = st.columns(6)
        with t1:
            st.markdown(f"""
            <div class="saas-card">
                <div class="saas-card-label">Original Input</div>
                <div class="saas-card-val">{res['orig_w']}×{res['orig_h']}</div>
                <div class="saas-card-sub">Source Image px</div>
            </div>
            """, unsafe_allow_html=True)
        with t2:
            st.markdown(f"""
            <div class="saas-card">
                <div class="saas-card-label">Model Processing</div>
                <div class="saas-card-val">{res['proc_w']}×{res['proc_h']}</div>
                <div class="saas-card-sub">{res['mode']}</div>
            </div>
            """, unsafe_allow_html=True)
        with t3:
            st.markdown(f"""
            <div class="saas-card">
                <div class="saas-card-label">Generated Output</div>
                <div class="saas-card-val">{res['out_w']}×{res['out_h']}</div>
                <div class="saas-card-sub">4× Super-Resolved</div>
            </div>
            """, unsafe_allow_html=True)
        with t4:
            st.markdown("""
            <div class="saas-card">
                <div class="saas-card-label">Spatial Scale</div>
                <div class="saas-card-val">4.0 ×</div>
                <div class="saas-card-sub">16× Total Pixels</div>
            </div>
            """, unsafe_allow_html=True)
        with t5:
            st.markdown(f"""
            <div class="saas-card">
                <div class="saas-card-label">ESRGAN Latency</div>
                <div class="saas-card-val">{res['esrgan_time']:.2f}s</div>
                <div class="saas-card-sub">Bicubic: {res['bicubic_time']*1000:.1f}ms</div>
            </div>
            """, unsafe_allow_html=True)
        with t6:
            st.markdown(f"""
            <div class="saas-card">
                <div class="saas-card-label">Compute Device</div>
                <div class="saas-card-val">{res['device'].upper()}</div>
                <div class="saas-card-sub">{'GPU Active' if 'cuda' in res['device'] else 'CPU Multi-Thread'}</div>
            </div>
            """, unsafe_allow_html=True)

        # INTERACTIVE BEFORE/AFTER DRAGGABLE COMPARISON VIEWER
        st.markdown("---")
        st.markdown("#### ↔️ Interactive Before / After Comparison Slider")
        st.caption("Drag the glowing cyan divider horizontally to contrast the original input against the ESRGAN 4× super-resolved reconstruction. Use toolbar controls for zoom, pan, and fullscreen inspection.")

        render_before_after_slider(
            img_before=res["exec_input"],
            img_after=res["esrgan_out"],
            label_before="Original Processing Input",
            label_after="ESRGAN (4× Output)",
            height=540
        )

        # 3-WAY METHOD COMPARISON GALLERY
        st.markdown("---")
        st.markdown("#### 🖼️ Three-Way Method Output Gallery")
        g_c1, g_c2, g_c3 = st.columns(3)

        with g_c1:
            st.markdown("##### 🔍 1. Original Input")
            st.caption(f"Processing Resolution: **{res['proc_w']} × {res['proc_h']}** px")
            st.image(res["exec_input"], use_container_width=True)

        with g_c2:
            st.markdown("##### 📐 2. Bicubic Interpolation (4×)")
            st.caption(f"Output: **{res['bicubic_out'].shape[1]} × {res['bicubic_out'].shape[0]}** px | Latency: **{res['bicubic_time']*1000:.1f} ms**")
            st.image(res["bicubic_out"], use_container_width=True)
            st.download_button(
                "⬇️ Download Bicubic PNG",
                data=get_image_download_bytes(res["bicubic_out"]),
                file_name="bicubic_4x_baseline.png",
                mime="image/png",
                use_container_width=True
            )

        with g_c3:
            st.markdown("##### 🧠 3. ESRGAN Output (4×)")
            st.caption(f"Output: **{res['esrgan_out'].shape[1]} × {res['esrgan_out'].shape[0]}** px | Latency: **{res['esrgan_time']:.2f} s**")
            st.image(res["esrgan_out"], use_container_width=True)
            st.download_button(
                "⬇️ Download ESRGAN 4× PNG",
                data=get_image_download_bytes(res["esrgan_out"]),
                file_name="esrgan_4x_enhanced.png",
                mime="image/png",
                type="primary",
                use_container_width=True
            )

        # DOWNLOAD ALL RESULTS (ZIP)
        st.markdown("---")
        zip_meta = {
            "application": "SR Vision Lab",
            "model": "ESRGAN (RRDBNet)",
            "scale_factor": "4x",
            "processing_mode": res["mode"],
            "original_uploaded_resolution": f"{res['orig_w']}x{res['orig_h']}",
            "processing_input_resolution": f"{res['proc_w']}x{res['proc_h']}",
            "generated_output_resolution": f"{res['out_w']}x{res['out_h']}",
            "esrgan_latency_seconds": round(res["esrgan_time"], 4),
            "bicubic_latency_seconds": round(res["bicubic_time"], 4),
            "device": res["device"]
        }
        zip_bytes = create_results_zip(res["esrgan_out"], res["bicubic_out"], res["exec_input"], zip_meta)

        z_col, _ = st.columns([1.5, 2.5])
        with z_col:
            st.download_button(
                "📦 Download All Results (ZIP Archive)",
                data=zip_bytes,
                file_name=f"sr_vision_lab_results_{Path(res['filename']).stem}.zip",
                mime="application/zip",
                use_container_width=True
            )


# =============================================================================
# SECTION 2: COMPARE METHODS
# =============================================================================
with tab_compare:
    st.markdown("### ⚖️ Classical Interpolation vs. Deep Learning Super-Resolution")
    st.markdown(
        "A rigorous side-by-side scientific comparison between classical spatial kernel interpolation "
        "and learned deep generative adversarial reconstruction."
    )

    # Conceptual Comparison Table
    st.markdown("#### 📋 Algorithmic Paradigms")
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("""
        <div class="saas-card">
            <h4 style="color: #38BDF8; margin-top: 0;">📐 Bicubic Interpolation (Classical CV)</h4>
            <p style="font-size: 0.88rem; color: #CBD5E1;">
                <strong>Mathematical Principle:</strong> Estimates unknown pixels via 2D convolution with a 16-point cubic spline kernel (Catmull-Rom spline, <em>a = -0.5</em>).
            </p>
            <ul style="font-size: 0.84rem; color: #94A3B8; margin-left: 20px;">
                <li>Operates purely as a spatial low-pass filter.</li>
                <li>Estimates pixels strictly from local 4×4 pixel neighborhood.</li>
                <li>Computationally lightweight (&lt;1 ms runtime).</li>
                <li>Cannot synthesize lost high-frequency textures or micro-edges.</li>
                <li>Exhibits edge blurring, ringing (Gibbs phenomenon), and staircase jaggies.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_t2:
        st.markdown("""
        <div class="saas-card">
            <h4 style="color: #06B6D4; margin-top: 0;">🧠 ESRGAN (Deep Perceptual GAN)</h4>
            <p style="font-size: 0.88rem; color: #CBD5E1;">
                <strong>Mathematical Principle:</strong> 23-block Residual-in-Residual Dense Network (RRDBNet) trained with Relativistic Average GAN (RaGAN) and VGG-19 perceptual feature loss.
            </p>
            <ul style="font-size: 0.84rem; color: #94A3B8; margin-left: 20px;">
                <li>Leverages deep learned prior distributions of natural imagery.</li>
                <li>Synthesizes plausible high-frequency micro-ridges and organic textures.</li>
                <li>Multi-scale residual feedback without Batch Normalization artifacts.</li>
                <li>Compute-intensive (~3–7s on CPU, ~0.15s on CUDA).</li>
                <li>Hallucinates realistic textures that may differ in exact pixel phase from ground truth.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Active Visual Comparison
    st.markdown("---")
    st.markdown("#### 🔬 Live Method Visualizer & High-Frequency Difference Heatmap")

    # Use active session result or fallback sample
    comp_input = None
    comp_bicubic = None
    comp_esrgan = None

    if "results" in st.session_state and st.session_state["results"] is not None:
        comp_input = st.session_state["results"]["exec_input"]
        comp_bicubic = st.session_state["results"]["bicubic_out"]
        comp_esrgan = st.session_state["results"]["esrgan_out"]
    elif config.SAMPLES_DIR.exists() and (config.SAMPLES_DIR / "sample_urban_architecture.png").exists():
        sample_img = load_image(config.SAMPLES_DIR / "sample_urban_architecture.png")
        comp_input = sample_img
        comp_bicubic, _ = bicubic_upscale(sample_img, scale=4)
        comp_esrgan, _ = super_resolve(sample_img)

    if comp_input is not None and comp_bicubic is not None and comp_esrgan is not None:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("##### 1. Input Image")
            st.caption(f"{comp_input.shape[1]} × {comp_input.shape[0]} px")
            st.image(comp_input, use_container_width=True)

        with c2:
            st.markdown("##### 2. Bicubic (4×)")
            st.caption(f"{comp_bicubic.shape[1]} × {comp_bicubic.shape[0]} px")
            st.image(comp_bicubic, use_container_width=True)

        with c3:
            st.markdown("##### 3. ESRGAN (4×)")
            st.caption(f"{comp_esrgan.shape[1]} × {comp_esrgan.shape[0]} px")
            st.image(comp_esrgan, use_container_width=True)

        with c4:
            st.markdown("##### 4. Residual Divergence")
            st.caption("|ESRGAN - Bicubic| Amplified")
            diff_map = compute_difference_map(comp_esrgan, comp_bicubic, amplification=5.0)
            st.image(diff_map, use_container_width=True)
            st.caption(
                "Bright regions illustrate spatial divergence between models (synthesized micro-textures, "
                "edge sharpening gradients, and GAN priors). Note: these differences represent generative divergence, "
                "not necessarily ground-truth recovered data."
            )

    # Scientific Caution & Hallucination Note
    st.markdown("---")
    st.info(
        "💡 **Critical CVIP Principle — Hallucination vs. Physical Ground Truth:**\n\n"
        "Single-image super-resolution is fundamentally an **ill-posed inverse problem**. An infinite number of plausible high-resolution "
        "scenes can produce the exact same low-resolution pixel observation after downsampling. While ESRGAN synthesizes crisp, "
        "photorealistic details (foliage micro-textures, masonry crevices, sharp contours), these structures represent **learned priors** "
        "rather than physically measured scene data. They should not be treated as forensic ground truth for tasks like license plate decoding or biometric verification."
    )


# =============================================================================
# SECTION 3: EVALUATION LAB
# =============================================================================
with tab_eval:
    st.markdown("### 📊 Quantitative Evaluation Lab & Benchmark Experiments")
    st.markdown(
        "Benchmarking super-resolution algorithms according to formal computer vision research standards: "
        "Peak Signal-to-Noise Ratio (**PSNR**), Structural Similarity Index (**SSIM**), and Inference Latency."
    )

    # Benchmark Specifications Card
    st.markdown("#### ⚙️ Experimental Benchmark Specifications & Environment")
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        st.markdown(f"""
        - **Compute Platform:** `{config.get_device()}` ({'CUDA GPU Acceleration' if avail_cuda else 'Multi-Threaded CPU Engine'})
        - **Generator Architecture:** `RRDBNet` (ESRGAN)
        - **Total Parameters:** `16,697,987` (~16.7M verified)
        """)
    with b_col2:
        st.markdown(f"""
        - **Residual Blocks:** `23 RRDB Blocks` (69 Dense Sub-Layers)
        - **Scale Factor:** `4× Super-Resolution` (16× Pixel Density)
        - **Pretrained Checkpoint:** `{config.MODEL_WEIGHTS_FILENAME}`
        """)
    with b_col3:
        st.markdown(f"""
        - **Degradation Model:** `4× Anti-Aliased Bicubic Decimation (Pillow Standard)`
        - **Evaluation Metric Standard:** `PSNR` & `SSIM` on Luminance ($Y$-Channel)
        - **Border Cropping:** `4 Pixels` (DIV2K / NTIRE Standard)
        """)

    st.markdown("---")
    eval_c1, eval_c2 = st.columns([1.5, 1.0])

    with eval_c1:
        st.markdown("#### 📂 Reference Benchmark Dataset")
        default_dir = str(config.EVAL_HR_DIR)
        selected_eval_dir = st.text_input("High-Resolution Reference Directory:", value=default_dir)

        eval_path = Path(selected_eval_dir)
        num_ref_images = len(list(eval_path.glob("*.png"))) if eval_path.exists() else 0
        st.caption(f"Status: **{num_ref_images}** high-resolution reference targets discovered.")

    with eval_c2:
        st.markdown("#### 🚀 Benchmark Controls")
        run_bench_btn = st.button("▶️ Execute Full Benchmark Suite", type="primary", use_container_width=True)

    csv_file = config.OUTPUTS_DIR / "evaluation_results.csv"
    chart_file = config.OUTPUTS_DIR / "benchmark_comparison_chart.png"
    images_out_dir = config.OUTPUTS_DIR / "benchmark_visuals"

    if run_bench_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()

        def app_progress(current, total, name):
            progress_bar.progress(current / total)
            status_text.text(f"Evaluating [{current}/{total}]: {name}")

        with st.spinner("Running quantitative benchmark on reference dataset..."):
            try:
                results_dict = run_benchmark(
                    hr_dir=eval_path,
                    output_dir=config.OUTPUTS_DIR,
                    progress_callback=app_progress
                )
                progress_bar.progress(1.0)
                status_text.text("Benchmark complete!")
                st.success("✅ Evaluation suite completed successfully!")
            except Exception as e:
                st.error(f"❌ Benchmark execution failed: {str(e)}")

    # Display results if CSV exists
    if csv_file.exists():
        df_bench = pd.read_csv(csv_file)

        st.markdown("---")
        st.markdown("#### 📈 Aggregate Performance Summary")

        b_psnr_mean = df_bench["Bicubic_PSNR_Y_dB"].mean()
        e_psnr_mean = df_bench["ESRGAN_PSNR_Y_dB"].mean()
        b_ssim_mean = df_bench["Bicubic_SSIM_Y"].mean()
        e_ssim_mean = df_bench["ESRGAN_SSIM_Y"].mean()
        b_time_mean = df_bench["Bicubic_Time_sec"].mean()
        e_time_mean = df_bench["ESRGAN_Time_sec"].mean()
        total_eval = len(df_bench)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="saas-card">
                <div class="saas-card-label">Mean PSNR (Bicubic)</div>
                <div class="saas-card-val">{b_psnr_mean:.2f} dB</div>
                <div class="saas-card-sub">Luminance Y-Channel ({total_eval} imgs)</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="saas-card">
                <div class="saas-card-label">Mean PSNR (ESRGAN)</div>
                <div class="saas-card-val">{e_psnr_mean:.2f} dB</div>
                <div class="saas-card-sub">Luminance Y-Channel ({total_eval} imgs)</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="saas-card">
                <div class="saas-card-label">Mean SSIM (Bicubic / ESRGAN)</div>
                <div class="saas-card-val">{b_ssim_mean:.4f} / {e_ssim_mean:.4f}</div>
                <div class="saas-card-sub">Structural Similarity Index</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="saas-card">
                <div class="saas-card-label">Mean Latency (ESRGAN)</div>
                <div class="saas-card-val">{e_time_mean:.3f} s</div>
                <div class="saas-card-sub">Bicubic: {b_time_mean*1000:.1f} ms / image</div>
            </div>
            """, unsafe_allow_html=True)

        # 4-Way Visual Comparison Explorer
        if images_out_dir.exists() and not df_bench.empty:
            st.markdown("---")
            st.markdown("#### 🔬 Visual Comparison Explorer: 4-Way Ground Truth Inspection")
            st.caption("Select any reference target from the benchmark suite to inspect degradation and 4× super-resolution reconstruction against ground truth.")

            img_options = df_bench["Image"].tolist()
            default_idx = 6 if len(img_options) > 6 else 0
            selected_eval_image = st.selectbox(
                "Select Benchmark Reference Target:",
                options=img_options,
                index=default_idx
            )

            if selected_eval_image:
                stem = Path(selected_eval_image).stem
                lr_file = images_out_dir / f"{stem}_01_LR.png"
                bic_file = images_out_dir / f"{stem}_02_Bicubic.png"
                esr_file = images_out_dir / f"{stem}_03_ESRGAN.png"
                hr_file = images_out_dir / f"{stem}_04_HR_Reference.png"

                row_match = df_bench[df_bench["Image"] == selected_eval_image]
                if not row_match.empty:
                    row_data = row_match.iloc[0]
                    v1, v2, v3, v4 = st.columns(4)

                    with v1:
                        st.markdown("##### 🔍 1. Low-Res Input")
                        st.caption(f"Degraded 4× ({row_data['LR_Resolution']})")
                        if lr_file.exists():
                            st.image(str(lr_file), use_container_width=True)

                    with v2:
                        st.markdown("##### 📐 2. Bicubic Baseline")
                        st.caption(f"Upscaled 4× ({row_data['HR_Resolution']})")
                        if bic_file.exists():
                            st.image(str(bic_file), use_container_width=True)
                        st.markdown(
                            f"**PSNR:** `{row_data['Bicubic_PSNR_Y_dB']:.2f} dB`  \n"
                            f"**SSIM:** `{row_data['Bicubic_SSIM_Y']:.4f}`  \n"
                            f"**Time:** `{row_data['Bicubic_Time_sec']*1000:.1f} ms`"
                        )

                    with v3:
                        st.markdown("##### 🧠 3. ESRGAN Output")
                        st.caption(f"Reconstructed 4× ({row_data['HR_Resolution']})")
                        if esr_file.exists():
                            st.image(str(esr_file), use_container_width=True)
                        st.markdown(
                            f"**PSNR:** `{row_data['ESRGAN_PSNR_Y_dB']:.2f} dB`  \n"
                            f"**SSIM:** `{row_data['ESRGAN_SSIM_Y']:.4f}`  \n"
                            f"**Time:** `{row_data['ESRGAN_Time_sec']:.2f} s`"
                        )

                    with v4:
                        st.markdown("##### 🎯 4. Ground Truth")
                        st.caption(f"Reference Target ({row_data['HR_Resolution']})")
                        if hr_file.exists():
                            st.image(str(hr_file), use_container_width=True)
                        st.caption("Ground Truth Standard")

        # Comparative Visualization Chart
        if chart_file.exists():
            st.markdown("---")
            st.markdown("#### 📊 Comparative Metric Visualizations")
            st.image(str(chart_file), use_container_width=True)

        # Per-Image Quantitative Benchmark Table
        st.markdown("---")
        st.markdown("#### 📋 Detailed Per-Image Quantitative Benchmark Table")
        st.dataframe(df_bench, use_container_width=True)

        # Download CSV Button
        csv_bytes = csv_file.read_bytes()
        st.download_button(
            "⬇️ Export Complete Benchmark CSV",
            data=csv_bytes,
            file_name="sr_vision_lab_benchmark_results.csv",
            mime="text/csv",
            use_container_width=False
        )

        # The Perception-Distortion Trade-off Theoretical Insight
        st.markdown("---")
        st.info(
            "💡 **CVIP Research Insight — The Perception-Distortion Trade-off (Blau & Michaeli, CVPR 2018):**\n\n"
            "PSNR directly measures pixel-wise Mean Squared Error (MSE). When an algorithm outputs a blurred image (like Bicubic), "
            "it is predicting the conditional expectation (mean) of all possible high-frequency textures, which mathematically minimizes MSE and elevates PSNR.\n\n"
            "ESRGAN optimizes for human visual perceptual realism using a Relativistic GAN loss and VGG perceptual loss. "
            "It synthesizes sharp, plausible micro-edges. If a reconstructed micro-texture is sharp but shifted by even a single sub-pixel "
            "from the ground-truth reference, pixel-wise subtraction severely penalizes MSE, lowering PSNR despite human observers judging the output to be vastly superior."
        )

    else:
        st.warning("⚠️ No benchmark results found yet. Click 'Execute Full Benchmark Suite' above to run the evaluation.")


# =============================================================================
# SECTION 4: ABOUT THE PROJECT
# =============================================================================
with tab_about:
    st.markdown("### 📖 About SR Vision Lab")
    st.markdown("**Comparative Image Super-Resolution Using ESRGAN and Deep Learning**")

    st.markdown("""
    <div class="saas-card" style="margin-top: 14px;">
        <h4 style="color: #38BDF8; margin-top: 0;">🎯 Project Objective & Mission</h4>
        <p style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.6;">
            <strong>SR Vision Lab</strong> bridges fundamental classical Computer Vision and Image Processing (CVIP) 
            techniques with state-of-the-art deep generative networks. It demonstrates how traditional 2D polynomial 
            interpolation compares against a 23-block deep convolutional GAN architecture (ESRGAN) across real-world images 
            and reproducible benchmark targets.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Core CVIP Syllabus Concepts
    st.markdown("#### 📚 Core CVIP Syllabus Principles Implemented")
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.markdown("""
        <div class="saas-card">
            <h5 style="color: #06B6D4;">1. Sampling, Decimation & Aliasing</h5>
            <p style="font-size: 0.85rem; color: #94A3B8;">
                According to the <strong>Nyquist-Shannon Sampling Theorem</strong>, an image signal must be sampled at 
                least twice its maximum spatial frequency (<em>f_s &ge; 2 f_max</em>). Downsampling by a factor of 4 
                removes 93.75% of pixel samples, causing high-frequency components to fold back into lower frequencies 
                as aliasing, blur, and Moiré artifacts.
            </p>
            <h5 style="color: #06B6D4; margin-top: 12px;">2. Spline-Based Interpolation</h5>
            <p style="font-size: 0.85rem; color: #94A3B8;">
                Classical Bicubic interpolation fits a 16-point cubic spline across nearest neighbors using the Catmull-Rom 
                kernel (<em>a = -0.5</em>). While continuous and differentiable, it acts as a low-pass filter that cannot 
                reconstruct high frequencies destroyed during decimation.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c_col2:
        st.markdown("""
        <div class="saas-card">
            <h5 style="color: #06B6D4;">3. Deep Generator Network (RRDBNet)</h5>
            <p style="font-size: 0.85rem; color: #94A3B8;">
                ESRGAN utilizes the <strong>Residual-in-Residual Dense Block (RRDB)</strong> architecture. By removing 
                Batch Normalization layers, it avoids unpleasant halo artifacts and expands feature capacity with dense 
                residual connections and residual scaling factor <em>&beta; = 0.2</em>.
            </p>
            <h5 style="color: #06B6D4; margin-top: 12px;">4. Multi-Component Loss Functions</h5>
            <p style="font-size: 0.85rem; color: #94A3B8;">
                Unlike PSNR-oriented models trained purely on L1/MSE, ESRGAN combines:
                <br>&bull; <strong>Perceptual Loss:</strong> L1 distance on VGG-19 feature activations before ReLU.
                <br>&bull; <strong>Relativistic GAN Loss (RaGAN):</strong> Estimates whether real images are more realistic than fake images.
                <br>&bull; <strong>Content Loss:</strong> Pixel-wise L1 loss for low-frequency structural fidelity.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Verified Architecture Specifications
    st.markdown("---")
    st.markdown("#### 🔬 Verified Model Architecture Specifications")
    st.markdown(f"""
    | Architectural Hyperparameter | Verified Configuration | Engineering Significance |
    | :--- | :---: | :--- |
    | **Generator Topology** | `RRDBNet` | Multi-level residual learning with dense feature reuse |
    | **Number of RRDB Blocks** | `23 Blocks` | 69 Dense convolutional sub-layers |
    | **Growth Channels ($gc$)** | `32 Channels` | Rate of feature expansion within dense blocks |
    | **Feature Maps ($nf$)** | `64 Channels` | Internal trunk convolutional representation dimension |
    | **Total Parameters** | `16,697,987` (~16.7M) | Exact count calculated from state dict tensors |
    | **Scaling Factor** | `4×` | 16× spatial pixel density expansion |
    | **Weights Checkpoint** | `{config.MODEL_WEIGHTS_FILENAME}` | Official pretrained weights archive (63.8 MB) |
    | **Current Compute Device** | `{config.get_device()}` | {'GPU acceleration enabled' if avail_cuda else 'Multi-threaded CPU execution'} |
    """)

    # Literature & Citations
    st.markdown("---")
    st.markdown("#### 📑 Scientific References & Research Literature")
    st.markdown("""
    1. **Wang et al. (ECCV 2018):** *ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.* [arXiv:1809.00219](https://arxiv.org/abs/1809.00219)
    2. **Ledig et al. (CVPR 2017):** *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGAN).* [arXiv:1609.04802](https://arxiv.org/abs/1609.04802)
    3. **Blau & Michaeli (CVPR 2018):** *The Perception-Distortion Tradeoff.* [arXiv:1711.06077](https://arxiv.org/abs/1711.06077)
    4. **Agustsson & Timofte (CVPRW 2017):** *NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study (DIV2K).* [ETH CVL DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
    """)
