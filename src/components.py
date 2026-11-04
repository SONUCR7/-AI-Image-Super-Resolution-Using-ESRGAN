"""
UI Components Module for SR Vision Lab
Provides interactive HTML/JS widgets (Before/After comparison slider with zoom/pan),
glassmorphic stat cards, Lucide SVG icons, and responsive layouts.
"""

import io
import base64
from typing import Union
import numpy as np
from PIL import Image
import streamlit.components.v1 as components


def image_to_base64(img: Union[np.ndarray, Image.Image], format: str = "PNG") -> str:
    """Converts a numpy array or PIL Image into a base64 Data URI."""
    if isinstance(img, np.ndarray):
        pil_img = Image.fromarray(img)
    else:
        pil_img = img

    buffer = io.BytesIO()
    # Use WebP for fast transfer if supported, or PNG
    if format.upper() == "WEBP":
        pil_img.save(buffer, format="WEBP", quality=92)
        mime = "image/webp"
    else:
        pil_img.save(buffer, format="PNG")
        mime = "image/png"

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def render_comparison_slider(
    img_before: Union[np.ndarray, Image.Image],
    img_after: Union[np.ndarray, Image.Image],
    label_before: str = "Original (LR)",
    label_after: str = "ESRGAN (4x)",
    height: int = 560
) -> None:
    """
    Renders a responsive, interactive client-side Before/After comparison slider
    with draggable handle, zoom/pan controls, and full-screen toggle.
    """
    # Resize before image to match after image dimensions for perfect visual alignment in slider
    if isinstance(img_before, np.ndarray):
        pil_b = Image.fromarray(img_before)
    else:
        pil_b = img_before

    if isinstance(img_after, np.ndarray):
        pil_a = Image.fromarray(img_after)
    else:
        pil_a = img_after

    target_size = (pil_a.width, pil_a.height)
    if pil_b.size != target_size:
        # Scale with nearest-neighbor or bilinear for clear pixel difference visualization
        pil_b_aligned = pil_b.resize(target_size, resample=Image.Resampling.NEAREST)
    else:
        pil_b_aligned = pil_b

    b64_before = image_to_base64(pil_b_aligned, format="WEBP")
    b64_after = image_to_base64(pil_a, format="WEBP")

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
      * {{
        box-sizing: border-box;
        margin: 0;
        padding: 0;
        user-select: none;
        -webkit-user-select: none;
      }}
      body {{
        background: transparent;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        overflow: hidden;
      }}
      .viewer-wrapper {{
        position: relative;
        width: 100%;
        max-width: 1000px;
        height: {height - 60}px;
        background: #0b0f17;
        border-radius: 12px;
        border: 1px solid rgba(0, 210, 255, 0.2);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.6);
        overflow: hidden;
        cursor: ew-resize;
      }}
      .viewer-img-container {{
        position: absolute;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        display: flex;
        align-items: center;
        justify-content: center;
      }}
      .viewer-img {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: contain;
        pointer-events: none;
        transition: transform 0.15s ease-out;
      }}
      .after-clip {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
        width: 50%;
        border-right: 2px solid #00d2ff;
        box-shadow: 2px 0 15px rgba(0, 210, 255, 0.5);
      }}
      .after-clip img {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: contain;
      }}
      .slider-handle {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: #00d2ff;
        box-shadow: 0 0 15px #00d2ff, 0 4px 8px rgba(0,0,0,0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #0b0f17;
        font-weight: 900;
        font-size: 14px;
        pointer-events: none;
        z-index: 20;
      }}
      .label-badge {{
        position: absolute;
        top: 14px;
        padding: 5px 12px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.8px;
        border-radius: 6px;
        backdrop-filter: blur(8px);
        z-index: 15;
        text-transform: uppercase;
      }}
      .badge-before {{
        right: 14px;
        background: rgba(18, 24, 38, 0.75);
        color: #94a3b8;
        border: 1px solid rgba(255, 255, 255, 0.15);
      }}
      .badge-after {{
        left: 14px;
        background: rgba(0, 210, 255, 0.2);
        color: #00d2ff;
        border: 1px solid rgba(0, 210, 255, 0.4);
      }}
      .controls-bar {{
        display: flex;
        gap: 8px;
        margin-top: 10px;
        align-items: center;
        background: rgba(18, 24, 38, 0.7);
        padding: 6px 14px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.08);
      }}
      .ctrl-btn {{
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #e2e8f0;
        padding: 4px 10px;
        border-radius: 5px;
        font-size: 11px;
        cursor: pointer;
        transition: all 0.2s;
      }}
      .ctrl-btn:hover {{
        background: #00d2ff;
        color: #0b0f17;
        border-color: #00d2ff;
      }}
      .ctrl-label {{
        font-size: 11px;
        color: #64748b;
        margin-right: 4px;
      }}
    </style>
    </head>
    <body>
      <div class="viewer-wrapper" id="viewer">
        <div class="label-badge badge-after">{label_after}</div>
        <div class="label-badge badge-before">{label_before}</div>
        
        <!-- Base: Original / Before Image -->
        <img class="viewer-img" id="imgBefore" src="{b64_before}" alt="Original" />
        
        <!-- Clipped Overlay: Enhanced / After Image -->
        <div class="after-clip" id="clipDiv">
          <img class="viewer-img" id="imgAfter" src="{b64_after}" alt="Enhanced" />
        </div>
        
        <div class="slider-handle" id="handle">⮂ ⮃</div>
      </div>

      <div class="controls-bar">
        <span class="ctrl-label">Split:</span>
        <button class="ctrl-btn" onclick="setPos(25)">25%</button>
        <button class="ctrl-btn" onclick="setPos(50)">50%</button>
        <button class="ctrl-btn" onclick="setPos(75)">75%</button>
        <span class="ctrl-label" style="margin-left:8px;">Zoom:</span>
        <button class="ctrl-btn" onclick="adjustZoom(1.2)">+ Zoom</button>
        <button class="ctrl-btn" onclick="adjustZoom(0.8)">- Zoom</button>
        <button class="ctrl-btn" onclick="resetZoom()">Reset</button>
        <button class="ctrl-btn" onclick="toggleFullscreen()" style="margin-left:auto;">⛶ Fullscreen</button>
      </div>

      <script>
        const viewer = document.getElementById('viewer');
        const clipDiv = document.getElementById('clipDiv');
        const handle = document.getElementById('handle');
        const imgBefore = document.getElementById('imgBefore');
        const imgAfter = document.getElementById('imgAfter');

        let isDown = false;
        let zoom = 1.0;

        function updateSlider(clientX) {{
          const rect = viewer.getBoundingClientRect();
          let x = clientX - rect.left;
          x = Math.max(0, Math.min(x, rect.width));
          const pct = (x / rect.width) * 100;
          clipDiv.style.width = pct + '%';
          handle.style.left = pct + '%';
        }}

        function setPos(pct) {{
          clipDiv.style.width = pct + '%';
          handle.style.left = pct + '%';
        }}

        function adjustZoom(factor) {{
          zoom = Math.max(1.0, Math.min(3.5, zoom * factor));
          imgBefore.style.transform = `scale(${{zoom}})`;
          imgAfter.style.transform = `scale(${{zoom}})`;
        }}

        function resetZoom() {{
          zoom = 1.0;
          imgBefore.style.transform = 'scale(1.0)';
          imgAfter.style.transform = 'scale(1.0)';
          setPos(50);
        }}

        function toggleFullscreen() {{
          if (!document.fullscreenElement) {{
            viewer.requestFullscreen().catch(err => {{}});
          }} else {{
            document.exitFullscreen();
          }}
        }}

        viewer.addEventListener('mousedown', (e) => {{
          isDown = true;
          updateSlider(e.clientX);
        }});

        window.addEventListener('mouseup', () => {{ isDown = false; }});

        window.addEventListener('mousemove', (e) => {{
          if (!isDown) return;
          updateSlider(e.clientX);
        }});

        // Touch support
        viewer.addEventListener('touchstart', (e) => {{
          isDown = true;
          if (e.touches.length > 0) updateSlider(e.touches[0].clientX);
        }});
        window.addEventListener('touchend', () => {{ isDown = false; }});
        window.addEventListener('touchmove', (e) => {{
          if (!isDown) return;
          if (e.touches.length > 0) updateSlider(e.touches[0].clientX);
        }});

        // Initial 50% split
        setPos(50);
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=height)
