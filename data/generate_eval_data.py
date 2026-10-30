"""
Generate Reference Evaluation and Sample Images
SR Vision Lab: Comparative Image Super-Resolution Using ESRGAN and Deep Learning

Creates 25 diverse high-resolution CVIP benchmark reference images in data/eval_hr/
and 3 sample test images in data/samples/ for reproducible evaluation.
Covers Nyquist limits, aliasing, step edges, natural foliage crops, typography,
fractals, cellular tissues, wavelets, PCB traces, and Moiré patterns.
"""

from pathlib import Path
import numpy as np
import cv2

DATA_DIR = Path(__file__).resolve().parent
EVAL_HR_DIR = DATA_DIR / "eval_hr"
SAMPLES_DIR = DATA_DIR / "samples"

EVAL_HR_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def create_radial_chirp(h: int = 384, w: int = 384) -> np.ndarray:
    """Siemens star / radial frequency chirp pattern to evaluate directional resolution & aliasing."""
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    r = np.hypot(x - cx, y - cy)
    theta = np.arctan2(y - cy, x - cx)
    
    spokes = 0.5 + 0.5 * np.cos(36 * theta)
    rings = 0.5 + 0.5 * np.sin(0.10 * (r ** 1.25))
    
    img = (0.6 * spokes + 0.4 * rings) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    b = img
    g = np.clip(img * (x / w), 0, 255).astype(np.uint8)
    r_ch = np.clip(img * (y / h), 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r_ch])


def create_geometric_structures(h: int = 384, w: int = 384) -> np.ndarray:
    """Crisp geometric structures, text, diagonals, and sharp step edges."""
    canvas = np.full((h, w, 3), 245, dtype=np.uint8)
    
    for i in range(16, w - 16, 16):
        thickness = 1 if i % 32 == 0 else 2
        cv2.line(canvas, (i, 16), (i, h - 16), (40, 40, 180), thickness)
    for j in range(16, h - 16, 16):
        cv2.line(canvas, (16, j), (w - 16, j), (180, 40, 40), 1)

    for radius in range(15, 100, 15):
        cv2.circle(canvas, (w // 2, h // 2), radius, (20, 150, 40), 2, cv2.LINE_AA)

    for k in range(-120, 120, 24):
        cv2.line(canvas, (80 + k, 40), (280 + k, 340), (120, 30, 200), 2, cv2.LINE_AA)

    cv2.putText(canvas, "SR VISION LAB", (40, 70), cv2.FONT_HERSHEY_DUPLEX, 1.0, (10, 10, 10), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Bicubic vs ESRGAN 4x", (40, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2, cv2.LINE_AA)
    cv2.putText(canvas, "CVIP Benchmark Reference", (40, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
    return canvas


def create_procedural_texture(h: int = 384, w: int = 384) -> np.ndarray:
    """Multi-scale textural checkerboard and brick pattern."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    brick_h, brick_w = 20, 40
    for row, y in enumerate(range(0, h, brick_h)):
        offset = (brick_w // 2) if (row % 2 == 1) else 0
        for x in range(-offset, w, brick_w):
            color = (
                int(160 + 40 * np.sin(x * 0.05)),
                int(80 + 30 * np.cos(y * 0.05)),
                int(40 + 20 * np.sin((x + y) * 0.02))
            )
            cv2.rectangle(canvas, (x + 2, y + 2), (x + brick_w - 2, y + brick_h - 2), color, -1)
            cv2.rectangle(canvas, (x, y), (x + brick_w, y + brick_h), (210, 210, 210), 1)
            
    noise = np.random.RandomState(42).normal(0, 12, (h, w, 3)).astype(np.int16)
    textured = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return textured


def create_synthetic_landscape(h: int = 384, w: int = 384) -> np.ndarray:
    """Natural scene: Sky gradient, layered mountain silhouettes, sun, and foreground trees."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / h
        b = int(220 * (1 - t) + 40 * t)
        g = int(140 * (1 - t) + 80 * t)
        r = int(70 * (1 - t) + 180 * t)
        canvas[y, :] = (b, g, r)

    cv2.circle(canvas, (290, 90), 35, (80, 220, 255), -1, cv2.LINE_AA)

    x_pts = np.linspace(0, w, 240)
    y_dist = 200 + 30 * np.sin(x_pts * 0.015) + 15 * np.cos(x_pts * 0.035)
    pts_dist = np.vstack([x_pts, y_dist]).T.astype(np.int32)
    pts_dist = np.vstack([pts_dist, [[w, h], [0, h]]])
    cv2.fillPoly(canvas, [pts_dist], (120, 80, 60))

    y_mid = 250 + 28 * np.sin(x_pts * 0.02 + 1.2) + 20 * np.cos(x_pts * 0.04)
    pts_mid = np.vstack([x_pts, y_mid]).T.astype(np.int32)
    pts_mid = np.vstack([pts_mid, [[w, h], [0, h]]])
    cv2.fillPoly(canvas, [pts_mid], (80, 110, 40))

    rng = np.random.RandomState(101)
    for tx in range(16, w, 22):
        th = rng.randint(35, 75)
        ty = h - 16
        cv2.line(canvas, (tx, ty), (tx, ty - th), (30, 45, 20), 2)
        cv2.circle(canvas, (tx, ty - th), rng.randint(10, 18), (25, 75, 20), -1)

    return canvas


def create_mandala_pattern(h: int = 384, w: int = 384) -> np.ndarray:
    """Complex kaleidoscopic mandala pattern testing high-frequency symmetric curves."""
    canvas = np.full((h, w, 3), 20, dtype=np.uint8)
    cx, cy = w // 2, h // 2
    colors = [(255, 180, 50), (50, 220, 255), (220, 50, 220), (80, 255, 120), (255, 255, 255)]

    for r in range(25, 175, 20):
        num_petals = 12
        for p in range(num_petals):
            angle = (p * 2 * np.pi) / num_petals
            px = int(cx + r * np.cos(angle))
            py = int(cy + r * np.sin(angle))
            col = colors[(r // 20) % len(colors)]
            cv2.circle(canvas, (px, py), r // 3, col, 2, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy), (px, py), (col[0] // 2, col[1] // 2, col[2] // 2), 1, cv2.LINE_AA)
    return canvas


def create_checkerboard_multiscale(h: int = 384, w: int = 384) -> np.ndarray:
    """Multi-scale checkerboard grids testing frequency cutoffs and Nyquist limits."""
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    quad_h, quad_w = h // 2, w // 2
    sizes = [4, 8, 16, 32]
    quads = [(0, 0), (0, quad_w), (quad_h, 0), (quad_h, quad_w)]
    
    for (qy, qx), sz in zip(quads, sizes):
        for y in range(qy, qy + quad_h, sz):
            for x in range(qx, qx + quad_w, sz):
                if ((y - qy) // sz + (x - qx) // sz) % 2 == 0:
                    cv2.rectangle(canvas, (x, y), (x + sz, y + sz), (25, 25, 25), -1)
        cv2.putText(canvas, f"{sz}px", (qx + 10, qy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2, cv2.LINE_AA)
    return canvas


def extract_or_generate_foliage(h: int = 384, w: int = 384) -> np.ndarray:
    """Photographic natural foliage crop from CVIP assignment image or synthetic dense vegetation."""
    ref_paths = [
        Path("D:/CVIP/beautiful-natural-image-1844362_1280.jpg"),
        Path("D:/CVIP/ASSIGNMENT-1/beautiful-natural-image-1844362_1280.jpg")
    ]
    for p in ref_paths:
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None and img.shape[0] >= h + 100 and img.shape[1] >= w + 100:
                # Crop lush trees / foliage region
                return img[200:200 + h, 100:100 + w]

    # Synthetic fallback: dense foliage fractal noise
    rng = np.random.RandomState(77)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(1200):
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        rad = rng.randint(6, 22)
        color = (rng.randint(15, 60), rng.randint(110, 210), rng.randint(20, 80))
        cv2.circle(canvas, (cx, cy), rad, color, -1)
    return canvas


def extract_or_generate_water_reflection(h: int = 384, w: int = 384) -> np.ndarray:
    """Photographic water ripple reflection crop from CVIP assignment image or wave synthesis."""
    ref_paths = [
        Path("D:/CVIP/beautiful-natural-image-1844362_1280.jpg"),
        Path("D:/CVIP/ASSIGNMENT-1/beautiful-natural-image-1844362_1280.jpg")
    ]
    for p in ref_paths:
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None and img.shape[0] >= h and img.shape[1] >= w:
                # Crop lake / reflection region at the bottom
                return img[img.shape[0] - h - 20:img.shape[0] - 20, 200:200 + w]

    # Synthetic fallback: water ripples
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y, x = np.ogrid[:h, :w]
    ripples = np.sin(x * 0.08) * np.cos(y * 0.15) * 40
    for i in range(h):
        t = i / h
        canvas[i, :, 0] = np.clip(180 + t * 40 + ripples[i, :], 0, 255)
        canvas[i, :, 1] = np.clip(120 + t * 50 + ripples[i, :], 0, 255)
        canvas[i, :, 2] = np.clip(40 + t * 30 + ripples[i, :], 0, 255)
    return canvas


def extract_or_generate_mountain_horizon(h: int = 384, w: int = 384) -> np.ndarray:
    """Photographic mountain horizon boundary crop from CVIP natural image or synthesis."""
    ref_paths = [
        Path("D:/CVIP/beautiful-natural-image-1844362_1280.jpg"),
        Path("D:/CVIP/ASSIGNMENT-1/beautiful-natural-image-1844362_1280.jpg")
    ]
    for p in ref_paths:
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None and img.shape[0] >= h and img.shape[1] >= w:
                # Crop mountain boundary against sky
                return img[100:100 + h, 400:400 + w]

    # Synthetic fallback: crisp alpine peaks
    canvas = np.full((h, w, 3), (230, 200, 160), dtype=np.uint8)
    pts = np.array([[0, h], [0, 220], [90, 110], [180, 190], [280, 90], [384, 210], [w, h]], np.int32)
    cv2.fillPoly(canvas, [pts], (90, 80, 70))
    return canvas


def create_butterfly_macro(h: int = 384, w: int = 384) -> np.ndarray:
    """Macro wing scale textures with fine veins and vibrant gradients."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    y, x = np.ogrid[:h, :w]
    r = np.hypot(x - cx, y - cy)
    theta = np.arctan2(y - cy, x - cx)
    
    veins = np.abs(np.sin(16 * theta + 0.05 * r))
    iridescence = 0.5 + 0.5 * np.sin(0.12 * r + 3 * theta)
    
    canvas[:, :, 0] = np.clip((iridescence * 200 + veins * 55), 0, 255)
    canvas[:, :, 1] = np.clip((np.cos(0.08 * r) * 120 + 130), 0, 255)
    canvas[:, :, 2] = np.clip((255 - iridescence * 180), 0, 255)
    return canvas


def create_urban_facade(h: int = 384, w: int = 384) -> np.ndarray:
    """Modern architectural skyscraper facade with regular window grids."""
    canvas = np.full((h, w, 3), 60, dtype=np.uint8)
    win_w, win_h = 24, 36
    margin_x, margin_y = 12, 16
    for y in range(20, h - win_h, win_h + margin_y):
        for x in range(20, w - win_w, win_w + margin_x):
            # Window reflection color
            refl = int(180 + 60 * np.sin((x + y) * 0.04))
            cv2.rectangle(canvas, (x, y), (x + win_w, y + win_h), (refl, refl - 20, 220), -1)
            cv2.rectangle(canvas, (x, y), (x + win_w, y + win_h), (200, 200, 200), 1)
            # Window mullion divider
            cv2.line(canvas, (x + win_w // 2, y), (x + win_w // 2, y + win_h), (80, 80, 80), 1)
    return canvas


def create_voronoi_cellular(h: int = 384, w: int = 384) -> np.ndarray:
    """Stochastic Voronoi cellular tissue testing irregular biological boundary reconstruction."""
    rng = np.random.RandomState(42)
    num_cells = 65
    points = rng.randint(10, min(h, w) - 10, size=(num_cells, 2))
    colors = rng.randint(40, 240, size=(num_cells, 3))
    
    # Distance transform
    y, x = np.indices((h, w))
    coords = np.stack([y, x], axis=-1)  # (H, W, 2)
    
    # Compute closest point
    dists = np.linalg.norm(coords[:, :, None, :] - points[None, None, :, :], axis=-1)
    closest = np.argmin(dists, axis=-1)
    
    canvas = colors[closest].astype(np.uint8)
    # Detect boundaries using Sobel filter
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 90)
    canvas[edges > 0] = (20, 20, 20)
    return canvas


def create_mandelbrot_fractal(h: int = 384, w: int = 384) -> np.ndarray:
    """Mandelbrot set boundary with infinite self-similar high-frequency detail."""
    xa, xb = -0.748, -0.744
    ya, yb = 0.100, 0.104
    max_iter = 100
    
    y, x = np.ogrid[:h, :w]
    c = (xa + (xb - xa) * x / w) + 1j * (ya + (yb - ya) * y / h)
    z = np.zeros_like(c)
    divtime = np.full(c.shape, max_iter, dtype=np.int32)
    
    for i in range(max_iter):
        mask = (np.abs(z) <= 2.0) & (divtime == max_iter)
        z[mask] = z[mask] ** 2 + c[mask]
        divtime[mask & (np.abs(z) > 2.0)] = i
        
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    norm_div = (divtime / max_iter * 255).astype(np.uint8)
    canvas[:, :, 0] = cv2.applyColorMap(norm_div, cv2.COLORMAP_JET)[:, :, 0]
    canvas[:, :, 1] = cv2.applyColorMap(norm_div, cv2.COLORMAP_JET)[:, :, 1]
    canvas[:, :, 2] = cv2.applyColorMap(norm_div, cv2.COLORMAP_JET)[:, :, 2]
    return canvas


def create_concentric_ellipses(h: int = 384, w: int = 384) -> np.ndarray:
    """High-density eccentric elliptical contours testing sub-pixel curvature."""
    canvas = np.full((h, w, 3), 250, dtype=np.uint8)
    cx, cy = w // 2, h // 2
    for r in range(12, 175, 7):
        color = (
            int(128 + 120 * np.sin(r * 0.1)),
            int(60 + 50 * np.cos(r * 0.1)),
            int(180 + 70 * np.sin(r * 0.05))
        )
        cv2.ellipse(canvas, (cx, cy), (r, int(r * 0.65)), 35, 0, 360, color, 1, cv2.LINE_AA)
    return canvas


def create_iso_resolution_chart(h: int = 384, w: int = 384) -> np.ndarray:
    """ISO 12233-style optical converging frequency wedges."""
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    cx, cy = w // 2, h // 2
    
    # Converging 5-line bursts in 4 cardinal quadrants
    for angle_deg in [0, 90, 180, 270]:
        rad = np.radians(angle_deg)
        for offset in np.linspace(-30, 30, 9):
            x1 = int(cx + 40 * np.cos(rad) - offset * np.sin(rad))
            y1 = int(cy + 40 * np.sin(rad) + offset * np.cos(rad))
            x2 = int(cx + 170 * np.cos(rad) - (offset * 0.1) * np.sin(rad))
            y2 = int(cy + 170 * np.sin(rad) + (offset * 0.1) * np.cos(rad))
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 0), 1, cv2.LINE_AA)
            
    cv2.circle(canvas, (cx, cy), 28, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (cx - 15, cy - 15), (cx + 15, cy + 15), (0, 0, 0), -1)
    return canvas


def create_perlin_terrain(h: int = 384, w: int = 384) -> np.ndarray:
    """Multi-octave procedural fractal terrain simulating rocky surface textures."""
    canvas = np.zeros((h, w), dtype=np.float32)
    rng = np.random.RandomState(88)
    for octave in [4, 8, 16, 32]:
        grid = rng.randn(octave, octave).astype(np.float32)
        upscaled = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        canvas += upscaled * (1.0 / octave)
        
    norm = ((canvas - canvas.min()) / (canvas.max() - canvas.min()) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_OCEAN)
    return colored


def create_micro_typography(h: int = 384, w: int = 384) -> np.ndarray:
    """Typography and optical character recognition (OCR) resolution target."""
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    texts = [
        ("AI IMAGE SUPER-RESOLUTION", 0.7, 2, (10, 10, 10)),
        ("Enhanced Super-Resolution GAN (ESRGAN)", 0.5, 1, (30, 30, 140)),
        ("The quick brown fox jumps over the lazy dog.", 0.42, 1, (20, 20, 20)),
        ("PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS.", 0.38, 1, (60, 60, 60)),
        ("0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ", 0.35, 1, (40, 100, 40)),
        ("Nyquist-Shannon Sampling Theorem: fs >= 2*fmax", 0.35, 1, (120, 40, 40)),
        ("CVIP 2026: Traditional Bicubic vs Deep Learning", 0.32, 1, (50, 50, 50)),
        ("Fine text sharpness and edge definition benchmark", 0.30, 1, (80, 80, 80)),
    ]
    curr_y = 35
    for txt, scale, thick, col in texts:
        cv2.putText(canvas, txt, (20, curr_y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)
        curr_y += int(38 * scale + 16)
    return canvas


def create_woodgrain_fibers(h: int = 384, w: int = 384) -> np.ndarray:
    """Anisotropic organic wood fiber grain patterns."""
    y, x = np.ogrid[:h, :w]
    rng = np.random.RandomState(99)
    noise = rng.normal(0, 1, (h, w)) * 8
    grain = np.sin((x * 0.08) + 12 * np.sin(y * 0.02) + noise * 0.1)
    norm = ((grain + 1) / 2 * 255).astype(np.uint8)
    
    b = np.clip(norm * 0.35 + 20, 0, 255).astype(np.uint8)
    g = np.clip(norm * 0.65 + 40, 0, 255).astype(np.uint8)
    r = np.clip(norm * 0.95 + 80, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])


def create_gabor_wavelets(h: int = 384, w: int = 384) -> np.ndarray:
    """Multi-orientation Gabor wavelet filter bank testing directional spatial frequencies."""
    canvas = np.full((h, w, 3), 128, dtype=np.uint8)
    angles = [0, 45, 90, 135]
    positions = [(96, 96), (288, 96), (96, 288), (288, 288)]
    
    for (cx, cy), theta_deg in zip(positions, angles):
        theta = np.radians(theta_deg)
        kernel = cv2.getGaborKernel((91, 91), 10.0, theta, 14.0, 0.5, 0, ktype=cv2.CV_32F)
        kernel_norm = ((kernel - kernel.min()) / (kernel.max() - kernel.min()) * 255).astype(np.uint8)
        kh, kw = kernel_norm.shape
        x1, y1 = cx - kw // 2, cy - kh // 2
        canvas[y1:y1 + kh, x1:x1 + kw, 0] = kernel_norm
        canvas[y1:y1 + kh, x1:x1 + kw, 1] = int(theta_deg * 1.5)
        canvas[y1:y1 + kh, x1:x1 + kw, 2] = 255 - kernel_norm
        cv2.putText(canvas, f"{theta_deg} deg", (cx - 25, cy + kh // 2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def create_pcb_traces(h: int = 384, w: int = 384) -> np.ndarray:
    """Printed Circuit Board (PCB) routing traces with 45-degree angle copper paths."""
    canvas = np.full((h, w, 3), (25, 75, 20), dtype=np.uint8)  # Solder mask green
    copper = (40, 180, 220)  # Golden copper
    
    # Horizontal bus
    for y in range(40, 140, 16):
        cv2.line(canvas, (20, y), (180, y), copper, 2, cv2.LINE_AA)
        cv2.line(canvas, (180, y), (240, y + 60), copper, 2, cv2.LINE_AA)
        cv2.line(canvas, (240, y + 60), (360, y + 60), copper, 2, cv2.LINE_AA)
        cv2.circle(canvas, (20, y), 4, (190, 190, 190), -1)  # Via
        cv2.circle(canvas, (360, y + 60), 4, (190, 190, 190), -1)

    # IC chip outline
    cv2.rectangle(canvas, (80, 240), (220, 350), (20, 20, 20), -1)
    cv2.rectangle(canvas, (80, 240), (220, 350), (140, 140, 140), 1)
    cv2.putText(canvas, "ESRGAN-IC", (95, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    for px in range(95, 210, 16):
        cv2.rectangle(canvas, (px, 230), (px + 6, 240), (200, 200, 200), -1)
        cv2.rectangle(canvas, (px, 350), (px + 6, 360), (200, 200, 200), -1)
    return canvas


def create_stained_glass(h: int = 384, w: int = 384) -> np.ndarray:
    """High-contrast stained glass polygon mosaic with dark boundaries."""
    rng = np.random.RandomState(55)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    pts = rng.randint(0, min(h, w), size=(35, 2))
    
    # Delaunay triangulation
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    for p in pts:
        subdiv.insert((float(p[0]), float(p[1])))
    triangles = subdiv.getTriangleList()
    
    for t in triangles:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        color = (rng.randint(50, 250), rng.randint(50, 250), rng.randint(50, 250))
        cv2.fillConvexPoly(canvas, np.array([pt1, pt2, pt3]), color, cv2.LINE_AA)
        cv2.polylines(canvas, [np.array([pt1, pt2, pt3])], True, (15, 15, 15), 3, cv2.LINE_AA)
    return canvas


def create_starfield_nebula(h: int = 384, w: int = 384) -> np.ndarray:
    """Astronomical Point Spread Functions (stars) and smooth luminous gas clouds."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    # Luminous gas cloud
    y, x = np.ogrid[:h, :w]
    c1 = np.exp(-((x - 140)**2 + (y - 160)**2) / (2 * 75**2))
    c2 = np.exp(-((x - 260)**2 + (y - 240)**2) / (2 * 65**2))
    canvas[:, :, 0] = np.clip(c1 * 180 + c2 * 80, 0, 255)
    canvas[:, :, 1] = np.clip(c1 * 40 + c2 * 140, 0, 255)
    canvas[:, :, 2] = np.clip(c1 * 220 + c2 * 190, 0, 255)

    # Pinpoint stars
    rng = np.random.RandomState(123)
    for _ in range(160):
        sx, sy = rng.randint(10, w - 10), rng.randint(10, h - 10)
        brightness = rng.randint(160, 255)
        rad = 1 if rng.rand() > 0.15 else 2
        cv2.circle(canvas, (sx, sy), rad, (brightness, brightness, brightness), -1)
    return canvas


def create_moire_interference(h: int = 384, w: int = 384) -> np.ndarray:
    """Dual rotated grid layers producing controlled optical Moiré patterns."""
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    layer1 = np.zeros((h, w), dtype=np.uint8)
    layer2 = np.zeros((h, w), dtype=np.uint8)
    
    # Layer 1: vertical lines
    for x in range(0, w, 6):
        cv2.line(layer1, (x, 0), (x, h), 255, 2)
        
    # Layer 2: rotated lines by 9 degrees
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 9, 1.0)
    layer2 = cv2.warpAffine(layer1, M, (w, h))
    
    moire = cv2.bitwise_and(layer1, layer2)
    canvas[moire > 0] = (20, 20, 180)
    canvas[layer1 > 0] = np.minimum(canvas[layer1 > 0], (180, 40, 40))
    return canvas


def create_halftone_dots(h: int = 384, w: int = 384) -> np.ndarray:
    """Variable-diameter halftone dot screening."""
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    step = 16
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            # Gradient intensity determines radius
            intensity = (x / w + y / h) / 2.0
            radius = int(1 + intensity * 6)
            cv2.circle(canvas, (x, y), radius, (20, 20, 20), -1, cv2.LINE_AA)
    return canvas


def create_textile_weave(h: int = 384, w: int = 384) -> np.ndarray:
    """Cross-hatched high-density textile fabric weave."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    thread_w = 8
    for y in range(0, h, thread_w):
        for x in range(0, w, thread_w):
            is_warp = ((x // thread_w) + (y // thread_w)) % 2 == 0
            if is_warp:
                col = (180, 120, 50)  # Warp thread
            else:
                col = (60, 80, 190)   # Weft thread
            cv2.rectangle(canvas, (x, y), (x + thread_w - 1, y + thread_w - 1), col, -1)
            cv2.rectangle(canvas, (x, y), (x + thread_w, y + thread_w), (30, 30, 30), 1)
    return canvas


def main():
    print("Generating comprehensive 25-image CVIP benchmark reference suite in data/eval_hr/...")
    
    benchmark_suite = {
        "eval_01_chirp_rings.png": create_radial_chirp(),
        "eval_02_geometric_edges.png": create_geometric_structures(),
        "eval_03_textured_bricks.png": create_procedural_texture(),
        "eval_04_synthetic_landscape.png": create_synthetic_landscape(),
        "eval_05_mandala_curves.png": create_mandala_pattern(),
        "eval_06_checkerboard_multiscale.png": create_checkerboard_multiscale(),
        "eval_07_natural_foliage_crop.png": extract_or_generate_foliage(),
        "eval_08_natural_water_reflection.png": extract_or_generate_water_reflection(),
        "eval_09_natural_mountain_horizon.png": extract_or_generate_mountain_horizon(),
        "eval_10_butterfly_wing_macro.png": create_butterfly_macro(),
        "eval_11_urban_facade_grid.png": create_urban_facade(),
        "eval_12_voronoi_cellular_tissue.png": create_voronoi_cellular(),
        "eval_13_mandelbrot_fractal_edge.png": create_mandelbrot_fractal(),
        "eval_14_concentric_ellipses.png": create_concentric_ellipses(),
        "eval_15_iso_resolution_chart.png": create_iso_resolution_chart(),
        "eval_16_perlin_noise_terrain.png": create_perlin_terrain(),
        "eval_17_micro_typography.png": create_micro_typography(),
        "eval_18_woodgrain_fibers.png": create_woodgrain_fibers(),
        "eval_19_wavelet_gabor_patches.png": create_gabor_wavelets(),
        "eval_20_circuit_board_traces.png": create_pcb_traces(),
        "eval_21_stained_glass_mosaic.png": create_stained_glass(),
        "eval_22_starfield_nebula.png": create_starfield_nebula(),
        "eval_23_moire_interference.png": create_moire_interference(),
        "eval_24_halftone_dot_matrix.png": create_halftone_dots(),
        "eval_25_fiber_weave_textile.png": create_textile_weave(),
    }

    for name, img_bgr in benchmark_suite.items():
        out_path = EVAL_HR_DIR / name
        cv2.imwrite(str(out_path), img_bgr)
        print(f"Saved [{len(list(EVAL_HR_DIR.glob('*.png')))}/25]: {out_path.name} ({img_bgr.shape[1]}x{img_bgr.shape[0]} px)")

    # Save 3 sidebar sample demo images in data/samples/
    sample_images = {
        "sample_butterfly.png": create_butterfly_macro(384, 384),
        "sample_urban_architecture.png": create_urban_facade(384, 384),
        "sample_landscape.png": create_synthetic_landscape(384, 384),
    }

    for name, img_bgr in sample_images.items():
        out_path = SAMPLES_DIR / name
        cv2.imwrite(str(out_path), img_bgr)
        print(f"Saved demo sample: {out_path.name} ({img_bgr.shape[1]}x{img_bgr.shape[0]} px)")

    print(f"\nSuccessfully generated {len(benchmark_suite)} reference benchmark images in: {EVAL_HR_DIR}")


if __name__ == "__main__":
    main()
