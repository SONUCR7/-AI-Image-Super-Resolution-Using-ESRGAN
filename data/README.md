# 📂 Benchmark Datasets: DIV2K & 25-Target Reference Evaluation Suite

This directory contains reference imagery for evaluating super-resolution algorithms according to formal CVIP standards.

---

## 1. DIV2K Primary Benchmark

The **DIV2K dataset** (Diverse 2K Resolution Images) is the standard benchmark used in computer vision super-resolution challenges (NTIRE / PIRM).

- **Official Dataset Portal:** [https://data.vision.ee.ethz.ch/cvl/DIV2K/](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- **Recommended Split for Evaluation:** Validation HR images (100 images, `0801.png` to `0900.png`).
- **Resolution:** 2K high-definition RGB images (approx. 2048 × 1080 to 2048 × 1536 px).
- **License / Terms of Use:** Academic and educational research purposes (Computer Vision Laboratory, ETH Zurich).

### How to use external DIV2K images in SR Vision Lab:
1. Download the validation archive from the official portal: `DIV2K_valid_HR.zip` (~1.1 GB).
2. Unpack the images into `data/eval_hr/` or any custom folder.
3. In the web dashboard (**Evaluation Lab** tab), enter the directory path and click **Execute Full Benchmark Suite**, or run via CLI:
   ```powershell
   python scripts/evaluate.py --hr-dir data/eval_hr
   ```

---

## 2. Standard 25-Target Benchmark Reference Suite (`data/eval_hr/`)

To enable **instant, reproducible benchmarking without requiring multi-gigabyte downloads**, SR Vision Lab includes a comprehensive 25-image evaluation suite. Each image isolates and tests specific fundamental CVIP phenomena:

| ID | Filename | Resolution | Evaluation Purpose | CVIP Concepts Tested |
| :---: | :--- | :---: | :--- | :--- |
| 01 | `eval_01_chirp_rings.png` | 384 × 384 | Radial Siemens star frequency chirp | Directional resolution, spatial aliasing & Nyquist limits |
| 02 | `eval_02_geometric_edges.png` | 384 × 384 | Antialiased lines, circles, and sharp text | Edge sharpness vs. Gibbs ringing / halo artifacts |
| 03 | `eval_03_textured_bricks.png` | 384 × 384 | Multi-scale masonry & stochastic noise | Micro-texture hallucination vs. smoothing |
| 04 | `eval_04_synthetic_landscape.png` | 384 × 384 | Twilight gradient, mountain silhouettes | Luminance transitions & smooth silhouette gradients |
| 05 | `eval_05_mandala_curves.png` | 384 × 384 | Kaleidoscopic multi-petal geometry | High-frequency symmetry & phase accuracy |
| 06 | `eval_06_checkerboard_multiscale.png` | 384 × 384 | Cascaded checkerboard grids (4px to 32px) | Spatial frequency cutoffs & pixel decimation aliasing |
| 07 | `eval_07_natural_foliage_crop.png` | 384 × 384 | Dense natural photographic tree foliage | Real-world stochastic texture reconstruction |
| 08 | `eval_08_natural_water_reflection.png` | 384 × 384 | Photographic water ripples & reflections | Continuous wave phase reconstruction & specular detail |
| 09 | `eval_09_natural_mountain_horizon.png` | 384 × 384 | Photographic alpine horizon against sky | High-contrast natural boundary edge definition |
| 10 | `eval_10_butterfly_wing_macro.png` | 384 × 384 | Macro wing scales and fine organic veins | Fine line continuity & color gradient fidelity |
| 11 | `eval_11_urban_facade_grid.png` | 384 × 384 | Skyscraper window grids & mullions | Periodic structural geometry & grid preservation |
| 12 | `eval_12_voronoi_cellular_tissue.png` | 384 × 384 | Stochastic Voronoi cellular boundaries | Irregular polygon border reconstruction |
| 13 | `eval_13_mandelbrot_fractal_edge.png` | 384 × 384 | Infinite self-similar fractal boundary | Sub-pixel scale-invariance & boundary stability |
| 14 | `eval_14_concentric_ellipses.png` | 384 × 384 | High-density eccentric elliptical contours | Sub-pixel curvature & anti-aliased staircasing |
| 15 | `eval_15_iso_resolution_chart.png` | 384 × 384 | ISO 12233 optical frequency wedges | Modulation Transfer Function (MTF) limiting resolution |
| 16 | `eval_16_perlin_noise_terrain.png` | 384 × 384 | Multi-octave Perlin fractal terrain | Continuous elevation color-space preservation |
| 17 | `eval_17_micro_typography.png` | 384 × 384 | Multi-size typography (7pt to 24pt) | Optical character recognition (OCR) legibility |
| 18 | `eval_18_woodgrain_fibers.png` | 384 × 384 | Anisotropic organic wood fiber grain | Directional flow alignment and micro-texture |
| 19 | `eval_19_wavelet_gabor_patches.png` | 384 × 384 | Multi-angle Gabor wavelet kernels | Directional spatial frequency sensitivity (0°, 45°, 90°, 135°) |
| 20 | `eval_20_circuit_board_traces.png` | 384 × 384 | PCB routing traces & IC semiconductor pins | 45-degree angle copper trace continuity |
| 21 | `eval_21_stained_glass_mosaic.png` | 384 × 384 | High-contrast Delaunay polygon mosaic | Strong step-edge preservation without color bleeding |
| 22 | `eval_22_starfield_nebula.png` | 384 × 384 | Astronomical Point Spread Functions (stars) | Point source preservation and low-contrast cloud gradients |
| 23 | `eval_23_moire_interference.png` | 384 × 384 | Dual rotated high-frequency line grids | Phase sensitivity & optical Moiré interference |
| 24 | `eval_24_halftone_dot_matrix.png` | 384 × 384 | Variable-diameter CMYK screening dots | Micro-circular geometry & dot-gain distortion |
| 25 | `eval_25_fiber_weave_textile.png` | 384 × 384 | Dense warp-and-weft fabric cross-hatch | Cross-hatched thread boundary reconstruction |

To regenerate or expand these benchmark images:
```powershell
python data/generate_eval_data.py
```

---

## 3. Demo Sample Images (`data/samples/`)

Located in `data/samples/`, these 3 images provide instant 1-click test patterns in the Streamlit web application sidebar:
- `sample_butterfly.png` (Macro biology)
- `sample_urban_architecture.png` (Geometric architecture)
- `sample_landscape.png` (Natural gradients)
