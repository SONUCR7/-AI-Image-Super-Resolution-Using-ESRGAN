# 📏 Evaluation Protocol & Metric Methodology

This document outlines the evaluation protocols, mathematical formulas, and scientific reporting standards used in **SR Vision Lab**.

---

## 1. Degradation Model

To ensure reproducible quantitative benchmarking matching NTIRE and PIRM literature:
1. High-resolution reference images $I_{HR}$ are downscaled by scale factor $s=4$ using bicubic interpolation (`cv2.INTER_CUBIC`):
   $$I_{LR} = \text{BicubicDown}(I_{HR}, s=4)$$
2. The downsampled input $I_{LR}$ is fed to the reconstruction algorithms (Bicubic and ESRGAN).
3. The reconstructed output $I_{SR}$ is aligned spatially to $I_{HR}$.

---

## 2. Quantitative Metrics

### A. Peak Signal-to-Noise Ratio (PSNR)
PSNR measures pixel-level reconstruction fidelity in decibels (dB):

$$\text{MSE} = \frac{1}{M \times N} \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} [I_{HR}(x, y) - I_{SR}(x, y)]^2$$

$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right) = 20 \cdot \log_{10}\left(\frac{255}{\sqrt{\text{MSE}}}\right)$$

- Higher is typically considered better in fidelity-oriented tasks.
- If images are identical ($\text{MSE} = 0$), PSNR is capped at $100.0\text{ dB}$.

### B. Structural Similarity Index (SSIM)
SSIM (Wang et al., 2004) measures structural degradation, luminance distortion, and contrast preservation:

$$\text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

- Evaluated using an $11\times 11$ Gaussian sliding window with dynamic range $255$.
- Ranges between $-1.0$ and $+1.0$ (values for 4x super-resolution typically range from $0.6$ to $0.98$).

---

## 3. Benchmark Standards

1. **Luminance ($Y$-channel) Evaluation:**  
   In CVIP super-resolution research (SRCNN, EDSR, RCAN, ESRGAN), metrics are primarily evaluated on the luminance ($Y$) channel of standard ITU-R BT.601 YCbCr:
   $$Y = 0.257 R + 0.504 G + 0.098 B + 16$$
   The human visual system is most sensitive to structural variations in luminance.

2. **Boundary Shaving (Border Cropping):**  
   Boundary pixels equal to the scale factor ($4\text{ px}$) are cropped from each image edge before computing metrics to avoid border-padding extrapolation artifacts.

3. **Spatial Dimension Alignment:**  
   If odd input dimensions cause rounding differences during downsampling/upsampling, both images are cropped to their minimum common dimensions.

---

## 4. The Perception-Distortion Trade-Off

A crucial CVIP theoretical insight documented by **Blau & Michaeli (CVPR 2018)**:

> **Perception-Distortion Theorem:**  
> In image restoration, distortion (measured by MSE / PSNR) and perceptual quality (human visual sharpness and realism) are inherently at trade-offs. Minimizing MSE mathematically forces an estimator to output the average of all plausible high-frequency textures, producing smooth, blurred images.

Consequently:
- **Bicubic Interpolation** and L2-trained models often achieve higher or comparable PSNR because their blurry outputs avoid large penalties when pixel phases misalign.
- **ESRGAN** synthesizes rich, authentic micro-textures that match human perceptual expectations, but slight sub-pixel phase shifts lower its raw PSNR score.
