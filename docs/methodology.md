# 🔬 Methodology & CVIP Theoretical Principles

This document explains the mathematical foundations of classical image interpolation and deep convolutional super-resolution with ESRGAN.

---

## 1. The Ill-Posed Super-Resolution Problem

Single Image Super-Resolution (SISR) seeks to recover a high-resolution image $I_{HR}$ from its low-resolution counterpart $I_{LR}$. The degradation process is modeled as:

$$I_{LR} = (I_{HR} \otimes \mathbf{k}) \downarrow_s + \mathbf{n}$$

Where:
- $\otimes$ denotes convolution with a blur kernel $\mathbf{k}$.
- $\downarrow_s$ represents spatial decimation by scale factor $s$ ($s=4$).
- $\mathbf{n}$ represents additive noise.

SISR is fundamentally an **ill-posed inverse problem**: for any given low-resolution patch, infinite possible high-resolution representations could have produced it when downsampled.

---

## 2. Classical Baseline: Bicubic Interpolation

Traditional methods reconstruct missing pixels using local spatial interpolation without prior knowledge of natural image structures.

Bicubic interpolation uses a 16-pixel ($4\times 4$) neighborhood and a separable 1D cubic polynomial convolution kernel (Catmull-Rom spline with $a = -0.5$):

$$W(x) = \begin{cases} 
(a + 2)|x|^3 - (a + 3)|x|^2 + 1 & \text{for } |x| \le 1 \\
a|x|^3 - 5a|x|^2 + 8a|x| - 4a & \text{for } 1 < |x| < 2 \\
0 & \text{otherwise}
\end{cases}$$

### Theoretical Characteristics:
1. **Continuous Derivatives:** Matches both function values and first derivatives at sample points, producing smoother transitions than nearest-neighbor or bilinear interpolation.
2. **Frequency Truncation:** Bicubic filtering acts as an approximate low-pass filter. It cannot synthesize missing high-frequency Fourier components lost during downsampling, leading to pronounced blurring and loss of fine textures.
3. **Artifacts:** Ringing effects (Gibbs phenomenon) and overshoot halos frequently appear near sharp edges.

---

## 3. Deep Learning: ESRGAN Architecture & Innovations

ESRGAN (Wang et al., ECCV 2018) overcomes classical smoothing by learning structural priors of natural scenes through a deep generator and relativistic adversarial training.

### A. Residual-in-Residual Dense Block (RRDB)
Standard residual networks (e.g. SRResNet) use basic residual blocks. ESRGAN introduces the **RRDB**, which stacks multiple Residual Dense Blocks (RDB):
- Within each RDB, 5 convolutional layers are densely interconnected: every layer receives the concatenated feature maps of all preceding layers.
- Multi-level residual connections allow gradients to bypass deep layers during backpropagation, enabling 23 RRDB blocks (~69 dense blocks) to train stably without gradient vanishing.
- Each residual branch is multiplied by a scaling factor of $0.2$ before addition to maintain numerical stability.

### B. Removal of Batch Normalization (BN)
Original GAN architectures used Batch Normalization to stabilize training. However, in super-resolution:
- BN normalizes feature representations using mini-batch mean and variance, washing out fine edge contrasts and subtle texture dynamics.
- BN introduces unpleasant boundary artifacts and color shifts.
- ESRGAN eliminates all BN layers, saving ~30% computational memory and improving natural dynamic range.

### C. Multi-Task Loss Formulation
ESRGAN is trained using a composite objective function:

$$\mathcal{L}_G = \mathcal{L}_{\text{percep}} + \lambda \mathcal{L}_{G}^{\text{Ra}} + \eta \mathcal{L}_1$$

1. **Perceptual Loss ($\mathcal{L}_{\text{percep}}$):**  
   Computed on VGG-19 features **before activation layers**:
   $$\mathcal{L}_{\text{percep}} = \frac{1}{V} \sum \|\phi_{i, j}(I_{HR}) - \phi_{i, j}(G(I_{LR}))\|_1$$
   Using features before activation prevents sparse representations and avoids brightness inconsistency.

2. **Relativistic Average GAN Loss ($\mathcal{L}_{G}^{\text{Ra}}$):**  
   Standard GAN discriminators judge whether an image is real or fake. The Relativistic Discriminator instead estimates the probability that a real image is relatively more realistic than a fake one:
   $$D_{\text{Ra}}(I_{HR}, I_{SR}) = \sigma(C(I_{HR}) - \mathbb{E}[C(I_{SR})])$$
   This drives the generator to synthesize sharp, natural textures rather than simply adding random high-frequency noise.

3. **Content Loss ($\mathcal{L}_1$):**  
   Pixel-wise L1 distance between reconstructed and ground truth images, providing low-frequency structural stability.
