# High-Resolution Evaluation Reference Images

Place ground-truth reference images (such as DIV2K validation images or high-resolution PNG/JPG files) in this directory.

The automated evaluation pipeline (`scripts/evaluate.py`) will:
1. Load every image in this folder.
2. Downsample it by 4× using standard bicubic decimation to synthesize a low-resolution input.
3. Reconstruct the image using both Bicubic interpolation and ESRGAN.
4. Compute PSNR and SSIM against the original ground truth.
5. Export detailed metrics to `outputs/evaluation_results.csv`.
