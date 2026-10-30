# 🧠 Pretrained Model Weights: ESRGAN x4

This directory manages the deep learning model weights used by **SR Vision Lab** for 4× Single-Image Super-Resolution.

---

## 1. Model Checkpoint Details

- **Model Identifier:** `RRDB_ESRGAN_x4.pth`
- **Architecture:** Residual-in-Residual Dense Block Network (`RRDBNet`)
- **Scaling Factor:** 4× spatial upscaling ($16\times$ pixel count increase)
- **Number of RRDB Blocks:** 23 (containing 69 Dense Blocks)
- **Intermediate Feature Channels ($nf$):** 64
- **Growth Channels ($gc$):** 32
- **Total Trainable Parameters:** 16,697,987 (~16.70 Million)
- **Checkpoint File Size:** ~63.8 MB (66,971,689 bytes)
- **Primary Citation:** 
  > *Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., Loy, C. C., Qiao, Y., & Tang, X.* (2018).  
  > **ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.**  
  > European Conference on Computer Vision (ECCV) Workshops (PIRM 2018 Super-Resolution Challenge 1st Place).  
  > [arXiv:1809.00219](https://arxiv.org/abs/1809.00219)

---

## 2. Automated Download

If the weights file is missing, you can automatically download it using the built-in downloader:

```powershell
python models/download_weights.py
```

The script features multi-mirror fallback and automatic file integrity validation.

---

## 3. Verified Mirror Sources

If you prefer to download manually using `curl` or a web browser:

1. **HuggingFace Primary Mirror:**  
   [https://huggingface.co/databuzzword/esrgan/resolve/main/RRDB_ESRGAN_x4.pth](https://huggingface.co/databuzzword/esrgan/resolve/main/RRDB_ESRGAN_x4.pth)

2. **HuggingFace suiOPS Mirror:**  
   [https://huggingface.co/suiOPS/upscaler/resolve/main/RRDB_ESRGAN_x4.pth](https://huggingface.co/suiOPS/upscaler/resolve/main/RRDB_ESRGAN_x4.pth)

Place the downloaded `RRDB_ESRGAN_x4.pth` file directly into this `models/` directory.

---

## 4. Git Exclusion Policy

Per best practices, binary weights files (`*.pth`, `*.pt`, `*.onnx`) are excluded from Git commits via `.gitignore` to maintain a lightweight repository footprint.
