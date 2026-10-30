# 🛠️ Troubleshooting Guide

Common issues encountered on Windows, CPU-only systems, and Anaconda/Python environments, along with verified solutions.

---

## 1. OpenMP Runtime Collision (`libiomp5md.dll`)

### Symptom:
```text
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

### Cause:
On Windows systems with Anaconda, PyTorch, OpenCV, and Intel MKL each link their own copy of the OpenMP multi-threading runtime.

### Solution:
SR Vision Lab automatically handles this at the top of `config.py` and entrypoint scripts:
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```
If running external custom scripts, export this environment variable in PowerShell:
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

---

## 2. Model Weights Not Found

### Symptom:
```text
FileNotFoundError: Model weights file not found at 'models/RRDB_ESRGAN_x4.pth'
```

### Solution:
Run the automated downloader to fetch the authentic weights from reliable HuggingFace mirrors:
```powershell
python models/download_weights.py
```
Alternatively, manually download `RRDB_ESRGAN_x4.pth` (63.8 MB) and place it in the `models/` directory.

---

## 3. High Latency on CPU Laptops

### Symptom:
Processing takes longer than 15 seconds per image.

### Explanation & Tips:
ESRGAN's generator contains **16.70 Million parameters** and executes ~50+ GFLOPs per megapixel. On CPU-only laptops:
1. Use input resolutions between $128\times 128$ and $384\times 384$ px for fast interactive inference (typically 1 to 4 seconds).
2. For larger inputs, SR Vision Lab automatically applies **tiled inference** to prevent memory thrashing.
3. If an NVIDIA GPU is available, ensure CUDA-enabled PyTorch is installed:
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

---

## 4. Corrupted Image Uploads

### Symptom:
```text
Input Validation Error: Image decoding error or corrupted stream
```

### Solution:
Ensure the uploaded file is a valid 8-bit image in JPG, PNG, or WebP format. SR Vision Lab uses two-pass Pillow verification (`verify()` and full decode) to detect incomplete or damaged files.
