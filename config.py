"""
Configuration Module for SR Vision Lab
Comparative Image Super-Resolution Using ESRGAN and Deep Learning
"""

import os
# Prevent OpenMP runtime collision on Windows (libiomp5md.dll duplicate initialization)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import torch

# Optimize PyTorch CPU multi-threading
if not torch.cuda.is_available():
    cpu_cores = os.cpu_count() or 4
    torch.set_num_threads(cpu_cores)

# Base directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
EVAL_HR_DIR = DATA_DIR / "eval_hr"
SAMPLES_DIR = DATA_DIR / "samples"
OUTPUTS_DIR = BASE_DIR / "outputs"
DOCS_DIR = BASE_DIR / "docs"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
EVAL_HR_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Super-Resolution Model Configuration
SCALE_FACTOR = 4
MODEL_NAME = "ESRGAN_x4"
MODEL_WEIGHTS_FILENAME = "RRDB_ESRGAN_x4.pth"
MODEL_WEIGHTS_PATH = MODELS_DIR / MODEL_WEIGHTS_FILENAME

# Model download URLs (primary and mirrors)
MODEL_URLS = [
    # HuggingFace databuzzword mirror (Direct link)
    "https://huggingface.co/databuzzword/esrgan/resolve/main/RRDB_ESRGAN_x4.pth",
    # HuggingFace suiOPS mirror
    "https://huggingface.co/suiOPS/upscaler/resolve/main/RRDB_ESRGAN_x4.pth",
    # HuggingFace kadirnar mirror alternative
    "https://huggingface.co/kadirnar/RRDB_ESRGAN_x4/resolve/main/RRDB_ESRGAN_x4.pth"
]

# RRDBNet Architecture Hyperparameters (Standard ESRGAN x4)
RRDB_CONFIG = {
    "in_nc": 3,
    "out_nc": 3,
    "nf": 64,      # Number of feature maps
    "nb": 23,      # Number of RRDB blocks
    "gc": 32       # Growth channels
}

# Image validation constraints
MAX_IMAGE_SIZE_MB = 15.0
MAX_IMAGE_DIMENSION = 2048  # Maximum upload dimension
RECOMMENDED_MAX_CPU_DIMENSION = 512  # For CPU latency warning
MIN_IMAGE_DIMENSION = 16
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Hardware Device Selection
def get_device() -> torch.device:
    """Returns torch.device('cuda') if CUDA is available, else torch.device('cpu')."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
