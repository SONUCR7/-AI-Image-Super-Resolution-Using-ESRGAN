# 🏗️ Project Architecture & Data Flow

This document details the software architecture, modular component structure, and tensor data lifecycle of **SR Vision Lab**.

---

## 1. High-Level Modular Design

```text
SR-Vision-Lab/
├── app.py                      # Interactive Streamlit Web UI & Dashboard
├── config.py                   # Central configuration, hardware detection, paths
├── requirements.txt            # Constrained dependencies
│
├── models/
│   ├── download_weights.py     # Automated weights downloader with multi-mirror fallback
│   ├── RRDB_ESRGAN_x4.pth      # 4x ESRGAN weights (63.8 MB, 16.7M parameters)
│   └── README.md               # Checkpoint specifications & citation
│
├── src/
│   ├── __init__.py             # Package initialization
│   ├── rrdb_net.py             # Pure PyTorch RRDBNet architecture (23 RRDB blocks)
│   ├── model_loader.py         # Checkpoint adaptation, memory caching, device routing
│   ├── inference.py            # Direct & memory-friendly tiled 4x super-resolution
│   ├── image_processing.py     # Bicubic baseline, degradation, alignment, RGB standardizer
│   ├── metrics.py              # PSNR & SSIM on Luminance (Y) channel & RGB
│   └── validation.py           # Robust stream verification, bounds, and corrupt file checks
│
├── scripts/
│   ├── evaluate.py             # Standalone quantitative benchmark pipeline (CSV & report)
│   └── smoke_test.py           # Automated end-to-end system health check
│
├── data/
│   ├── eval_hr/                # High-resolution benchmark reference images
│   ├── samples/                # Instant-demo benchmark sample images
│   └── README.md               # DIV2K dataset guide and terms of use
│
├── outputs/
│   ├── evaluation_results.csv  # Per-image benchmark measurements
│   ├── evaluation_summary.md   # Executive summary report
│   ├── benchmark_comparison_chart.png # Publication-grade comparison chart
│   └── README.md
│
├── tests/
│   ├── test_validation.py      # Input boundary and corrupt file tests
│   ├── test_image_processing.py# Bicubic scaling, degradation, alignment tests
│   ├── test_metrics.py         # PSNR and SSIM mathematical property tests
│   └── test_model_smoke.py     # RRDBNet architecture and inference tests
│
└── docs/
    ├── project_architecture.md  # System and tensor flow documentation
    ├── methodology.md           # Mathematical and CVIP theoretical foundations
    ├── evaluation_protocol.md   # Benchmark standards and protocol rules
    └── troubleshooting.md       # Windows and CPU runtime guide
```

---

## 2. End-to-End Inference Dataflow

```
[ User Upload / Sample Stream ] (JPG, PNG, WebP)
             │
             ▼
[ validation.validate_image_file ] ──> Checks format, bytes, bounds (16 to 2048 px)
             │
             ▼
[ image_processing.ensure_rgb ] ───> Converts Gray/RGBA to standardized (H, W, 3) uint8
             │
             ├─────────────────────────────────────────────────┐
             │                                                 │
             ▼ (Classical Path)                               ▼ (Deep Learning Path)
[ image_processing.bicubic_upscale ]            [ Normalization & PyTorch Tensor Conversion ]
  • OpenCV cv2.INTER_CUBIC (16-pixel kernel)      • Float32 scaling: [0.0, 1.0]
  • Spatial 4x scaling                            • Shape transpose: (1, 3, H, W)
             │                                    • Device allocation: CPU / CUDA
             ▼                                                 │
  (4H, 4W, 3) uint8 Bicubic Image                              ▼
             │                                  [ src.inference.super_resolve ]
             │                                    • Direct pass (or Tiled Feathering if large)
             │                                    • RRDBNet Generator (23 RRDB Blocks)
             │                                    • Sub-pixel nearest-neighbor 4x upsampling
             │                                                 │
             │                                                 ▼
             │                                  [ Tensor Post-Processing ]
             │                                    • Clamp [0.0, 1.0]
             │                                    • Squeeze, transpose: (4H, 4W, 3)
             │                                    • Scale to uint8 [0, 255]
             │                                                 │
             ▼                                                 ▼
[ Visual Comparison UI & Difference Map ] <────── (4H, 4W, 3) uint8 ESRGAN Image
             │
             ▼
[ High-Resolution PNG Downloads & Latency Telemetry ]
```

---

## 3. Memory-Efficient Tiled Inference

Processing high-resolution images (e.g. 1024×1024 px) on CPU-only laptops requires significant RAM. If memory limits are exceeded, raw tensor allocation can cause memory crashes.

To guarantee 100% stability on all laptops, `src/inference.py` implements **Linear Feathered Tiling**:
1. Large images are partitioned into overlapping tiles (e.g. $192\times 192$ px with a $24$-pixel overlap margin).
2. Each tile is processed independently through `RRDBNet`.
3. Overlapping boundary regions are blended using a linear cross-fade ramp:
   $$W(x) = \text{linspace}(0.01, 1.0, \text{overlap} \times 4)$$
4. The output tensor is normalized by accumulated weight masks, eliminating edge-seam artifacts.
