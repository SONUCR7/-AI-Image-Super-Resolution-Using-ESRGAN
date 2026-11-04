"""
Processing History Module
Manages local JSON persistence, thumbnail creation, and historical statistics
for all user image enhancement sessions in SR Vision Lab.
"""

import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image

import config

HISTORY_DIR = config.OUTPUTS_DIR / "history"
MANIFEST_FILE = HISTORY_DIR / "manifest.json"
THUMBS_DIR = HISTORY_DIR / "thumbnails"
ENHANCED_DIR = HISTORY_DIR / "enhanced"
MAX_ENTRIES = 30  # Prevent unbounded disk consumption


def _ensure_dirs():
    """Ensures history directories exist."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)
    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)


def load_history() -> List[Dict[str, Any]]:
    """Loads all historical enhancement records, sorted newest first."""
    _ensure_dirs()
    if not MANIFEST_FILE.exists():
        return []
    try:
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        return []
    return []


def _save_manifest(entries: List[Dict[str, Any]]) -> None:
    """Writes historical records back to disk."""
    _ensure_dirs()
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def add_history_entry(
    original_img: np.ndarray,
    enhanced_img: np.ndarray,
    filename: str,
    scale_factor: int,
    model_name: str,
    elapsed_sec: float,
    device: str
) -> Dict[str, Any]:
    """
    Saves an enhanced image session into persistent local storage.
    Creates lightweight thumbnails for fast dashboard display.
    """
    _ensure_dirs()
    entry_id = str(uuid.uuid4())[:8]
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Generate compressed thumbnail for original
    orig_pil = Image.fromarray(original_img)
    orig_thumb = orig_pil.copy()
    orig_thumb.thumbnail((180, 180), Image.Resampling.LANCZOS)
    orig_thumb_path = THUMBS_DIR / f"{entry_id}_orig.webp"
    orig_thumb.save(orig_thumb_path, format="WEBP", quality=80)

    # Generate compressed thumbnail for enhanced
    enh_pil = Image.fromarray(enhanced_img)
    enh_thumb = enh_pil.copy()
    enh_thumb.thumbnail((180, 180), Image.Resampling.LANCZOS)
    enh_thumb_path = THUMBS_DIR / f"{entry_id}_enh.webp"
    enh_thumb.save(enh_thumb_path, format="WEBP", quality=80)

    # Save full enhanced image
    enh_full_path = ENHANCED_DIR / f"{entry_id}_enhanced.png"
    enh_pil.save(enh_full_path, format="PNG", optimize=True)

    file_size_kb = round(enh_full_path.stat().st_size / 1024.0, 1) if enh_full_path.exists() else 0.0

    entry = {
        "id": entry_id,
        "timestamp": timestamp,
        "filename": filename,
        "orig_width": int(original_img.shape[1]),
        "orig_height": int(original_img.shape[0]),
        "enhanced_width": int(enhanced_img.shape[1]),
        "enhanced_height": int(enhanced_img.shape[0]),
        "scale_factor": f"{scale_factor}x",
        "model_name": model_name,
        "elapsed_sec": round(float(elapsed_sec), 3),
        "device": device,
        "orig_thumb": str(orig_thumb_path.relative_to(config.BASE_DIR)),
        "enh_thumb": str(enh_thumb_path.relative_to(config.BASE_DIR)),
        "enhanced_file": str(enh_full_path.relative_to(config.BASE_DIR)),
        "filesize_kb": file_size_kb
    }

    entries = load_history()
    entries.insert(0, entry)

    # Maintain max limit and clean orphaned files
    if len(entries) > MAX_ENTRIES:
        trimmed = entries[MAX_ENTRIES:]
        entries = entries[:MAX_ENTRIES]
        for t in trimmed:
            _cleanup_entry_files(t)

    _save_manifest(entries)
    return entry


def _cleanup_entry_files(entry: Dict[str, Any]) -> None:
    """Removes on-disk files associated with an entry."""
    for key in ["orig_thumb", "enh_thumb", "enhanced_file"]:
        rel = entry.get(key)
        if rel:
            p = config.BASE_DIR / rel
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass


def delete_history_entry(entry_id: str) -> bool:
    """Deletes a single history record and its associated files."""
    entries = load_history()
    target = None
    remaining = []
    for e in entries:
        if e.get("id") == entry_id:
            target = e
        else:
            remaining.append(e)

    if target:
        _cleanup_entry_files(target)
        _save_manifest(remaining)
        return True
    return False


def clear_history() -> bool:
    """Purges all history records and files."""
    entries = load_history()
    for e in entries:
        _cleanup_entry_files(e)
    _save_manifest([])
    return True


def get_history_stats() -> Dict[str, Any]:
    """Computes genuine statistical metrics from actual application usage."""
    entries = load_history()
    total_count = len(entries)
    if total_count == 0:
        return {
            "total_processed": 0,
            "avg_inference_time": None,
            "total_pixels_synthesized": 0,
            "latest_entry": None,
            "recent_entries": []
        }

    times = [e["elapsed_sec"] for e in entries if "elapsed_sec" in e]
    avg_time = round(sum(times) / len(times), 2) if times else 0.0

    total_pixels = sum(
        (e.get("enhanced_width", 0) * e.get("enhanced_height", 0))
        for e in entries
    )

    return {
        "total_processed": total_count,
        "avg_inference_time": avg_time,
        "total_pixels_synthesized": total_pixels,
        "latest_entry": entries[0] if entries else None,
        "recent_entries": entries[:6]
    }
