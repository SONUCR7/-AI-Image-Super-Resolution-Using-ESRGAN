"""
Processing History Module for SR Vision Lab
Manages user-isolated persistent database storage, thumbnail generation,
and historical statistics for image enhancement sessions.
Supports multi-user data isolation via SQLite and provides backward-compatible fallbacks.
"""

import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image

import config
from src.database import execute_query, fetch_one, fetch_all, init_db

init_db()

HISTORY_BASE_DIR = config.OUTPUTS_DIR / "history"
MANIFEST_FILE = HISTORY_BASE_DIR / "manifest.json"
MAX_ENTRIES_PER_USER = 50


def _get_user_dirs(user_id: Optional[int] = None):
    """Returns directory paths isolated by user ID."""
    folder_name = f"user_{user_id}" if user_id is not None else "common"
    user_dir = HISTORY_BASE_DIR / folder_name
    thumbs_dir = user_dir / "thumbnails"
    enhanced_dir = user_dir / "enhanced"

    user_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    return user_dir, thumbs_dir, enhanced_dir


def _row_to_entry(row: Dict[str, Any]) -> Dict[str, Any]:
    """Converts a database row into standard history entry format."""
    return {
        "id": row["id"],
        "user_id": row.get("user_id"),
        "timestamp": row["created_at"],
        "filename": row["filename"],
        "orig_width": row["orig_width"],
        "orig_height": row["orig_height"],
        "enhanced_width": row["enhanced_width"],
        "enhanced_height": row["enhanced_height"],
        "scale_factor": row["scale_factor"],
        "model_name": row["model_name"],
        "elapsed_sec": row["elapsed_sec"],
        "device": row["device"],
        "orig_thumb": row["orig_thumb_path"],
        "enh_thumb": row["enh_thumb_path"],
        "enhanced_file": row["enhanced_file_path"],
        "filesize_kb": row["file_size_kb"]
    }


def load_history(user_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Loads historical enhancement records, sorted newest first.
    If user_id is provided, strictly isolates records belonging to that user.
    """
    if user_id is not None:
        rows = fetch_all(
            "SELECT * FROM enhancement_history WHERE user_id = ? ORDER BY created_at DESC;",
            (user_id,)
        )
        return [_row_to_entry(r) for r in rows]

    # Legacy or global fallback
    try:
        rows = fetch_all("SELECT * FROM enhancement_history ORDER BY created_at DESC;")
        if rows:
            return [_row_to_entry(r) for r in rows]
    except Exception:
        pass

    if MANIFEST_FILE.exists():
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            return []
    return []


def add_history_entry(
    original_img: np.ndarray,
    enhanced_img: np.ndarray,
    filename: str,
    scale_factor: int,
    model_name: str,
    elapsed_sec: float,
    device: str,
    user_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Saves an enhanced image session into persistent user-isolated storage.
    Creates lightweight thumbnails and inserts record into SQLite database.
    """
    _, thumbs_dir, enhanced_dir = _get_user_dirs(user_id)
    entry_id = str(uuid.uuid4())[:8]
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Generate compressed thumbnail for original
    orig_pil = Image.fromarray(original_img)
    orig_thumb = orig_pil.copy()
    orig_thumb.thumbnail((180, 180), Image.Resampling.LANCZOS)
    orig_thumb_path = thumbs_dir / f"{entry_id}_orig.webp"
    orig_thumb.save(orig_thumb_path, format="WEBP", quality=80)

    # Generate compressed thumbnail for enhanced
    enh_pil = Image.fromarray(enhanced_img)
    enh_thumb = enh_pil.copy()
    enh_thumb.thumbnail((180, 180), Image.Resampling.LANCZOS)
    enh_thumb_path = thumbs_dir / f"{entry_id}_enh.webp"
    enh_thumb.save(enh_thumb_path, format="WEBP", quality=80)

    # Save full enhanced image
    enh_full_path = enhanced_dir / f"{entry_id}_enhanced.png"
    enh_pil.save(enh_full_path, format="PNG", optimize=True)

    file_size_kb = round(enh_full_path.stat().st_size / 1024.0, 1) if enh_full_path.exists() else 0.0

    orig_thumb_rel = str(orig_thumb_path.relative_to(config.BASE_DIR)).replace("\\", "/")
    enh_thumb_rel = str(enh_thumb_path.relative_to(config.BASE_DIR)).replace("\\", "/")
    enh_full_rel = str(enh_full_path.relative_to(config.BASE_DIR)).replace("\\", "/")

    entry = {
        "id": entry_id,
        "user_id": user_id,
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
        "orig_thumb": orig_thumb_rel,
        "enh_thumb": enh_thumb_rel,
        "enhanced_file": enh_full_rel,
        "filesize_kb": file_size_kb
    }

    # Store in database if user_id is provided or default user exists
    target_user_id = user_id if user_id is not None else 1
    try:
        execute_query("""
            INSERT INTO enhancement_history (
                id, user_id, filename, orig_width, orig_height,
                enhanced_width, enhanced_height, scale_factor, model_name,
                elapsed_sec, device, orig_thumb_path, enh_thumb_path,
                enhanced_file_path, file_size_kb, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            entry_id, target_user_id, filename, entry["orig_width"], entry["orig_height"],
            entry["enhanced_width"], entry["enhanced_height"], entry["scale_factor"],
            model_name, entry["elapsed_sec"], device, orig_thumb_rel,
            enh_thumb_rel, enh_full_rel, file_size_kb, timestamp
        ))
    except Exception:
        pass

    # Maintain legacy manifest for backward compatibility
    try:
        HISTORY_BASE_DIR.mkdir(parents=True, exist_ok=True)
        manifest_entries = []
        if MANIFEST_FILE.exists():
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                manifest_entries = json.load(f)
        manifest_entries.insert(0, entry)
        if len(manifest_entries) > MAX_ENTRIES_PER_USER:
            manifest_entries = manifest_entries[:MAX_ENTRIES_PER_USER]
        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest_entries, f, indent=2)
    except Exception:
        pass

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


def delete_history_entry(entry_id: str, user_id: Optional[int] = None) -> bool:
    """
    Deletes a single history record and its associated files.
    Enforces strict user ownership: users can only delete their own images.
    """
    # Check in DB with user_id constraint
    if user_id is not None:
        row = fetch_one(
            "SELECT * FROM enhancement_history WHERE id = ? AND user_id = ?;",
            (entry_id, user_id)
        )
        if not row:
            return False  # Record does not belong to this user or does not exist
        entry = _row_to_entry(row)
        _cleanup_entry_files(entry)
        execute_query("DELETE FROM enhancement_history WHERE id = ? AND user_id = ?;", (entry_id, user_id))
        return True

    # Global or legacy deletion
    row = fetch_one("SELECT * FROM enhancement_history WHERE id = ?;", (entry_id,))
    if row:
        entry = _row_to_entry(row)
        _cleanup_entry_files(entry)
        execute_query("DELETE FROM enhancement_history WHERE id = ?;", (entry_id,))
        return True

    # Fallback to manifest file
    if MANIFEST_FILE.exists():
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                entries = json.load(f)
            remaining = [e for e in entries if e.get("id") != entry_id]
            target = next((e for e in entries if e.get("id") == entry_id), None)
            if target:
                _cleanup_entry_files(target)
                with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
                    json.dump(remaining, f, indent=2)
                return True
        except Exception:
            pass

    return False


def clear_history(user_id: Optional[int] = None) -> bool:
    """Purges all history records and files for the specified user."""
    entries = load_history(user_id=user_id)
    for e in entries:
        _cleanup_entry_files(e)

    if user_id is not None:
        execute_query("DELETE FROM enhancement_history WHERE user_id = ?;", (user_id,))
    else:
        execute_query("DELETE FROM enhancement_history;")
        if MANIFEST_FILE.exists():
            with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    return True


def get_history_stats(user_id: Optional[int] = None) -> Dict[str, Any]:
    """Computes statistical metrics from actual application usage for a specific user."""
    entries = load_history(user_id=user_id)
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
