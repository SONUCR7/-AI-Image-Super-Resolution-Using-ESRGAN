"""
Unit Tests for History & Statistical Persistence Module
"""

import numpy as np
import pytest
from src.history import (
    load_history,
    add_history_entry,
    delete_history_entry,
    clear_history,
    get_history_stats
)


def test_history_lifecycle():
    # Setup dummy test images
    orig = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    enh = np.random.randint(0, 255, (192, 192, 3), dtype=np.uint8)

    # 1. Add entry
    entry = add_history_entry(
        original_img=orig,
        enhanced_img=enh,
        filename="test_unit.png",
        scale_factor=4,
        model_name="ESRGAN (Test)",
        elapsed_sec=1.23,
        device="CPU"
    )
    assert entry is not None
    assert "id" in entry
    assert entry["scale_factor"] == "4x"
    assert entry["orig_width"] == 48
    assert entry["enhanced_width"] == 192

    # 2. Verify load
    records = load_history()
    assert any(r["id"] == entry["id"] for r in records)

    # 3. Verify statistics
    stats = get_history_stats()
    assert stats["total_processed"] >= 1
    assert stats["avg_inference_time"] is not None

    # 4. Clean up test record
    deleted = delete_history_entry(entry["id"])
    assert deleted is True
