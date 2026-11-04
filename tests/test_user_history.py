"""
Unit Tests for User-Isolated History and Data Security
Verifies:
- User-specific history persistence
- Data isolation (User A cannot see User B's images)
- Ownership enforcement (User A cannot delete User B's record)
- User-specific statistical metrics
"""

import numpy as np
import pytest
from src.auth import register_user
from src.history import (
    add_history_entry,
    load_history,
    delete_history_entry,
    get_history_stats
)


def test_user_history_isolation():
    import uuid
    uid_a = str(uuid.uuid4())[:6]
    uid_b = str(uuid.uuid4())[:6]

    # Create two test users
    _, _, user_a = register_user("User Alpha", f"alpha_{uid_a}@test.com", "Password@123", "Password@123")
    _, _, user_b = register_user("User Beta", f"beta_{uid_b}@test.com", "Password@123", "Password@123")

    assert user_a is not None
    assert user_b is not None

    id_a = user_a["id"]
    id_b = user_b["id"]

    # Generate dummy images
    orig = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    enh = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    # User A enhances an image
    entry_a = add_history_entry(
        original_img=orig,
        enhanced_img=enh,
        filename="alpha_sample.png",
        scale_factor=4,
        model_name="ESRGAN_x4",
        elapsed_sec=0.75,
        device="CPU",
        user_id=id_a
    )

    # User B enhances a different image
    entry_b = add_history_entry(
        original_img=orig,
        enhanced_img=enh,
        filename="beta_sample.png",
        scale_factor=4,
        model_name="ESRGAN_x4",
        elapsed_sec=1.10,
        device="CPU",
        user_id=id_b
    )

    # 1. User A only sees their own history
    history_a = load_history(user_id=id_a)
    ids_in_a = [e["id"] for e in history_a]
    assert entry_a["id"] in ids_in_a
    assert entry_b["id"] not in ids_in_a

    # 2. User B only sees their own history
    history_b = load_history(user_id=id_b)
    ids_in_b = [e["id"] for e in history_b]
    assert entry_b["id"] in ids_in_b
    assert entry_a["id"] not in ids_in_b

    # 3. User A attempts to delete User B's record -> MUST FAIL (Ownership enforcement)
    unauthorized_delete = delete_history_entry(entry_b["id"], user_id=id_a)
    assert unauthorized_delete is False

    # Verify User B's record is still intact
    history_b_after = load_history(user_id=id_b)
    assert any(e["id"] == entry_b["id"] for e in history_b_after)

    # 4. User A deletes their own record -> MUST SUCCEED
    authorized_delete = delete_history_entry(entry_a["id"], user_id=id_a)
    assert authorized_delete is True

    # 5. User-specific statistics isolation
    stats_b = get_history_stats(user_id=id_b)
    assert stats_b["total_processed"] >= 1
    assert stats_b["latest_entry"]["filename"] == "beta_sample.png"
