"""
Unit Tests for Authentication and Security Module
Tests:
- User registration
- Duplicate email prevention
- Password hashing and verification
- Weak password rejection
- Email format validation
- Successful login
- Incorrect password rejection
- Password change
- Password recovery workflow (reset token generation and reset)
"""

import pytest
from src.auth import (
    validate_email,
    validate_password_strength,
    hash_password,
    verify_password,
    register_user,
    authenticate_user,
    change_user_password,
    generate_reset_token,
    reset_password_with_token
)


def test_email_validation():
    valid, _ = validate_email("user@example.com")
    assert valid is True

    valid, _ = validate_email("user.name+tag@sub.domain.co")
    assert valid is True

    invalid, msg = validate_email("not-an-email")
    assert invalid is False
    assert len(msg) > 0

    invalid, _ = validate_email("")
    assert invalid is False


def test_password_strength():
    # Valid strong password
    valid, _ = validate_password_strength("SecurePass123!")
    assert valid is True

    # Too short (< 8)
    invalid, msg = validate_password_strength("Pass1!")
    assert invalid is False
    assert "8 characters" in msg

    # No uppercase
    invalid, msg = validate_password_strength("lowercase123!")
    assert invalid is False
    assert "uppercase" in msg

    # No lowercase
    invalid, msg = validate_password_strength("UPPERCASE123!")
    assert invalid is False
    assert "lowercase" in msg

    # No number/symbol
    invalid, msg = validate_password_strength("OnlyLettersPass")
    assert invalid is False
    assert "number or special symbol" in msg


def test_password_hashing():
    plain = "SuperSecret_2026"
    hashed = hash_password(plain)
    assert hashed != plain
    assert hashed.startswith("$2b$")
    assert verify_password(plain, hashed) is True
    assert verify_password("WrongPass", hashed) is False


def test_user_registration_lifecycle():
    import uuid
    uid = str(uuid.uuid4())[:6]
    test_email = f"tester_{uid}@srvisionlab.ai"
    test_pw = "ValidPass123!"

    # 1. Successful registration
    success, msg, user = register_user(
        full_name="Alex Vance",
        email=test_email,
        password=test_pw,
        confirm_password=test_pw
    )
    assert success is True
    assert user is not None
    assert user["email"] == test_email

    # 2. Duplicate registration rejection
    dup_success, dup_msg, _ = register_user(
        full_name="Duplicate User",
        email=test_email,
        password=test_pw,
        confirm_password=test_pw
    )
    assert dup_success is False
    assert "already exists" in dup_msg

    # 3. Successful authentication
    auth_ok, auth_msg, auth_user = authenticate_user(test_email, test_pw)
    assert auth_ok is True
    assert auth_user["id"] == user["id"]

    # 4. Incorrect password rejection
    bad_ok, bad_msg, _ = authenticate_user(test_email, "WrongPassword999!")
    assert bad_ok is False
    assert "Invalid email address or password" in bad_msg

    # 5. Non-existent email rejection
    none_ok, _, _ = authenticate_user(f"nonexistent_{uid}@test.com", test_pw)
    assert none_ok is False

    # 6. Password change
    new_pw = "BrandNewPass999@"
    chg_ok, chg_msg = change_user_password(user["id"], test_pw, new_pw, new_pw)
    assert chg_ok is True
    # Verify login with new password
    new_auth, _, _ = authenticate_user(test_email, new_pw)
    assert new_auth is True

    # 7. Password reset workflow
    tok_ok, _, token = generate_reset_token(test_email)
    assert tok_ok is True
    assert token is not None
    assert len(token) == 6

    # Reset with valid token
    reset_pw = "ResetPassword456#"
    res_ok, _ = reset_password_with_token(test_email, token, reset_pw, reset_pw)
    assert res_ok is True

    # Verify login with reset password
    final_auth, _, _ = authenticate_user(test_email, reset_pw)
    assert final_auth is True
