"""
Authentication and User Security Module for SR Vision Lab
Handles password hashing (bcrypt), registration validation,
credential authentication, session verification, and password recovery.
"""

import re
import secrets
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
import bcrypt

from src.database import execute_query, fetch_one, init_db

# Ensure database is initialized upon module load
init_db()

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


def validate_email(email: str) -> Tuple[bool, str]:
    """Validates standard email formatting."""
    if not email or not isinstance(email, str):
        return False, "Email address is required."
    clean_email = email.strip()
    if not EMAIL_REGEX.match(clean_email):
        return False, "Please enter a valid email address (e.g., researcher@domain.com)."
    return True, ""


def validate_password_strength(password: str) -> Tuple[bool, str]:
    """
    Enforces SaaS password standards:
    - Minimum 8 characters
    - At least one lowercase letter
    - At least one uppercase letter
    - At least one numeric digit or symbol
    """
    if not password or not isinstance(password, str):
        return False, "Password is required."
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"[\d\W_]", password):
        return False, "Password must contain at least one number or special symbol."
    return True, ""


def hash_password(plain_password: str) -> str:
    """Hashes a plain text password using bcrypt with auto-generated salt."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain text password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def register_user(
    full_name: str,
    email: str,
    password: str,
    confirm_password: str
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Registers a new user into the database.
    Performs field validation, checks for email collisions, and hashes passwords securely.
    """
    name = (full_name or "").strip()
    clean_email = (email or "").strip().lower()

    if len(name) < 2:
        return False, "Please enter your full name (at least 2 characters).", None

    is_valid_email, email_err = validate_email(clean_email)
    if not is_valid_email:
        return False, email_err, None

    is_strong_pw, pw_err = validate_password_strength(password)
    if not is_strong_pw:
        return False, pw_err, None

    if password != confirm_password:
        return False, "Passwords do not match. Please verify both password fields.", None

    # Check for existing email registration
    existing = fetch_one("SELECT id FROM users WHERE email = ? COLLATE NOCASE;", (clean_email,))
    if existing:
        return False, "An account with this email address already exists. Please sign in instead.", None

    pw_hash = hash_password(password)

    try:
        new_id = execute_query(
            "INSERT INTO users (full_name, email, password_hash, role) VALUES (?, ?, ?, 'researcher');",
            (name, clean_email, pw_hash)
        )
        user = fetch_one("SELECT id, full_name, email, role, created_at FROM users WHERE id = ?;", (new_id,))
        return True, "Account successfully registered! You may now sign in.", user
    except Exception as e:
        return False, f"Registration failed due to a database error: {str(e)}", None


def authenticate_user(
    email: str,
    password: str
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Validates user credentials against stored bcrypt hash.
    Returns authenticated user profile if credentials match.
    """
    clean_email = (email or "").strip().lower()
    if not clean_email or not password:
        return False, "Please provide both email address and password.", None

    user = fetch_one("SELECT * FROM users WHERE email = ? COLLATE NOCASE;", (clean_email,))
    if not user:
        return False, "Invalid email address or password.", None

    if not verify_password(password, user["password_hash"]):
        return False, "Invalid email address or password.", None

    # Update last login timestamp
    execute_query("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?;", (user["id"],))

    safe_user = {
        "id": user["id"],
        "full_name": user["full_name"],
        "email": user["email"],
        "role": user.get("role", "user"),
        "created_at": user["created_at"],
        "last_login": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return True, "Sign in successful.", safe_user


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves safe user information by unique ID."""
    return fetch_one("SELECT id, full_name, email, role, created_at, last_login FROM users WHERE id = ?;", (user_id,))


def change_user_password(
    user_id: int,
    current_password: str,
    new_password: str,
    confirm_password: str
) -> Tuple[bool, str]:
    """Allows an authenticated user to change their password securely."""
    user = fetch_one("SELECT id, password_hash FROM users WHERE id = ?;", (user_id,))
    if not user:
        return False, "User account not found."

    if not verify_password(current_password, user["password_hash"]):
        return False, "Current password verification failed. Please try again."

    is_strong, pw_err = validate_password_strength(new_password)
    if not is_strong:
        return False, pw_err

    if new_password != confirm_password:
        return False, "New passwords do not match."

    if current_password == new_password:
        return False, "New password cannot be the same as your current password."

    new_hash = hash_password(new_password)
    execute_query("UPDATE users SET password_hash = ? WHERE id = ?;", (new_hash, user_id))
    return True, "Password updated successfully."


def generate_reset_token(email: str) -> Tuple[bool, str, Optional[str]]:
    """
    Generates a secure, 6-character recovery token for password reset.
    Valid for 15 minutes.
    """
    clean_email = (email or "").strip().lower()
    user = fetch_one("SELECT id FROM users WHERE email = ? COLLATE NOCASE;", (clean_email,))
    if not user:
        return False, "No registered account found with that email address.", None

    token = secrets.token_hex(3).upper()  # e.g., '4F9A2B'
    expiry = (datetime.now() + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")

    execute_query(
        "UPDATE users SET reset_token = ?, reset_token_expiry = ? WHERE id = ?;",
        (token, expiry, user["id"])
    )
    return True, "Recovery code generated.", token


def reset_password_with_token(
    email: str,
    token: str,
    new_password: str,
    confirm_password: str
) -> Tuple[bool, str]:
    """Validates token and updates user password."""
    clean_email = (email or "").strip().lower()
    clean_token = (token or "").strip().upper()

    user = fetch_one("SELECT id, reset_token, reset_token_expiry FROM users WHERE email = ? COLLATE NOCASE;", (clean_email,))
    if not user or not user["reset_token"]:
        return False, "Invalid or expired password reset request."

    if user["reset_token"] != clean_token:
        return False, "Invalid verification code. Please check and try again."

    expiry_str = user["reset_token_expiry"]
    try:
        expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d %H:%M:%S")
        if datetime.now() > expiry_dt:
            return False, "This recovery code has expired (15-minute validity). Please request a new one."
    except Exception:
        pass

    is_strong, pw_err = validate_password_strength(new_password)
    if not is_strong:
        return False, pw_err

    if new_password != confirm_password:
        return False, "New passwords do not match."

    new_hash = hash_password(new_password)
    execute_query(
        "UPDATE users SET password_hash = ?, reset_token = NULL, reset_token_expiry = NULL WHERE id = ?;",
        (new_hash, user["id"])
    )
    return True, "Password has been successfully reset! You can now sign in."
