"""
Authentication UI Views for SR Vision Lab
Renders premium AI SaaS-style Welcome/Landing, Sign In, Registration,
Forgot Password, and User Profile & Settings views.
Styled with obsidian dark theme, cyan/electric blue gradients, and glassmorphism.
Clean HTML formatting to prevent markdown code-block indentation leaks.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st

import config
from src.auth import (
    validate_email,
    validate_password_strength,
    register_user,
    authenticate_user,
    change_user_password,
    generate_reset_token,
    reset_password_with_token
)
from src.history import get_history_stats, clear_history


def get_user_initials(full_name: str) -> str:
    """Extracts 2-letter uppercase initials from full name."""
    parts = (full_name or "U").strip().split()
    if len(parts) >= 2:
        return f"{parts[0][0]}{parts[-1][0]}".upper()
    return parts[0][:2].upper() if parts else "SR"


def render_unauthenticated_sidebar():
    """Renders the sleek guest sidebar before login."""
    with st.sidebar:
        st.markdown(
            '<div style="display:flex; align-items:center; gap:10px; padding: 4px 0 16px 0;">'
            '<div style="background: linear-gradient(135deg, #00D2FF, #0070F3); width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center; box-shadow:0 0 15px rgba(0,210,255,0.4);">'
            '<span style="font-size:20px;">🔬</span></div>'
            '<div><div style="font-size:1.15rem; font-weight:800; color:#FFFFFF; letter-spacing:-0.02em;">SR Vision Lab</div>'
            '<div style="font-size:0.75rem; color:#00D2FF; font-weight:500;">Bring Every Pixel to Life</div></div></div>',
            unsafe_allow_html=True
        )

        st.caption("AUTHENTICATION & ACCESS")

        if st.button("🌐 Welcome Tour", use_container_width=True, type="primary" if st.session_state.auth_mode == "welcome" else "secondary"):
            st.session_state.auth_mode = "welcome"
            st.rerun()

        if st.button("🔑 Sign In", use_container_width=True, type="primary" if st.session_state.auth_mode == "signin" else "secondary"):
            st.session_state.auth_mode = "signin"
            st.rerun()

        if st.button("📝 Create Account", use_container_width=True, type="primary" if st.session_state.auth_mode == "register" else "secondary"):
            st.session_state.auth_mode = "register"
            st.rerun()

        if st.button("🔐 Forgot Password?", use_container_width=True, type="primary" if st.session_state.auth_mode == "forgot" else "secondary"):
            st.session_state.auth_mode = "forgot"
            st.rerun()

        st.markdown("---")

        st.markdown(
            '<div style="background:rgba(15,23,42,0.6); border:1px solid rgba(0,210,255,0.15); border-radius:10px; padding:12px; font-size:0.78rem; color:#94A3B8; line-height:1.6;">'
            '<div style="color:#00D2FF; font-weight:700; margin-bottom:4px;">🔒 SaaS Multi-User Isolation</div>'
            '<div>• Argon2/Bcrypt Hash Security</div>'
            '<div>• Private SQLite Database</div>'
            '<div>• Isolated Image Directories</div>'
            '<div>• Parameterized SQL Queries</div>'
            '</div>',
            unsafe_allow_html=True
        )


def render_welcome_page():
    """Renders the commercial AI SaaS landing page."""
    st.markdown(
        '<div class="hero-container" style="text-align: center; padding: 48px 30px;">'
        '<div class="hero-badge" style="margin: 0 auto 12px auto;">AI-POWERED IMAGE SUPER-RESOLUTION PLATFORM</div>'
        '<div class="hero-title" style="font-size: 3rem;">Bring Every Pixel to Life.</div>'
        '<div class="hero-subtitle" style="margin: 0 auto 24px auto; font-size: 1.2rem;">'
        'Transform low-resolution images into stunning high-resolution visuals using deep learning. '
        'Experience commercial-grade 4× super-resolution powered by ESRGAN and RRDBNet.'
        '</div></div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
    with c1:
        if st.button("🔑 Sign In to Workspace", type="primary", use_container_width=True):
            st.session_state.auth_mode = "signin"
            st.rerun()
    with c2:
        if st.button("📝 Create Free Account", use_container_width=True):
            st.session_state.auth_mode = "register"
            st.rerun()
    with c3:
        if st.button("⚡ 1-Click Demo Sign In (Jessica)", use_container_width=True, help="Instantly log in using pre-seeded researcher demo credentials"):
            ok, msg, user = authenticate_user("demo@srvisionlab.ai", "Password@123")
            if ok:
                st.session_state.authenticated = True
                st.session_state.user = user
                st.session_state.current_page = "🏠 Dashboard"
                st.rerun()

    st.markdown("<hr style='margin: 32px 0;' />", unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown(
            '<div class="saas-card" style="height:100%;">'
            '<div style="font-size:2rem; margin-bottom:8px;">⚡</div>'
            '<div style="font-weight:700; font-size:1.05rem; color:#FFFFFF; margin-bottom:6px;">4× ESRGAN Engine</div>'
            '<div style="color:#94A3B8; font-size:0.84rem; line-height:1.5;">'
            '23 Residual-in-Residual Dense Blocks (RRDB) synthesizing natural high-frequency textures without blur.'
            '</div></div>',
            unsafe_allow_html=True
        )

    with f2:
        st.markdown(
            '<div class="saas-card" style="height:100%;">'
            '<div style="font-size:2rem; margin-bottom:8px;">⚖️</div>'
            '<div style="font-weight:700; font-size:1.05rem; color:#FFFFFF; margin-bottom:6px;">Interactive Comparison</div>'
            '<div style="color:#94A3B8; font-size:0.84rem; line-height:1.5;">'
            'Real-time draggable before/after slider and 4-way comparison against Lanczos, Bicubic, and Bilinear.'
            '</div></div>',
            unsafe_allow_html=True
        )

    with f3:
        st.markdown(
            '<div class="saas-card" style="height:100%;">'
            '<div style="font-size:2rem; margin-bottom:8px;">📊</div>'
            '<div style="font-weight:700; font-size:1.05rem; color:#FFFFFF; margin-bottom:6px;">Scientific Metrics</div>'
            '<div style="color:#94A3B8; font-size:0.84rem; line-height:1.5;">'
            'Ground-truth PSNR, SSIM, and runtime latency profiling with publication-grade dark visualizations.'
            '</div></div>',
            unsafe_allow_html=True
        )

    with f4:
        st.markdown(
            '<div class="saas-card" style="height:100%;">'
            '<div style="font-size:2rem; margin-bottom:8px;">🔒</div>'
            '<div style="font-weight:700; font-size:1.05rem; color:#FFFFFF; margin-bottom:6px;">Private History</div>'
            '<div style="color:#94A3B8; font-size:0.84rem; line-height:1.5;">'
            'User-isolated workspaces ensuring your uploaded images and enhancements remain strictly confidential.'
            '</div></div>',
            unsafe_allow_html=True
        )


def render_signin_page():
    """Renders the split-screen Sign In page."""
    col_left, col_right = st.columns([1.1, 1.0], gap="large")

    with col_left:
        st.markdown(
            '<div style="display:inline-flex; align-items:center; gap:8px; background:rgba(0,210,255,0.12); '
            'border:1px solid rgba(0,210,255,0.3); border-radius:9999px; padding:4px 12px; font-size:0.8rem; '
            'font-weight:700; color:#00D2FF; margin-bottom:12px;">'
            '🔐 SECURE ENTERPRISE AUTHENTICATION</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<h1 style="font-size:2.5rem; font-weight:800; color:#FFFFFF; line-height:1.15; margin-bottom:12px;">'
            'Welcome back to <span style="background:linear-gradient(135deg, #00D2FF, #3A86FF); '
            '-webkit-background-clip:text; -webkit-text-fill-color:transparent;">SR Vision Lab</span>.</h1>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p style="color:#94A3B8; font-size:1.05rem; line-height:1.6; margin-bottom:24px;">'
            'Sign in to access your personal deep learning workspace, view saved image enhancements, and run high-resolution inference.</p>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="saas-card" style="margin-bottom:14px; padding:16px;">'
            '<div style="display:flex; align-items:center; gap:12px;">'
            '<span style="font-size:1.6rem;">⚡</span>'
            '<div>'
            '<div style="font-weight:700; color:#FFFFFF; font-size:0.95rem;">Instant 4× Super-Resolution</div>'
            '<div style="color:#94A3B8; font-size:0.82rem;">16.7M parameter RRDBNet generator running with multi-threaded CPU/CUDA optimization.</div>'
            '</div></div></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="saas-card" style="margin-bottom:14px; padding:16px;">'
            '<div style="display:flex; align-items:center; gap:12px;">'
            '<span style="font-size:1.6rem;">🔒</span>'
            '<div>'
            '<div style="font-weight:700; color:#FFFFFF; font-size:0.95rem;">Private & Isolated History</div>'
            '<div style="color:#94A3B8; font-size:0.82rem;">Each user\'s enhanced images, thumbnails, and telemetry are strictly isolated by unique user ID.</div>'
            '</div></div></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="saas-card" style="margin-bottom:14px; padding:16px;">'
            '<div style="display:flex; align-items:center; gap:12px;">'
            '<span style="font-size:1.6rem;">📊</span>'
            '<div>'
            '<div style="font-weight:700; color:#FFFFFF; font-size:0.95rem;">Scientific Evaluation Suite</div>'
            '<div style="color:#94A3B8; font-size:0.82rem;">Direct access to Set5/Set14 ground-truth benchmarks and comparative heatmap analytics.</div>'
            '</div></div></div>',
            unsafe_allow_html=True
        )

    with col_right:
        with st.container(border=True):
            st.markdown(
                '<div style="font-size:1.45rem; font-weight:800; color:#FFFFFF; margin-bottom:4px;">Sign In</div>'
                '<div style="color:#94A3B8; font-size:0.85rem; margin-bottom:16px;">Enter your credentials to access your account</div>',
                unsafe_allow_html=True
            )

            with st.form("signin_form"):
                email = st.text_input("Email Address", placeholder="name@domain.com")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                remember_me = st.checkbox("Remember this session", value=True)
                submit = st.form_submit_button("Sign In", type="primary", use_container_width=True)

                if submit:
                    if not email or not password:
                        st.error("Please enter both email and password.")
                    else:
                        success, msg, user = authenticate_user(email, password)
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.user = user
                            st.session_state.current_page = "🏠 Dashboard"
                            st.toast(f"Welcome back, {user['full_name']}!", icon="👋")
                            st.rerun()
                        else:
                            st.error(msg)

            st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
            if st.button("⚡ 1-Click Demo Sign In (`demo@srvisionlab.ai`)", use_container_width=True):
                success, msg, user = authenticate_user("demo@srvisionlab.ai", "Password@123")
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user = user
                    st.session_state.current_page = "🏠 Dashboard"
                    st.rerun()

            col_f1, col_f2 = st.columns(2)
            with col_f1:
                if st.button("Forgot password?", use_container_width=True):
                    st.session_state.auth_mode = "forgot"
                    st.rerun()
            with col_f2:
                if st.button("Create an account", use_container_width=True):
                    st.session_state.auth_mode = "register"
                    st.rerun()


def render_register_page():
    """Renders the split-screen Registration page."""
    col_left, col_right = st.columns([1.1, 1.0], gap="large")

    with col_left:
        st.markdown(
            '<div style="display:inline-flex; align-items:center; gap:8px; background:rgba(0,210,255,0.12); '
            'border:1px solid rgba(0,210,255,0.3); border-radius:9999px; padding:4px 12px; font-size:0.8rem; '
            'font-weight:700; color:#00D2FF; margin-bottom:12px;">'
            '🚀 NEW USER ONBOARDING</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<h1 style="font-size:2.5rem; font-weight:800; color:#FFFFFF; line-height:1.15; margin-bottom:12px;">'
            'Create your <span style="background:linear-gradient(135deg, #00D2FF, #3A86FF); '
            '-webkit-background-clip:text; -webkit-text-fill-color:transparent;">Researcher Account</span>.</h1>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p style="color:#94A3B8; font-size:1.05rem; line-height:1.6; margin-bottom:24px;">'
            'Join the platform to run high-resolution super-resolution models, preserve your private enhancement history, and conduct scientific comparisons.</p>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="saas-card" style="padding:18px; margin-bottom:16px;">'
            '<div style="color:#00D2FF; font-weight:700; margin-bottom:8px;">🔒 Password Requirements:</div>'
            '<div style="font-size:0.84rem; color:#94A3B8; line-height:1.7;">'
            '<div>• Minimum <b>8 characters</b> in length</div>'
            '<div>• At least <b>one uppercase</b> letter (A–Z)</div>'
            '<div>• At least <b>one lowercase</b> letter (a–z)</div>'
            '<div>• At least <b>one digit or special symbol</b> (0–9, @, #, $, etc.)</div>'
            '</div></div>',
            unsafe_allow_html=True
        )

    with col_right:
        with st.container(border=True):
            st.markdown(
                '<div style="font-size:1.45rem; font-weight:800; color:#FFFFFF; margin-bottom:4px;">Create an account.</div>'
                '<div style="color:#94A3B8; font-size:0.85rem; margin-bottom:16px;">Enter your details below to get started</div>',
                unsafe_allow_html=True
            )

            with st.form("register_form"):
                full_name = st.text_input("Full Name", placeholder="e.g. Dr. Jessica Smith")
                email = st.text_input("Email Address", placeholder="name@domain.com")
                password = st.text_input("Password", type="password", placeholder="At least 8 characters")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
                terms = st.checkbox("I agree to the Terms of Service & Privacy Policy", value=True)
                submit = st.form_submit_button("Sign Up", type="primary", use_container_width=True)

                if submit:
                    if not terms:
                        st.error("Please accept the terms to create an account.")
                    else:
                        success, msg, user = register_user(full_name, email, password, confirm_password)
                        if success:
                            st.success(f"{msg} You can now sign in.")
                            st.session_state.auth_mode = "signin"
                            st.rerun()
                        else:
                            st.error(msg)

            st.markdown("<div style='margin-top:14px; text-align:center;'>", unsafe_allow_html=True)
            if st.button("Already have an account? Log In", use_container_width=True):
                st.session_state.auth_mode = "signin"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


def render_forgot_password_page():
    """Renders the transparent, development-safe password recovery workflow."""
    st.markdown(
        '<div style="text-align: center; margin-bottom: 24px;">'
        '<div style="display:inline-flex; align-items:center; gap:8px; background:rgba(0,210,255,0.12); '
        'border:1px solid rgba(0,210,255,0.3); border-radius:9999px; padding:4px 12px; font-size:0.8rem; '
        'font-weight:700; color:#00D2FF; margin-bottom:12px;">'
        '🔐 PASSWORD RECOVERY</div>'
        '<h2 style="font-size:2.2rem; font-weight:800; color:#FFFFFF; margin-bottom:8px;">Reset Your Password</h2>'
        '<p style="color:#94A3B8; font-size:0.95rem;">'
        'Enter your registered email address to receive a secure recovery verification code.'
        '</p></div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):
        st.markdown("##### Step 1: Request Verification Code")
        c1, c2 = st.columns([3, 1])
        with c1:
            reset_email = st.text_input("Registered Email", placeholder="user@domain.com", key="fp_email")
        with c2:
            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
            get_code = st.button("Generate Code", use_container_width=True)

        if get_code:
            if not reset_email:
                st.error("Please enter your registered email address.")
            else:
                ok, msg, token = generate_reset_token(reset_email)
                if ok:
                    st.session_state["active_reset_email"] = reset_email.strip().lower()
                    st.session_state["active_reset_code"] = token
                    st.success("Verification code successfully generated!")
                else:
                    st.error(msg)

        if st.session_state.get("active_reset_code"):
            st.info(f"""
            🔐 **[LOCAL / DEVELOPMENT ENVIRONMENT CODE]**  
            Your 6-character recovery verification code is: **`{st.session_state['active_reset_code']}`**  
            *(Valid for 15 minutes. In a cloud deployment, this is dispatched via transactional SMTP).*
            """)

        st.markdown("---")

        st.markdown("##### Step 2: Set New Password")
        with st.form("reset_pw_form"):
            code_input = st.text_input("6-Character Verification Code", placeholder="e.g. 4F9A2B")
            new_pw = st.text_input("New Password", type="password", placeholder="At least 8 characters")
            conf_pw = st.text_input("Confirm New Password", type="password", placeholder="Repeat new password")
            reset_submit = st.form_submit_button("Reset Password & Sign In", type="primary", use_container_width=True)

            if reset_submit:
                target_email = st.session_state.get("active_reset_email") or reset_email
                if not target_email or not code_input:
                    st.error("Please provide both email and verification code.")
                else:
                    ok, msg = reset_password_with_token(target_email, code_input, new_pw, conf_pw)
                    if ok:
                        st.success(msg)
                        st.session_state["active_reset_code"] = None
                        st.session_state["active_reset_email"] = None
                        st.session_state.auth_mode = "signin"
                        st.rerun()
                    else:
                        st.error(msg)

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    if st.button("← Back to Sign In", use_container_width=True):
        st.session_state.auth_mode = "signin"
        st.rerun()


def render_authenticated_user_sidebar(user: Dict[str, Any]):
    """Renders the sleek user profile card at the bottom of the sidebar."""
    initials = get_user_initials(user.get("full_name", "User"))
    full_name = user.get("full_name", "Researcher")
    email = user.get("email", "")

    st.markdown(
        f'<div style="background:rgba(15,23,42,0.85); border:1px solid rgba(0,210,255,0.25); border-radius:12px; padding:12px; display:flex; align-items:center; gap:12px; margin-top:20px; box-shadow:0 4px 16px rgba(0,0,0,0.4);">'
        f'<div style="background:linear-gradient(135deg, #00D2FF, #3A86FF); width:38px; height:38px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:15px; color:#FFFFFF; box-shadow:0 0 10px rgba(0,210,255,0.4); flex-shrink:0;">'
        f'{initials}</div>'
        f'<div style="overflow:hidden; flex-grow:1;">'
        f'<div style="font-weight:700; color:#FFFFFF; font-size:0.88rem; white-space:nowrap; text-overflow:ellipsis; overflow:hidden;">{full_name}</div>'
        f'<div style="font-size:0.72rem; color:#94A3B8; white-space:nowrap; text-overflow:ellipsis; overflow:hidden;">{email}</div>'
        f'</div>'
        f'<div style="width:8px; height:8px; border-radius:50%; background:#10B981; box-shadow:0 0 8px #10B981;" title="Active Session"></div>'
        f'</div>',
        unsafe_allow_html=True
    )

    col_prof, col_logout = st.columns([1, 1])
    with col_prof:
        if st.button("👤 Profile", use_container_width=True, help="Manage your account profile and security"):
            st.session_state.current_page = "👤 Profile & Settings"
            st.rerun()
    with col_logout:
        if st.button("🚪 Log Out", use_container_width=True, help="End session and return to welcome page"):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.auth_mode = "welcome"
            st.rerun()


def render_profile_settings_page(user: Dict[str, Any]):
    """Renders the User Profile & Security Settings page."""
    initials = get_user_initials(user.get("full_name", "User"))
    full_name = user.get("full_name", "Researcher")
    email = user.get("email", "")
    created_at = user.get("created_at", "2026-11-04")
    user_id = user["id"]

    stats = get_history_stats(user_id=user_id)

    st.markdown("### 👤 User Profile & Account Settings")
    st.caption("Manage your research profile credentials, security password, and personal enhancement data.")

    st.markdown(
        f'<div style="background:linear-gradient(135deg, rgba(14,21,37,0.95), rgba(8,12,20,0.98)); border:1px solid rgba(0,210,255,0.25); border-radius:16px; padding:28px; display:flex; align-items:center; gap:24px; margin-bottom:24px;">'
        f'<div style="background:linear-gradient(135deg, #00D2FF, #3A86FF); width:72px; height:72px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:28px; color:#FFFFFF; box-shadow:0 0 20px rgba(0,210,255,0.4); flex-shrink:0;">'
        f'{initials}</div>'
        f'<div><div style="font-size:1.6rem; font-weight:800; color:#FFFFFF;">{full_name}</div>'
        f'<div style="color:#00D2FF; font-size:0.9rem; font-weight:600;">{email}</div>'
        f'<div style="color:#94A3B8; font-size:0.8rem; margin-top:4px;">'
        f'Role: <code>{user.get("role", "researcher").upper()}</code> • Member Since: <code>{created_at}</code>'
        f'</div></div></div>',
        unsafe_allow_html=True
    )

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(
            f'<div class="saas-card">'
            f'<div class="saas-card-label">Your Enhanced Images</div>'
            f'<div class="saas-card-val">{stats["total_processed"]}</div>'
            f'<div class="saas-card-sub">Strictly isolated to your account</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with p2:
        avg_display = f"{stats['avg_inference_time']:.2f} s" if stats['avg_inference_time'] else "N/A"
        st.markdown(
            f'<div class="saas-card">'
            f'<div class="saas-card-label">Your Mean Latency</div>'
            f'<div class="saas-card-val">{avg_display}</div>'
            f'<div class="saas-card-sub">Average processing duration</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with p3:
        mp = round(stats['total_pixels_synthesized'] / 1_000_000.0, 2)
        st.markdown(
            f'<div class="saas-card">'
            f'<div class="saas-card-label">Synthesized Pixels</div>'
            f'<div class="saas-card-val">{mp} MP</div>'
            f'<div class="saas-card-sub">Cumulative resolution gained</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    col_sec, col_data = st.columns([1.1, 1.0], gap="large")

    with col_sec:
        st.markdown("#### 🔒 Security: Change Password")
        st.caption("Update your password to maintain account integrity.")

        with st.form("change_password_form"):
            current_pw = st.text_input("Current Password", type="password")
            new_pw = st.text_input("New Password", type="password")
            conf_pw = st.text_input("Confirm New Password", type="password")
            submit_chg = st.form_submit_button("Update Password", type="primary", use_container_width=True)

            if submit_chg:
                if not current_pw or not new_pw or not conf_pw:
                    st.error("Please fill in all password fields.")
                else:
                    ok, msg = change_user_password(user_id, current_pw, new_pw, conf_pw)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

    with col_data:
        st.markdown("#### 🛡️ Personal Data Management")
        st.caption("Manage your isolated enhancement files and database records.")

        st.markdown(
            f'<div style="background:rgba(15,23,42,0.6); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:16px; margin-bottom:16px;">'
            f'<div style="font-weight:700; color:#FFFFFF; margin-bottom:4px;">Data Isolation Status: <span style="color:#10B981;">Active</span></div>'
            f'<div style="font-size:0.82rem; color:#94A3B8; line-height:1.6;">'
            f'All enhancements made under <code>{email}</code> are tagged with your unique database ID (<code>#{user_id}</code>). '
            f'Other platform users cannot query, view, or download your image files.'
            f'</div></div>',
            unsafe_allow_html=True
        )

        if st.button("🗑️ Purge My Enhancement History", type="secondary", use_container_width=True):
            clear_history(user_id=user_id)
            st.toast("Your personal enhancement history has been cleared.", icon="🗑️")
            st.rerun()

        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        if st.button("🚪 Log Out of SR Vision Lab", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.auth_mode = "welcome"
            st.rerun()
