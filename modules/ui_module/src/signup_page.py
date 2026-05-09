"""Signup page for AI Personal Tutor — collects student profile before entering lesson UI."""

from __future__ import annotations

import re

import gradio as gr

INDIAN_LANGUAGES = [
    "English",
    "Tamil",
    "Hindi",
    "Telugu",
    "Kannada",
    "Malayalam",
    "Marathi",
    "Bengali",
    "Gujarati",
    "Odia",
    "Punjabi",
    "Assamese",
    "Urdu",
    "Sanskrit",
]

BOARDS = [
    "CBSE",
    "ICSE / ISC",
    "Tamil Nadu State Board",
    "Maharashtra State Board",
    "Karnataka State Board",
    "Kerala State Board",
    "Andhra Pradesh State Board",
    "Telangana State Board",
    "West Bengal State Board",
    "Rajasthan State Board",
    "Uttar Pradesh State Board",
    "Gujarat State Board",
    "Bihar State Board",
    "Other State Board",
]

# Classes 6 – 12 shown as ordinal English labels in the dropdown.
STANDARDS = ["6th", "7th", "8th", "9th", "10th", "11th", "12th"]

SIGNUP_CSS = """
/* ── Signup page base ──────────────────────────────────── */
#signup-wrap {
    max-width: 520px !important;
    margin: 48px auto 0 !important;
}
#signup-logo {
    text-align: center;
    margin-bottom: 6px;
}
#signup-logo img {
    width: 72px;
    height: 72px;
}
#signup-card {
    background: rgba(255,255,255,0.96) !important;
    border-radius: 20px !important;
    padding: 36px 40px 32px !important;
    box-shadow: 0 8px 40px rgba(35,86,216,0.13) !important;
}
#signup-title {
    text-align: center;
    font-size: 1.6rem;
    font-weight: 700;
    color: #1a3a6b;
    margin-bottom: 4px;
}
#signup-subtitle {
    text-align: center;
    font-size: 0.95rem;
    color: #5a6e8c;
    margin-bottom: 24px;
}
#signup-submit button {
    background: linear-gradient(135deg, #2356d8, #1a8cff) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    min-height: 48px !important;
    width: 100% !important;
    margin-top: 8px !important;
    transition: opacity 0.2s ease !important;
}
#signup-submit button:hover { opacity: 0.88 !important; }
#signup-err {
    color: #d62839;
    font-size: 0.9rem;
    min-height: 20px;
    text-align: center;
}
/* Hide Gradio footer labels inside card */
#signup-card label.block { font-weight: 600; color: #1a3a6b; }
"""

SIGNUP_BG_CSS = """
html, body, #root { min-height: 100%; }
body {
    background-image: url('https://images.unsplash.com/photo-1503676260728-1c00da094a0b?auto=format&fit=crop&w=2000&q=80');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-color: #dbe7f4;
}
.gradio-container::before {
    content: "";
    position: fixed;
    inset: 0;
    backdrop-filter: blur(6px) saturate(1.05);
    background: linear-gradient(rgba(226,239,254,0.42), rgba(241,248,255,0.36));
    z-index: 0;
    pointer-events: none;
}
.gradio-container { position: relative; z-index: 1 !important; }
"""


def _validate(
    name: str,
    email: str,
    phone: str,
    school: str,
    standard: str,
    board: str,
    medium: str,
) -> str | None:
    """Return error string or None if valid."""
    name = (name or "").strip()
    email = (email or "").strip()
    phone = (phone or "").strip()
    school = (school or "").strip()

    if not name:
        return "Please enter your full name."
    if not email or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return "Please enter a valid email address."
    if phone and not re.fullmatch(r"[+\d\s\-()]{7,15}", phone):
        return "Phone number looks invalid. Leave blank if not applicable."
    if not school:
        return "Please enter your school / institution name."
    if not standard or standard not in STANDARDS:
        return "Please select your standard / class."
    if not board or board == "Select your board":
        return "Please select your board."
    if not medium or medium == "Select medium of instruction":
        return "Please select medium of instruction."
    return None


def handle_signup(
    name: str,
    email: str,
    phone: str,
    school: str,
    standard: str,
    board: str,
    medium: str,
) -> tuple[str, gr.update, gr.update, gr.update]:
    """Validate fields; on success hide signup, show tutor with personalised welcome.

    Returns: (error_html, signup_col_update, tutor_col_update, welcome_banner_update)
    """
    err = _validate(name, email, phone, school, standard, board, medium)
    if err:
        return err, gr.update(visible=True), gr.update(visible=False), gr.update()

    first = (name or "").strip().split()[0]
    welcome_md = f"# 🎓 Welcome, {first}! Let's learn today."
    return "", gr.update(visible=False), gr.update(visible=True), gr.update(value=welcome_md)


def build_signup_page(tutor_column: gr.Column, welcome_banner: gr.Markdown | None = None) -> gr.Column:
    """
    Build the signup form column.  Pass the already-created tutor_column so the
    submit handler can toggle visibility between the two.  Optionally pass
    welcome_banner to update it with the student's first name on submit.

    Returns the signup column (starts visible).
    """
    with gr.Column(visible=True, elem_id="signup-wrap") as signup_col:
        gr.HTML(
            "<div id='signup-logo'>"
            "<span style='font-size:64px;'>🎓</span>"
            "</div>"
        )
        with gr.Group(elem_id="signup-card"):
            gr.HTML("<div id='signup-title'>Welcome to AI Tutor</div>")
            gr.HTML(
                "<div id='signup-subtitle'>"
                "Tell us a little about yourself to get started"
                "</div>"
            )

            name_inp = gr.Textbox(
                label="Full Name",
                placeholder="e.g. Arun Kumar",
                max_lines=1,
            )
            email_inp = gr.Textbox(
                label="Email Address",
                placeholder="e.g. arun@school.edu",
                max_lines=1,
            )
            phone_inp = gr.Textbox(
                label="Phone Number (optional)",
                placeholder="e.g. +91 9876543210",
                max_lines=1,
            )
            school_inp = gr.Textbox(
                label="School / Institution Name",
                placeholder="e.g. Government Higher Secondary School",
                max_lines=1,
            )
            standard_inp = gr.Dropdown(
                label="Standard / Class",
                choices=STANDARDS,
                value=None,
                allow_custom_value=False,
            )
            board_inp = gr.Dropdown(
                label="Board",
                choices=BOARDS,
                value=None,
                allow_custom_value=False,
            )
            medium_inp = gr.Dropdown(
                label="Medium of Instruction",
                choices=INDIAN_LANGUAGES,
                value=None,
                allow_custom_value=False,
            )

            error_box = gr.HTML("<div id='signup-err'></div>")

            submit_btn = gr.Button(
                "Get Started →",
                variant="primary",
                elem_id="signup-submit",
            )

        def _on_submit(name, email, phone, school, standard, board, medium):
            err, signup_vis, tutor_vis, banner_update = handle_signup(
                name, email, phone, school, standard, board, medium
            )
            err_html = f"<div id='signup-err'>{err}</div>"
            if welcome_banner is not None:
                return err_html, signup_vis, tutor_vis, banner_update
            return err_html, signup_vis, tutor_vis

        outputs = [error_box, signup_col, tutor_column]
        if welcome_banner is not None:
            outputs.append(welcome_banner)

        submit_btn.click(
            fn=_on_submit,
            inputs=[name_inp, email_inp, phone_inp, school_inp, standard_inp, board_inp, medium_inp],
            outputs=outputs,
        )

    return signup_col
