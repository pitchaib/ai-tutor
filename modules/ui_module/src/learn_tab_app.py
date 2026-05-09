"""Gradio Learn UI for AI Personal Tutor (voice-first)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, List
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import gradio as gr

try:
    from signup_page import SIGNUP_BG_CSS, SIGNUP_CSS, build_signup_page
except Exception:
    from modules.ui_module.src.signup_page import SIGNUP_BG_CSS, SIGNUP_CSS, build_signup_page

try:
    from voice_qa_page import (
        INIT_JS as LESSON_INIT_JS,
        VOICE_JS as LESSON_VOICE_JS,
        run_lesson_step,
        start_lesson,
        toggle_pause_resume,
    )
except Exception:
    from modules.ui_module.src.voice_qa_page import (
        INIT_JS as LESSON_INIT_JS,
        VOICE_JS as LESSON_VOICE_JS,
        run_lesson_step,
        start_lesson,
        toggle_pause_resume,
    )


History = List[dict[str, str]]
LEARN_API_URL = os.getenv("LEARN_API_URL", "http://127.0.0.1:8000").rstrip("/")
PROJECT_ROOT = Path(os.getenv("AITUTOR_ROOT", "/home/bp/AiTutor"))
RESOURCE_DICT_PATH = PROJECT_ROOT / "modules/teacher_module/outputs/chapter_resource_dictionary.json"
RESOURCE_VOTES_PATH = PROJECT_ROOT / "modules/teacher_module/outputs/chapter_resource_votes.json"
FIXED_RESOURCE_CHAPTER = os.getenv("RESOURCE_CHAPTER", "Electrostatics").strip() or "Electrostatics"


def _load_resource_dictionary() -> dict[str, Any]:
    if not RESOURCE_DICT_PATH.exists():
        return {}
    try:
        with open(RESOURCE_DICT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _get_fixed_resources(limit: int = 3) -> list[dict[str, Any]]:
    db = _load_resource_dictionary()
    entry = db.get(FIXED_RESOURCE_CHAPTER)
    if not isinstance(entry, dict):
        # fallback: first chapter in dictionary
        for _, v in db.items():
            if isinstance(v, dict):
                entry = v
                break
    if not isinstance(entry, dict):
        return []
    resources = entry.get("resources", [])
    if not isinstance(resources, list):
        return []
    out: list[dict[str, Any]] = []
    for r in resources[:limit]:
        if not isinstance(r, dict):
            continue
        out.append(
            {
                "title": str(r.get("title", "")).strip(),
                "url": str(r.get("url", "")).strip(),
                "type": str(r.get("type", "article")).strip(),
                "reason": str(r.get("reason", "")).strip(),
            }
        )
    return out


RESOURCES_FIXED = _get_fixed_resources(limit=3)


def _load_votes() -> dict[str, Any]:
    if not RESOURCE_VOTES_PATH.exists():
        return {}
    try:
        with open(RESOURCE_VOTES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_votes(votes: dict[str, Any]) -> None:
    RESOURCE_VOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESOURCE_VOTES_PATH, "w", encoding="utf-8") as f:
        json.dump(votes, f, ensure_ascii=False, indent=2)


def _resource_label(index: int) -> str:
    if index < 0 or index >= len(RESOURCES_FIXED):
        return f"**{index + 1}.** Resource not available."
    r = RESOURCES_FIXED[index]
    title = r["title"] or "Untitled resource"
    url = r["url"] or "#"
    rtype = (r.get("type") or "article").lower()
    type_symbol = {
        "youtube": "🎥",
        "video": "🎥",
        "blog": "📝",
        "medium": "📝",
        "edutech": "🎓",
        "article": "📄",
    }.get(rtype, "🔗")
    # Single-line recommendation with type symbol (rank shown separately in UI).
    return f"**[{title}]({url}) {type_symbol}**"


def _resource_counts_markdown(index: int) -> str:
    if index < 0 or index >= len(RESOURCES_FIXED):
        return "👍 **0**  |  👎 **0**"
    url = RESOURCES_FIXED[index].get("url", "")
    votes = _load_votes()
    chapter_votes = votes.get(FIXED_RESOURCE_CHAPTER, {}) if isinstance(votes.get(FIXED_RESOURCE_CHAPTER, {}), dict) else {}
    row = chapter_votes.get(url, {}) if isinstance(chapter_votes.get(url, {}), dict) else {}
    likes = int(row.get("likes", 0) or 0)
    dislikes = int(row.get("dislikes", 0) or 0)
    return f"👍 **{likes}**  |  👎 **{dislikes}**"


def _all_resource_counts() -> tuple[str, str, str]:
    vals = [_resource_counts_markdown(i) for i in range(3)]
    while len(vals) < 3:
        vals.append("👍 **0**  |  👎 **0**")
    return (vals[0], vals[1], vals[2])


def vote_resource(index: int, vote_kind: str) -> tuple[str, str, str]:
    if index < 0 or index >= len(RESOURCES_FIXED):
        return _all_resource_counts()
    url = RESOURCES_FIXED[index].get("url", "")
    if not url:
        return _all_resource_counts()
    votes = _load_votes()
    chapter_votes = votes.setdefault(FIXED_RESOURCE_CHAPTER, {})
    if not isinstance(chapter_votes, dict):
        chapter_votes = {}
        votes[FIXED_RESOURCE_CHAPTER] = chapter_votes
    row = chapter_votes.setdefault(url, {"likes": 0, "dislikes": 0})
    if not isinstance(row, dict):
        row = {"likes": 0, "dislikes": 0}
        chapter_votes[url] = row
    if vote_kind == "like":
        row["likes"] = int(row.get("likes", 0) or 0) + 1
    elif vote_kind == "dislike":
        row["dislikes"] = int(row.get("dislikes", 0) or 0) + 1
    _save_votes(votes)
    return _all_resource_counts()


def update_current_learning(subject: str, chapter: str) -> str:
    """Update the current learning markdown label."""
    safe_subject = subject or "Not selected"
    safe_chapter = chapter or "Not selected"
    return f"### 📘 Currently learning: {safe_subject} - {safe_chapter}"


def handle_text_input(text: str, history: History) -> tuple[str, History]:
    """Append text question and return mock tutor response."""
    history = history or []
    text = (text or "").strip()
    if not text:
        return "", history

    updated = history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": "This is a sample explanation for your question."},
    ]
    return "", updated


def handle_audio_input(audio: str, history: History) -> History:
    """Append audio question event and return mock tutor response."""
    history = history or []
    if not audio:
        return history

    # Keep it simple for now: indicate that voice input was received.
    audio_name = os.path.basename(audio)
    user_msg = f"[Voice question received: {audio_name}]"
    updated = history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": "This is a sample explanation for your question."},
    ]
    return updated


def _http_get(path: str, query: dict[str, str | int] | None = None) -> dict:
    qs = f"?{urlencode(query)}" if query else ""
    req = Request(f"{LEARN_API_URL}{path}{qs}", method="GET")
    with urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post(path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        f"{LEARN_API_URL}{path}",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _format_learning(subject: str, chapter: str) -> str:
    return f"### 📘 Currently learning: {subject or 'Not selected'} - {chapter or 'Not selected'}"


def _chapter_selected(chapter: str) -> bool:
    c = (chapter or "").strip()
    return bool(c) and c.lower() != "none"


def _page_image_local_path(page_no: int) -> str | None:
    """Fetch page image from API and return local file path for Gradio Image."""
    try:
        req = Request(f"{LEARN_API_URL}/learn/page_image?page_no={int(page_no)}", method="GET")
        with urlopen(req, timeout=90) as resp:
            data = resp.read()
        tmp = tempfile.NamedTemporaryFile(prefix=f"book_page_{int(page_no)}_", suffix=".png", delete=False)
        tmp.write(data)
        tmp.flush()
        tmp.close()
        return tmp.name
    except Exception:  # noqa: BLE001
        return None


def _language_explanation(resp: dict, language: str) -> str:
    lang = (language or "English").strip().lower()
    en = str(resp.get("teacher_explanation", "")).strip()
    ta = str(resp.get("teacher_explanation_tamil", "")).strip()
    if lang == "tamil":
        return ta or en
    if lang == "mix":
        if en and ta:
            return f"English:\n{en}\n\nTamil:\n{ta}"
        return en or ta
    return en or ta


def _response_to_ui(resp: dict, chapter: str, language: str) -> tuple[int, History, str | None]:
    page_no = int(resp.get("page_no", 1))
    explanation = _language_explanation(resp, language)
    if not explanation:
        explanation = "No explanation available for this page yet."
    chat: History = [{"role": "assistant", "content": explanation}]
    return (
        page_no,
        chat,
        _page_image_local_path(page_no),
    )


def load_chapter_init(subject: str, chapter: str, language: str) -> tuple[int, History, str, str | None]:
    """Fetch default page for selected chapter from FastAPI."""
    if not _chapter_selected(chapter):
        return (
            1,
            [{"role": "assistant", "content": "Select a chapter to load explanation."}],
            _format_learning(subject, chapter),
            None,
        )
    try:
        resp = _http_get("/learn/init", {"chapter_name": chapter})
        page_no, chat, image_url = _response_to_ui(resp, chapter, language)
        return page_no, chat, _format_learning(subject, chapter), image_url
    except Exception as e:  # noqa: BLE001
        msg = f"Could not load from API ({e}). Start backend: python src/learn_api.py"
        return (
            1,
            [{"role": "assistant", "content": f"⚠️ API Error\n\n{msg}"}],
            _format_learning(subject, chapter),
            None,
        )


def navigate_page(chapter: str, current_page: int, action: str, language: str) -> tuple[int, History, str | None]:
    """Navigate previous/next page via FastAPI."""
    if not _chapter_selected(chapter):
        page = int(current_page or 1)
        return (
            page,
            [{"role": "assistant", "content": "Select a chapter first."}],
            _page_image_local_path(page),
        )
    try:
        resp = _http_post(
            "/learn/navigate",
            {
                "chapter_name": chapter,
                "current_page": int(current_page or 1),
                "action": action,
            },
        )
        return _response_to_ui(resp, chapter, language)
    except Exception as e:  # noqa: BLE001
        msg = f"Navigation failed ({e})."
        page = int(current_page or 1)
        return (
            page,
            [{"role": "assistant", "content": f"⚠️ API Error\n\n{msg}"}],
            _page_image_local_path(page),
        )


def refresh_language(chapter: str, current_page: int, language: str) -> tuple[int, History, str | None]:
    """Reload current page and re-render explanation in selected language."""
    if not _chapter_selected(chapter):
        page = int(current_page or 1)
        return (
            page,
            [{"role": "assistant", "content": "Select a chapter first."}],
            _page_image_local_path(page),
        )
    try:
        resp = _http_post(
            "/learn/page",
            {
                "chapter_name": chapter,
                "page_no": int(current_page or 1),
            },
        )
        return _response_to_ui(resp, chapter, language)
    except Exception as e:  # noqa: BLE001
        page = int(current_page or 1)
        msg = f"Language refresh failed ({e})."
        return (
            page,
            [{"role": "assistant", "content": f"⚠️ API Error\n\n{msg}"}],
            _page_image_local_path(page),
        )


def _audio_path_for_language(resp: dict, language: str) -> str:
    lang = (language or "English").strip().lower()
    en = str(resp.get("teacher_explanation_audio_english_path", "")).strip()
    ta = str(resp.get("teacher_explanation_audio_tamil_path", "")).strip()
    if lang == "tamil":
        return ta or en
    if lang == "mix":
        return en or ta
    return en or ta


def load_voice_for_page(chapter: str, page_no: int, language: str) -> str | None:
    """Load pre-generated WAV path from API/session based on selected language."""
    if not _chapter_selected(chapter):
        return None
    try:
        resp = _http_post(
            "/learn/page",
            {
                "chapter_name": chapter,
                "page_no": int(page_no or 1),
            },
        )
        path = _audio_path_for_language(resp, language)
        if not path:
            return None
        p = Path(path)
        if p.exists() and p.is_file():
            # Gradio only allows files from cwd/temp unless explicitly allowed.
            # Copy to temp and return that path to avoid InvalidPathError.
            with open(p, "rb") as src:
                data = src.read()
            tmp = tempfile.NamedTemporaryFile(
                prefix=f"voice_page_{int(page_no or 1)}_",
                suffix=".wav",
                delete=False,
            )
            tmp.write(data)
            tmp.flush()
            tmp.close()
            return tmp.name
        return None
    except Exception:  # noqa: BLE001
        return None


CUSTOM_CSS = """
/* Soft classroom-like background with blur + readability overlay */
html, body, #root {
    min-height: 100%;
}

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
    background: linear-gradient(
        rgba(226, 239, 254, 0.40),
        rgba(241, 248, 255, 0.34)
    );
    z-index: 0;
    pointer-events: none;
}

.gradio-container::after {
    content: "";
    position: fixed;
    inset: 0;
    background:
      radial-gradient(circle at 12% 18%, rgba(112, 170, 225, 0.18), transparent 32%),
      radial-gradient(circle at 82% 26%, rgba(255, 210, 125, 0.17), transparent 33%),
      radial-gradient(circle at 50% 82%, rgba(158, 196, 245, 0.14), transparent 34%);
    z-index: 0;
    pointer-events: none;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    position: relative;
    z-index: 1 !important;
    padding-top: 20px !important;
    padding-bottom: 24px !important;
    background: transparent !important;
}

/* Keep cards/components readable over image */
.gr-box,
.gr-panel,
.block,
.gradio-container .form,
.gradio-container .gr-group {
    background: rgba(255, 255, 255, 0.88) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(80, 100, 130, 0.14) !important;
    box-shadow: 0 12px 26px rgba(33, 51, 79, 0.08) !important;
}

#chatbot {
    min-height: 520px !important;
    max-height: 560px !important;
    overflow-y: auto !important;
}

.center-action {
    display: flex;
    justify-content: center;
    margin-top: 12px;
}

.center-action button {
    min-width: 280px !important;
}

#audio_b64, #process_btn, #lesson_mode, #lesson_paused {
  display: none !important;
}
#lesson_audio {
  opacity: 0 !important;
  max-height: 0 !important;
  min-height: 0 !important;
  overflow: hidden !important;
  margin: 0 !important;
  padding: 0 !important;
}
#ask_btn button {
  background: #2356d8 !important;
  color: #fff !important;
  border: none !important;
  min-height: 48px !important;
  font-weight: 700 !important;
  transition: all 0.2s ease !important;
}
#ask_btn button.ask-recording {
  background: #d62839 !important;
  box-shadow: 0 0 0 3px rgba(214, 40, 57, 0.18) !important;
}
#ask_btn button.ask-processing {
  background: #f59f00 !important;
  color: #111 !important;
}
#record_wave {
  display: none;
  align-items: center;
  gap: 4px;
  height: 30px;
  margin: 8px 0 2px;
}
#record_wave.active {
  display: inline-flex;
}
#record_wave span {
  width: 4px;
  height: 8px;
  border-radius: 4px;
  background: #d62839;
  animation: wavePulse 1s ease-in-out infinite;
}
#record_wave span:nth-child(2) { animation-delay: 0.1s; }
#record_wave span:nth-child(3) { animation-delay: 0.2s; }
#record_wave span:nth-child(4) { animation-delay: 0.3s; }
#record_wave span:nth-child(5) { animation-delay: 0.4s; }
#record_wave span:nth-child(6) { animation-delay: 0.5s; }
@keyframes wavePulse {
  0%, 100% { transform: scaleY(0.7); opacity: 0.65; }
  50% { transform: scaleY(2.2); opacity: 1; }
}
#ai_voice_anim {
  width: 92px;
  height: 92px;
  border-radius: 46px;
  margin: 8px auto 10px;
  position: relative;
  background: radial-gradient(circle at 30% 30%, #a5c8ff, #3b6cc9);
  box-shadow: 0 0 0 0 rgba(59,108,201,0.45);
}
#ai_voice_anim::before,
#ai_voice_anim::after {
  content: "";
  position: absolute;
  inset: 18px;
  border-radius: 50%;
  border: 3px solid rgba(255, 255, 255, 0.75);
}
#ai_voice_anim::after {
  inset: 30px;
  border-color: rgba(255, 255, 255, 0.45);
}
#ai_voice_anim.playing {
  animation: aiPulse 1.1s ease-in-out infinite;
}
@keyframes aiPulse {
  0% { transform: scale(0.96); box-shadow: 0 0 0 0 rgba(59,108,201,0.35); }
  50% { transform: scale(1.05); box-shadow: 0 0 0 16px rgba(59,108,201,0.08); }
  100% { transform: scale(0.96); box-shadow: 0 0 0 0 rgba(59,108,201,0.35); }
}

.resource-card {
  align-items: center !important;
  border: 1px solid rgba(98, 129, 168, 0.25);
  border-radius: 14px;
  padding: 8px 10px;
  background: rgba(255, 255, 255, 0.75);
}
.resource-rank p {
  font-size: 34px !important;
  font-weight: 700 !important;
  color: #2e5fae !important;
  margin: 0 !important;
  line-height: 1 !important;
}
.resource-title p {
  margin: 0 0 2px 0 !important;
  font-size: 20px !important;
  font-weight: 600 !important;
  line-height: 1.25 !important;
}
.resource-count p {
  margin: 2px 0 0 0 !important;
  color: #415b85 !important;
  font-size: 16px !important;
  line-height: 1.1 !important;
}
.resource-actions button {
  min-width: 62px !important;
  background: rgba(255, 255, 255, 0.75) !important;
  border: 1px solid rgba(98, 129, 168, 0.25) !important;
  color: #2e5fae !important;
  box-shadow: none !important;
}
.resource-actions button:hover {
  background: rgba(245, 250, 255, 0.9) !important;
}
.resource-card a {
  color: #2e5fae !important;
  text-decoration: none !important;
}
.resource-card a:hover {
  text-decoration: underline !important;
}
"""


def build_demo() -> gr.Blocks:
    """Build the Learn tab UI as a standalone Gradio Blocks interface."""
    with gr.Blocks(title="AI Personal Tutor") as demo:
        gr.HTML(f"<style>{SIGNUP_BG_CSS}{SIGNUP_CSS}{CUSTOM_CSS}</style>")

        # ── Tutor column (hidden until signup is completed) ──────────────────
        with gr.Column(visible=False, elem_id="tutor-main") as tutor_col:
            welcome_banner = gr.Markdown("# 🎓 AI Personal Tutor", elem_id="welcome-banner")

            # 1) Top section: context selection
            gr.Markdown("## Concept Overview")
            with gr.Row():
                subject = gr.Dropdown(
                    label="Subject",
                    choices=["None", "Physics", "Chemistry", "Mathematics", "Biology"],
                    value="None",
                )
                chapter = gr.Dropdown(
                    label="Chapter",
                    choices=[
                        "None",
                        "Electrostatics",
                        "Current Electricity",
                        "Magnetism",
                        "Optics",
                    ],
                    value="None",
                )
                language = gr.Radio(
                    label="Language",
                    choices=["Tamil", "English", "Mix"],
                    value="English",
                )
            current_learning = gr.Markdown("### 📘 Currently learning: None - None")
            page_no = gr.Number(label="Page No", value=1, precision=0, interactive=False)

            # 2) Main section: large chatbot
            gr.Markdown("## Tutor Guidance")
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="Tutor Conversation", elem_id="chatbot")
                with gr.Column(scale=2):
                    book_page_preview = gr.Image(
                        label="📖 Book Page Preview",
                        interactive=False,
                    )
            with gr.Row():
                prev_page_btn = gr.Button("⬅️ Previous Page")
                next_page_btn = gr.Button("Next Page ➡️")

            # 3) Voice playback section
            gr.Markdown("## Listen", elem_id="listen-section")
            with gr.Row():
                play_voice_btn = gr.Button("▶️ Generate Voice Playback", variant="secondary")
            voice_output = gr.Audio(label="Tutor Voice Playback", interactive=False)

            # 4) Interactive lesson voice mode (start/pause/ask-answer)
            gr.Markdown("## Tutor Mode", elem_id="tutor-section")
            with gr.Row():
                asr_language = gr.Radio(
                    label="Answer ASR Language",
                    choices=["English (en-US)", "Tamil (ta-IN)"],
                    value="English (en-US)",
                )
            with gr.Row():
                start_lesson_btn = gr.Button("Start Lesson", variant="secondary")
                pause_lesson_btn = gr.Button("Pause Lesson", variant="secondary", elem_id="pause_btn")
                ask_answer_btn = gr.Button("Ask/Answer", variant="primary", elem_id="ask_btn")
            gr.HTML(
                "<div id='record_wave'>"
                "<span></span><span></span><span></span><span></span><span></span><span></span>"
                "</div>"
            )
            gr.HTML("<div id='ai_voice_anim' title='AI voice playing'></div>")
            lesson_state = gr.State(value={})
            lesson_audio_b64 = gr.Textbox(value="", label="", elem_id="audio_b64")
            lesson_mode = gr.Textbox(value="idle", label="", elem_id="lesson_mode")
            lesson_paused = gr.Textbox(value="false", label="", elem_id="lesson_paused")
            process_lesson_btn = gr.Button("Process", elem_id="process_btn")
            lesson_audio = gr.Audio(label="Lesson Audio", interactive=False, autoplay=True, elem_id="lesson_audio")
            lesson_status = gr.Markdown("Click Start Lesson to begin.", elem_id="status_md")

            # 5) Additional resources section
            gr.Markdown("## Explore More")
            with gr.Group():
                with gr.Row(elem_classes=["resource-card"]):
                    rank_1 = gr.Markdown("1.", elem_classes=["resource-rank"])
                    with gr.Column(scale=8):
                        resource_1 = gr.Markdown(_resource_label(0))
                        count_1 = gr.Markdown(_resource_counts_markdown(0), elem_classes=["resource-count"])
                    with gr.Column(scale=2, elem_classes=["resource-actions"]):
                        like_1 = gr.Button("👍", variant="secondary")
                        dislike_1 = gr.Button("👎", variant="secondary")
            with gr.Group():
                with gr.Row(elem_classes=["resource-card"]):
                    rank_2 = gr.Markdown("2.", elem_classes=["resource-rank"])
                    with gr.Column(scale=8):
                        resource_2 = gr.Markdown(_resource_label(1))
                        count_2 = gr.Markdown(_resource_counts_markdown(1), elem_classes=["resource-count"])
                    with gr.Column(scale=2, elem_classes=["resource-actions"]):
                        like_2 = gr.Button("👍", variant="secondary")
                        dislike_2 = gr.Button("👎", variant="secondary")
            with gr.Group():
                with gr.Row(elem_classes=["resource-card"]):
                    rank_3 = gr.Markdown("3.", elem_classes=["resource-rank"])
                    with gr.Column(scale=8):
                        resource_3 = gr.Markdown(_resource_label(2))
                        count_3 = gr.Markdown(_resource_counts_markdown(2), elem_classes=["resource-count"])
                    with gr.Column(scale=2, elem_classes=["resource-actions"]):
                        like_3 = gr.Button("👍", variant="secondary")
                        dislike_3 = gr.Button("👎", variant="secondary")
            with gr.Row(elem_classes=["center-action"]):
                test_understanding_btn = gr.Button("🧠 Test My Understanding", variant="primary")

        # ── Signup page (shown first, hides itself and reveals tutor_col) ────
        signup_col = build_signup_page(tutor_col, welcome_banner)  # noqa: F841

        # ── Event wiring (must be outside column contexts) ───────────────────
        # Context label updates
        subject.change(fn=update_current_learning, inputs=[subject, chapter], outputs=current_learning)
        chapter.change(
            fn=load_chapter_init,
            inputs=[subject, chapter, language],
            outputs=[page_no, chatbot, current_learning, book_page_preview],
        )
        prev_page_btn.click(
            fn=lambda ch, p, lang: navigate_page(ch, p, "previous", lang),
            inputs=[chapter, page_no, language],
            outputs=[page_no, chatbot, book_page_preview],
        )
        next_page_btn.click(
            fn=lambda ch, p, lang: navigate_page(ch, p, "next", lang),
            inputs=[chapter, page_no, language],
            outputs=[page_no, chatbot, book_page_preview],
        )
        language.change(
            fn=refresh_language,
            inputs=[chapter, page_no, language],
            outputs=[page_no, chatbot, book_page_preview],
        )

        play_voice_btn.click(
            fn=load_voice_for_page,
            inputs=[chapter, page_no, language],
            outputs=voice_output,
        )
        start_lesson_btn.click(
            fn=lambda ch, lang: start_lesson("Electrostatics" if not _chapter_selected(ch) else ch, lang),
            inputs=[chapter, language],
            outputs=[
                lesson_state,
                lesson_audio,
                lesson_status,
                ask_answer_btn,
                lesson_mode,
                lesson_audio_b64,
                lesson_paused,
                pause_lesson_btn,
            ],
        )
        process_event = process_lesson_btn.click(
            fn=run_lesson_step,
            inputs=[lesson_audio_b64, lesson_state, chapter, language, asr_language],
            outputs=[
                lesson_state,
                lesson_audio,
                lesson_status,
                ask_answer_btn,
                lesson_mode,
                lesson_audio_b64,
            ],
        )
        pause_lesson_btn.click(
            fn=toggle_pause_resume,
            inputs=[lesson_state],
            outputs=[lesson_state, lesson_status, lesson_paused, pause_lesson_btn, lesson_mode, lesson_audio],
            queue=False,
            cancels=[process_event],
        )
        ask_answer_btn.click(fn=None, inputs=[], outputs=[], js=LESSON_VOICE_JS)
        like_1.click(
            fn=lambda: vote_resource(0, "like"),
            inputs=[],
            outputs=[count_1, count_2, count_3],
        )
        dislike_1.click(
            fn=lambda: vote_resource(0, "dislike"),
            inputs=[],
            outputs=[count_1, count_2, count_3],
        )
        like_2.click(
            fn=lambda: vote_resource(1, "like"),
            inputs=[],
            outputs=[count_1, count_2, count_3],
        )
        dislike_2.click(
            fn=lambda: vote_resource(1, "dislike"),
            inputs=[],
            outputs=[count_1, count_2, count_3],
        )
        like_3.click(
            fn=lambda: vote_resource(2, "like"),
            inputs=[],
            outputs=[count_1, count_2, count_3],
        )
        dislike_3.click(
            fn=lambda: vote_resource(2, "dislike"),
            inputs=[],
            outputs=[count_1, count_2, count_3],
        )
        test_understanding_btn.click(
            fn=None,
            inputs=[chapter, page_no],
            outputs=[],
            js=f"""
            (chapterName, pageNo) => {{
                const page = Math.max(1, parseInt(pageNo || 1));
                const url = "{LEARN_API_URL}/quiz?chapter_name=" + encodeURIComponent(chapterName || "Electrostatics") + "&current_page=" + page;
                window.open(url, "_blank");
                return [];
            }}
            """,
        )
        demo.load(
            fn=lambda s, c, l: (
                1,
                [{"role": "assistant", "content": "Select a chapter to begin."}],
                _format_learning(s, c),
                None,
            ),
            inputs=[subject, chapter, language],
            outputs=[page_no, chatbot, current_learning, book_page_preview],
        )
        demo.load(fn=None, inputs=[], outputs=[], js=LESSON_INIT_JS)

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch(server_name="127.0.0.1", server_port=7860)

