"""
AI Tutor — HTML frontend server.

Routes
------
GET  /              Signup page
POST /signup        Validate + set session cookie, redirect to /welcome
GET  /welcome       Personalised welcome / tutor launcher page
GET  /app           Redirect to Gradio lesson UI (port 7860)
GET  /static/*      Static assets (CSS, JS, images)
GET  /health        Health check
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import tempfile
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import httpx
from fastapi import Cookie, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ── Paths ──────────────────────────────────────────────────────────────────
MODULE_DIR = Path(__file__).parent.parent          # .../ui_module_html
TEMPLATES_DIR = MODULE_DIR / "templates"
STATIC_DIR = MODULE_DIR / "static"
PROJECT_ROOT = Path(os.getenv("AITUTOR_ROOT", "/home/bp/AiTutor"))
RESOURCE_DICT_PATH = PROJECT_ROOT / "modules/teacher_module/outputs/chapter_resource_dictionary.json"
RESOURCE_FEEDBACK_PATH = PROJECT_ROOT / "modules/teacher_module/outputs/resource_feedback.json"
TEACHING_SESSION_PATH = PROJECT_ROOT / "modules/teacher_module/outputs/teaching_session.json"
TUTOR_OUTPUT_ROOT = PROJECT_ROOT / "outputs/voice_qa/lesson"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GRADIO_URL = os.getenv("GRADIO_URL", "http://127.0.0.1:7860")
LEARN_API_URL = os.getenv("LEARN_API_URL", "http://127.0.0.1:8000").rstrip("/")

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="AI Tutor — HTML Frontend", version="1.0.0")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ── In-memory session store (replace with Redis / DB for production) ───────
_sessions: dict[str, dict] = {}

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

LANGUAGES = [
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

# Classes 6 – 12 shown as ordinal English labels in the dropdown.
STANDARDS = ["6th", "7th", "8th", "9th", "10th", "11th", "12th"]


def _validate(
    name: str,
    email: str,
    phone: str,
    school: str,
    standard: str,
    board: str,
    medium: str,
) -> str | None:
    if not name.strip():
        return "Please enter your full name."
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email.strip()):
        return "Please enter a valid email address."
    if phone.strip() and not re.fullmatch(r"[+\d\s\-()]{7,15}", phone.strip()):
        return "Phone number looks invalid. Leave blank if not applicable."
    if not school.strip():
        return "Please enter your school / institution name."
    if not standard or standard not in STANDARDS:
        return "Please select your standard / class."
    if not board or board not in BOARDS:
        return "Please select your board."
    if not medium or medium not in LANGUAGES:
        return "Please select medium of instruction."
    return None


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse(
        request,
        "signup.html",
        {"boards": BOARDS, "languages": LANGUAGES, "standards": STANDARDS, "error": None},
    )


@app.post("/signup")
async def handle_signup(
    request: Request,
    name: Annotated[str, Form()],
    email: Annotated[str, Form()],
    phone: Annotated[str, Form()] = "",
    school: Annotated[str, Form()] = "",
    standard: Annotated[str, Form()] = "",
    board: Annotated[str, Form()] = "",
    medium: Annotated[str, Form()] = "",
):
    err = _validate(name, email, phone, school, standard, board, medium)
    if err:
        return templates.TemplateResponse(
            request,
            "signup.html",
            {
                "boards": BOARDS,
                "languages": LANGUAGES,
                "standards": STANDARDS,
                "error": err,
                "prev": {
                    "name": name, "email": email, "phone": phone,
                    "school": school, "standard": standard,
                    "board": board, "medium": medium,
                },
            },
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    sid = uuid.uuid4().hex
    _sessions[sid] = {
        "name": name.strip(),
        "email": email.strip(),
        "phone": phone.strip(),
        "school": school.strip(),
        "standard": standard,
        "board": board,
        "medium": medium,
    }
    response = RedirectResponse(url="/welcome", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(key="aitutor_sid", value=sid, httponly=True, max_age=86400)
    return response


def _session_ctx(aitutor_sid: str | None) -> dict:
    """Return {student, first_name} based on the session cookie."""
    student = _sessions.get(aitutor_sid or "", {}) if aitutor_sid else {}
    first_name = (student.get("name") or "Student").split()[0]
    return {"student": student, "first_name": first_name}


def _load_resources_raw() -> dict:
    """Read curated chapter-resource dictionary; return {} if unreadable."""
    if not RESOURCE_DICT_PATH.exists():
        return {}
    try:
        with open(RESOURCE_DICT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ── Resource feedback (like / dislike) ────────────────────────────────────
# Shape on disk:
#   { <chapter>: { <url>: { "likes": int, "dislikes": int,
#                           "voters": { <sid>: "like"|"dislike" } } } }
_FEEDBACK_LOCK_NOTE = "single-process uvicorn assumed; swap for Redis for prod"


def _load_feedback() -> dict:
    if not RESOURCE_FEEDBACK_PATH.exists():
        return {}
    try:
        with open(RESOURCE_FEEDBACK_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_feedback(fb: dict) -> None:
    RESOURCE_FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESOURCE_FEEDBACK_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(fb, f, ensure_ascii=False, indent=2)
    tmp.replace(RESOURCE_FEEDBACK_PATH)


def _entry_for(fb: dict, chapter: str, url: str) -> dict:
    return fb.setdefault(chapter, {}).setdefault(
        url, {"likes": 0, "dislikes": 0, "voters": {}}
    )


def _rating(curator_score: float, likes: int, dislikes: int) -> float:
    """Weighted community rating.

    Gives the curator score strong prior weight (10 virtual votes) so a single
    opposite student vote doesn't swing a 9.8 to 5. Each like adds 10 points,
    each dislike adds 0. Keeps values in the 0–10 range.
    """
    prior_weight = 10.0
    total_votes = likes + dislikes
    if total_votes == 0:
        return round(float(curator_score), 1)
    numerator = curator_score * prior_weight + 10.0 * likes + 0.0 * dislikes
    denominator = prior_weight + total_votes
    return round(numerator / denominator, 1)


def _load_resources(sid: str | None = None) -> dict:
    """Curated resources with per-resource feedback + this-user's vote merged in."""
    raw = _load_resources_raw()
    if not raw:
        return {}
    fb = _load_feedback()
    out: dict = {}
    for chapter, entry in raw.items():
        chapter_fb = fb.get(chapter, {})
        merged_resources = []
        for r in entry.get("resources", []):
            url = r.get("url") or ""
            rec = chapter_fb.get(url, {})
            likes = int(rec.get("likes", 0))
            dislikes = int(rec.get("dislikes", 0))
            my_vote = (rec.get("voters") or {}).get(sid or "", "")
            curator_score = float(r.get("score", 0) or 0)
            merged_resources.append({
                **r,
                "likes": likes,
                "dislikes": dislikes,
                "my_vote": my_vote,  # "", "like", or "dislike"
                "rating": _rating(curator_score, likes, dislikes),
            })
        out[chapter] = {**entry, "resources": merged_resources}
    return out


@app.get("/welcome", response_class=HTMLResponse)
async def welcome_page(request: Request, aitutor_sid: Annotated[str | None, Cookie()] = None):
    ctx = _session_ctx(aitutor_sid)
    return templates.TemplateResponse(
        request,
        "welcome.html",
        {**ctx, "gradio_url": GRADIO_URL, "active_page": "home"},
    )


@app.get("/textbook", response_class=HTMLResponse)
async def textbook_page(request: Request, aitutor_sid: Annotated[str | None, Cookie()] = None):
    ctx = _session_ctx(aitutor_sid)
    return templates.TemplateResponse(
        request, "textbook.html",
        {**ctx, "gradio_url": GRADIO_URL, "active_page": "textbook"},
    )


@app.get("/voice", response_class=HTMLResponse)
async def voice_page(request: Request, aitutor_sid: Annotated[str | None, Cookie()] = None):
    ctx = _session_ctx(aitutor_sid)
    return templates.TemplateResponse(
        request, "voice.html",
        {**ctx, "gradio_url": GRADIO_URL, "active_page": "voice"},
    )


@app.get("/practice", response_class=HTMLResponse)
async def practice_page(request: Request, aitutor_sid: Annotated[str | None, Cookie()] = None):
    ctx = _session_ctx(aitutor_sid)
    return templates.TemplateResponse(
        request, "practice.html",
        {**ctx, "gradio_url": GRADIO_URL, "active_page": "practice"},
    )


@app.get("/resources", response_class=HTMLResponse)
async def resources_page(request: Request, aitutor_sid: Annotated[str | None, Cookie()] = None):
    ctx = _session_ctx(aitutor_sid)
    return templates.TemplateResponse(
        request, "resources.html",
        {
            **ctx,
            "gradio_url": GRADIO_URL,
            "active_page": "resources",
            "resources": _load_resources(sid=aitutor_sid),
        },
    )


# ── Listen Mode (audiobook-style playback from teaching_session.json) ─────
# Everything here is static/disk-backed: no Vertex call is needed.
def _load_teaching_session() -> dict:
    if not TEACHING_SESSION_PATH.exists():
        return {}
    try:
        with open(TEACHING_SESSION_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _listen_sections() -> list[dict]:
    """Flatten teaching_session.json into a list of section dicts."""
    data = _load_teaching_session()
    out: list[dict] = []
    for chapter in data.get("chapters", []):
        for section in chapter.get("sections", []):
            out.append(section)
    out.sort(key=lambda s: int(s.get("section_id", 0)))
    return out


def _listen_chapter_name() -> str:
    """Derive a readable chapter name from the pdf_path."""
    data = _load_teaching_session()
    pdf = str(data.get("pdf_path") or "")
    # e.g. ".../Physics/books/Class_12_Physics_English_Volume_1_2024_Edition-www.tntextbooks.in.pdf"
    stem = Path(pdf).stem.replace("_", " ")
    if stem:
        return stem
    return "Lesson"


def _audio_path_for(section: dict, lang: str) -> Path | None:
    key = (
        "teacher_explanation_audio_tamil_path"
        if lang == "tamil"
        else "teacher_explanation_audio_english_path"
    )
    path = section.get(key)
    if not path:
        return None
    return Path(str(path))


def _explanation_for(section: dict, lang: str) -> str:
    if lang == "tamil":
        return str(section.get("teacher_explanation_tamil", "") or "").strip()
    return str(section.get("teacher_explanation", "") or "").strip()


@app.get("/api/listen/session")
async def listen_session():
    """Return the section list with language availability — no audio bytes."""
    sections = _listen_sections()
    if not sections:
        raise HTTPException(status_code=404, detail="teaching_session.json not found")
    summary = []
    for sec in sections:
        en = _audio_path_for(sec, "english")
        ta = _audio_path_for(sec, "tamil")
        preview = (str(sec.get("content") or "")).strip().replace("\n", " ")
        if len(preview) > 240:
            preview = preview[:240].rsplit(" ", 1)[0] + "…"
        summary.append({
            "section_id": int(sec.get("section_id") or 0),
            "title": str(sec.get("title") or "").strip(),
            "page_no": int(sec.get("page_no") or 0),
            "preview": preview,
            "has_english": bool(en and en.exists()),
            "has_tamil": bool(ta and ta.exists()),
        })
    return {
        "chapter": _listen_chapter_name(),
        "total_sections": len(summary),
        "sections": summary,
    }


@app.get("/api/listen/explanation")
async def listen_explanation(section_id: int, lang: str = "english"):
    lang_norm = "tamil" if (lang or "").lower().startswith("ta") else "english"
    for sec in _listen_sections():
        if int(sec.get("section_id") or 0) == int(section_id):
            return {
                "section_id": section_id,
                "lang": lang_norm,
                "title": str(sec.get("title") or "").strip(),
                "page_no": int(sec.get("page_no") or 0),
                "markdown": _explanation_for(sec, lang_norm),
            }
    raise HTTPException(status_code=404, detail=f"section_id {section_id} not found")


@app.get("/api/listen/audio")
async def listen_audio(section_id: int, lang: str = "english"):
    lang_norm = "tamil" if (lang or "").lower().startswith("ta") else "english"
    for sec in _listen_sections():
        if int(sec.get("section_id") or 0) == int(section_id):
            path = _audio_path_for(sec, lang_norm)
            if not path:
                raise HTTPException(status_code=404, detail=f"No {lang_norm} audio for this section")
            # Path-traversal guard: only serve the literal path declared in the JSON.
            if not path.exists():
                raise HTTPException(status_code=404, detail=f"Audio file missing on disk: {path}")
            return FileResponse(
                path,
                media_type="audio/wav",
                filename=path.name,
                headers={"Cache-Control": "public, max-age=3600"},
            )
    raise HTTPException(status_code=404, detail=f"section_id {section_id} not found")


# ── Tutor Mode (interactive Q&A via VoiceQAPipeline) ──────────────────────
# Sessions live in this process; a full tutor session state is kept server-side
# so the client only needs to POST audio and replay the returned WAV.
_tutor_sessions: dict[str, dict] = {}
_tutor_lock = threading.Lock()
_tutor_pipeline_cache: list = []  # boxed so we can lazy-init


def _get_voice_qa_pipeline():
    """Lazy-instantiate VoiceQAPipeline (Vertex init happens here)."""
    if _tutor_pipeline_cache:
        return _tutor_pipeline_cache[0]
    from modules.voice_qa_module.src.voice_qa_pipeline import (
        VoiceQAPipeline,
        VoiceQAPipelineConfig,
    )
    cfg = VoiceQAPipelineConfig(
        dictionary_json=PROJECT_ROOT / "modules/teacher_module/outputs/chunk_summary_dictionary.json",
        output_dir=PROJECT_ROOT / "outputs/voice_qa",
    )
    pipeline = VoiceQAPipeline(cfg)
    _tutor_pipeline_cache.append(pipeline)
    return pipeline


@app.on_event("startup")
def _warm_tutor_pipeline() -> None:
    """Warm up the tutor pipeline at server boot.

    Lazy-initialises VoiceQAPipeline (loads dictionary, builds GCP clients)
    and fires tiny LLM + TTS probes in a background thread so the first
    real `/api/tutor/start` or `/api/tutor/step` doesn't pay cold-start
    costs (TLS handshake, ADC refresh, model routing).
    """
    def _warm() -> None:
        try:
            pipeline = _get_voice_qa_pipeline()
        except Exception as exc:  # noqa: BLE001
            print(f"[tutor] warm-up skipped (pipeline init failed): {exc}", flush=True)
            return
        try:
            pipeline.warmup()
            print("[tutor] pipeline warmed up.", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[tutor] warm-up probe failed (non-fatal): {exc}", flush=True)

    threading.Thread(target=_warm, daemon=True, name="tutor-warmup").start()


def _flatten_tutor_chunks() -> list[dict]:
    """Mirror voice_qa_page._flatten_chunks but without the Gradio dependency."""
    dict_path = PROJECT_ROOT / "modules/teacher_module/outputs/chunk_summary_dictionary.json"
    if not dict_path.exists():
        return []
    with open(dict_path, encoding="utf-8") as f:
        data = json.load(f)
    out: list[dict] = []
    for page in data.get("pages", []):
        page_no = int(page.get("page_no", 0))
        for idx, chunk in enumerate(page.get("teacher_explanation_chunks", []), start=1):
            out.append({
                "page_no": page_no,
                "chunk_no": idx,
                "teacher_explanation": str(chunk.get("teacher_explanation", "")).strip(),
                "summary": str(chunk.get("summary", "")).strip(),
                "question": str(chunk.get("question", "")).strip(),
                "answer": str(chunk.get("answer", "")).strip(),
            })
    out.sort(key=lambda c: (c["page_no"], c["chunk_no"]))
    return out


def _tutor_asr_code(choice: str) -> str:
    val = (choice or "").strip().lower()
    if val.startswith("ta"):
        return "ta-IN"
    return "en-US"


def _build_tutor_segment(chunks: list[dict], start_idx: int) -> tuple[str, int, int | None]:
    """Identical semantics to voice_qa_page._build_narration_segment."""
    if start_idx >= len(chunks):
        return "", start_idx, None
    parts: list[str] = []
    i = start_idx
    question_idx: int | None = None
    while i < len(chunks):
        c = chunks[i]
        txt = (c.get("teacher_explanation") or "").strip()
        if txt:
            parts.append(txt)
        q = (c.get("question") or "").strip()
        if q:
            parts.append(f"Now answer this question: {q}")
            question_idx = i
            i += 1
            break
        i += 1
    return "\n\n".join(parts).strip(), i, question_idx


def _tutor_session_dir(session_id: str) -> Path:
    d = TUTOR_OUTPUT_ROOT / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tutor_audio_url(session_id: str, path: Path) -> str:
    return f"/api/tutor/audio/{session_id}/{path.name}"


def _synth(pipeline, text: str, session_id: str, label: str) -> Path:
    filename = f"{label}_{int(time.time() * 1000)}_{uuid4().hex[:6]}.wav"
    out_path = _tutor_session_dir(session_id) / filename
    pipeline.synthesize_answer(text, out_path)
    return out_path


# ── TTS cache for static phrases (greeting, canned replies) ──────────────
# The greeting line is the same on every `/api/tutor/start`; synthesising
# it from scratch costs 3-5s.  We hash (text, voice, model) and keep the WAV
# on disk under outputs/voice_qa/_cache so subsequent starts are ~100 ms.
import hashlib as _hashlib
_TTS_CACHE_DIR = PROJECT_ROOT / "outputs/voice_qa/_cache"


def _tts_cache_key(text: str, voice: str, model: str) -> str:
    h = _hashlib.sha256(f"{model}|{voice}|{text}".encode("utf-8")).hexdigest()[:20]
    return h


def _synth_cached(pipeline, text: str, session_id: str, label: str) -> Path:
    """Synthesise (or reuse) audio for a static phrase, then expose it
    under the per-session dir so the existing /api/tutor/audio route serves it."""
    text = (text or "").strip()
    if not text:
        return _synth(pipeline, "Let us continue.", session_id, label)
    voice = pipeline.config.tts_voice
    model = pipeline.config.tts_model
    key = _tts_cache_key(text, voice, model)
    _TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = _TTS_CACHE_DIR / f"{key}.wav"
    if not cached.exists():
        pipeline.synthesize_answer(text, cached)
    # Symlink / hardlink into the session dir so existing audio URL serving
    # still works untouched.  Fall back to copy if link syscalls are denied.
    filename = f"{label}_{int(time.time() * 1000)}_{uuid4().hex[:6]}.wav"
    dst = _tutor_session_dir(session_id) / filename
    try:
        import os as _os
        _os.link(cached, dst)
    except OSError:
        import shutil as _shutil
        _shutil.copy2(cached, dst)
    return dst


def _save_wav_upload(upload: UploadFile) -> Path:
    suffix = ".wav"
    if upload.filename and "." in upload.filename:
        suffix = "." + upload.filename.rsplit(".", 1)[-1].lower()
        if suffix not in (".wav", ".webm", ".ogg", ".mp3", ".m4a"):
            suffix = ".wav"
    tmp = tempfile.NamedTemporaryFile(prefix="tutor_ans_", suffix=suffix, delete=False)
    tmp.write(upload.file.read())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _ensure_wav_linear16(src: Path) -> Path:
    """Return a path to a mono-16 kHz-LINEAR16 WAV.

    Fast path: if the upload is already a valid LINEAR16 WAV we return the
    original file untouched — no ffmpeg spawn.  Slow path: invoke ffmpeg
    (~250-400 ms) to transcode WebM/Opus / OGG etc.
    """
    # First cheap probe — a RIFF header tells us it's a WAV.
    try:
        with open(src, "rb") as f:
            head = f.read(4)
    except Exception:
        return src

    if head == b"RIFF":
        # Second cheap probe — the built-in `wave` module tells us PCM /
        # channels / rate without decoding the payload.  If the file is
        # already mono / 16 kHz / 16-bit PCM we can hand it to ASR as-is.
        try:
            with wave.open(str(src), "rb") as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                sample_width = wf.getsampwidth()
            if channels == 1 and sample_rate == 16000 and sample_width == 2:
                return src
        except Exception:
            # Malformed WAV — fall through to ffmpeg for a clean remux.
            pass

    # Try ffmpeg conversion.
    import shutil as _shutil
    import subprocess as _sp
    ffmpeg = _shutil.which("ffmpeg")
    if not ffmpeg:
        return src
    dst = src.with_suffix(".converted.wav")
    try:
        _sp.run(
            [ffmpeg, "-y", "-i", str(src), "-ac", "1", "-ar", "16000",
             "-f", "wav", "-acodec", "pcm_s16le", str(dst)],
            check=True, capture_output=True,
        )
        return dst
    except Exception:
        return src


def _lesson_status(state: dict) -> str:
    total = int(state.get("total_chunks", 0))
    idx = int(state.get("idx", 0))
    mode = state.get("mode", "idle")
    paused = bool(state.get("is_paused", False))
    suffix = " | Paused" if paused else ""
    if mode == "waiting_for_greeting":
        return "Tap Ask / Answer, say hi, then tap Stop & Send."
    if mode == "done":
        return f"Lesson completed. Reviewed {total} segments."
    if idx < total:
        c = state["chunks"][idx]
        return (
            f"{mode}{suffix} · page {c['page_no']} chunk {c['chunk_no']} · "
            f"progress {idx + 1}/{total}"
        )
    return f"{mode}{suffix} · progress {idx}/{total}"


# Canned greeting replies — avoids a 3-5s Gemma round-trip on the first step.
# If you want the tutor to truly "listen" to the student's greeting, flip
# `_TUTOR_LLM_GREETING = True` and the code below falls back to the Gemma call.
_TUTOR_LLM_GREETING = False


# ── Segment pre-synthesis ─────────────────────────────────────────────────
# Every teaching segment's text comes from the deterministic
# chunk_summary_dictionary.json and is identical across students.  TTS is the
# single most expensive stage of a tutor turn (~3-7s), so we push it off the
# critical path by pre-synthesising every segment into the content-hashed
# _TTS_CACHE_DIR the moment a session starts.  Subsequent /api/tutor/step
# calls then resolve the segment audio as a near-instant hardlink lookup.
import concurrent.futures as _cf

_SEGMENT_EXECUTOR = _cf.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="tutor-presynth"
)


def _all_segment_texts(chunks: list[dict]) -> list[tuple[int, str]]:
    """Return the list of (start_idx, segment_text) the tutor will narrate.

    Must mirror `_build_tutor_segment` so pre-cached WAVs key on the same
    text the runtime will hash.
    """
    out: list[tuple[int, str]] = []
    idx = 0
    while idx < len(chunks):
        seg_text, next_idx, _q_idx = _build_tutor_segment(chunks, idx)
        if seg_text:
            out.append((idx, seg_text))
        if next_idx <= idx:
            break
        idx = next_idx
    return out


def _kickoff_cache_text(greet_reply: str, seg_text: str) -> str:
    """Replicates how /step builds the first-step kickoff utterance.

    Keeping this in one place means the pre-synth pass and the live call
    hash the exact same bytes.
    """
    greet_reply = (greet_reply or "").strip()
    seg_text = (seg_text or "").strip()
    return f"{greet_reply}\n\n{seg_text}".strip() if seg_text else greet_reply


def _spawn_segment_presynth(
    pipeline, chunks: list[dict], language: str
) -> None:
    """Fire-and-forget background pre-synth of every segment + kickoff WAV.

    Uses `_synth_cached` so the cached filename is the content hash of the
    text — the live `/step` path can then reuse the cached bytes with zero
    extra TTS cost.
    """

    def _worker() -> None:
        scratch_session = f"_presynth_{uuid4().hex[:8]}"
        segments = _all_segment_texts(chunks)
        try:
            # Canned greet reply + canned "correct answer" ack — both tiny
            # but on the critical path of the earliest /step calls, so
            # synthesise them before segments.
            greet_reply = (
                "நன்று! நாம் பாடத்தைத் தொடங்குவோம்."
                if (language or "").lower().startswith("ta")
                else "Lovely! Let us begin the lesson."
            )
            canned_phrases = [greet_reply, "Correct! Good job. Let us continue."]
            for phrase in canned_phrases:
                try:
                    _synth_cached(pipeline, phrase, scratch_session, "canned_pre")
                except Exception as exc:  # noqa: BLE001
                    print(f"[tutor] pre-synth canned skipped: {exc}", flush=True)

            # Segment texts (identical across students) → cache once, reuse forever.
            for _idx, seg_text in segments:
                try:
                    _synth_cached(pipeline, seg_text, scratch_session, "segment_pre")
                except Exception as exc:  # noqa: BLE001
                    print(f"[tutor] pre-synth segment skipped: {exc}", flush=True)

            print(
                f"[tutor] pre-synth complete: "
                f"{len(segments)} segments + 1 greet cached.",
                flush=True,
            )
        finally:
            # Clean scratch session dir — we only cared about the shared cache.
            try:
                import shutil as _shutil
                _shutil.rmtree(_tutor_session_dir(scratch_session), ignore_errors=True)
            except Exception:
                pass

    _SEGMENT_EXECUTOR.submit(_worker)


def _build_greeting_reply(*, pipeline, student_text: str, language: str, chapter: str) -> str:
    if _TUTOR_LLM_GREETING:
        prompt = (
            "You are a warm and friendly AI tutor greeting a student before a lesson.\n\n"
            f"Student said: {student_text}\n"
            f"Chapter to teach: {chapter}\n"
            f"Preferred language: {language}\n\n"
            "Rules:\n"
            "1) Reply naturally in 2-3 short sentences.\n"
            "2) Show empathy.\n"
            "3) Invite them into the lesson.\n"
            "4) Do NOT repeat their sentence verbatim.\n"
            "5) Plain text only."
        )
        try:
            out = (pipeline.endpoint_client.generate_text(prompt, max_new_tokens=120) or "").strip()
        except Exception:
            out = ""
        if out:
            return out
    # Fast path — fixed copy; TTS result is cached so this adds ~0 ms.
    if (language or "").lower().startswith("ta"):
        return "நன்று! நாம் பாடத்தைத் தொடங்குவோம்."
    return "Lovely! Let us begin the lesson."


@app.post("/api/tutor/start")
async def tutor_start(request: Request):
    """Kick off a new tutor session: synthesise greeting + set up state."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    language = str(body.get("language") or "English").strip() or "English"
    asr_language = str(body.get("asr_language") or "English").strip() or "English"
    chapter = str(body.get("chapter") or "Electrostatics").strip() or "Electrostatics"

    chunks = _flatten_tutor_chunks()
    if not chunks:
        raise HTTPException(status_code=500, detail="No lesson chunks available.")

    try:
        pipeline = _get_voice_qa_pipeline()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Tutor pipeline unavailable: {exc}")

    session_id = f"{int(time.time() * 1000)}_{uuid4().hex[:8]}"
    state = {
        "session_id": session_id,
        "language": language,
        "asr_language": asr_language,
        "chapter": chapter,
        "chunks": chunks,
        "total_chunks": len(chunks),
        "idx": 0,
        "mode": "waiting_for_greeting",
        "is_paused": False,
        "current_idx": None,
        "struggled": [],
        "last_segment_start_idx": 0,
    }

    greeting_line = (
        "வணக்கம்! இன்று எப்படி இருக்கிறீர்கள்?"
        if (language or "").lower().startswith("ta")
        else "Hi! How are you today?"
    )
    try:
        greet_path = _synth_cached(pipeline, greeting_line, session_id, "greeting")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"TTS failed: {exc}")

    with _tutor_lock:
        _tutor_sessions[session_id] = state

    # Kick off a background pass that synthesises every segment + kickoff
    # utterance into the shared TTS cache.  By the time the student finishes
    # the greeting turn, most segments are already on disk and subsequent
    # /step calls resolve in milliseconds.
    _spawn_segment_presynth(pipeline, chunks, language=language)

    return {
        "session_id": session_id,
        "mode": state["mode"],
        "status": "Greeting is ready. Tap Ask / Answer, say hi, then tap Stop & Send.",
        "audio_url": _tutor_audio_url(session_id, greet_path),
        "lesson_status": _lesson_status(state),
    }


def _require_session(session_id: str) -> dict:
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    with _tutor_lock:
        state = _tutor_sessions.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Unknown session_id — start a new lesson.")
    return state


@app.post("/api/tutor/step")
async def tutor_step(
    session_id: Annotated[str, Form()],
    audio: Annotated[UploadFile | None, File()] = None,
):
    """Run one tutor step: optional student audio → TTS audio + next state."""
    state = _require_session(session_id)
    if state.get("is_paused"):
        return {
            "session_id": session_id,
            "mode": state.get("mode"),
            "status": "Lesson paused. Resume to continue.",
            "audio_url": None,
            "lesson_status": _lesson_status(state),
        }

    chunks: list[dict] = state["chunks"]
    total: int = state["total_chunks"]
    idx: int = int(state.get("idx", 0))
    mode: str = state.get("mode", "idle")

    if mode == "done" or idx >= total:
        state["mode"] = "done"
        return {
            "session_id": session_id,
            "mode": "done",
            "status": "Lesson completed.",
            "audio_url": None,
            "lesson_status": _lesson_status(state),
        }

    try:
        pipeline = _get_voice_qa_pipeline()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Tutor pipeline unavailable: {exc}")

    # Save student audio if supplied.
    student_wav: Path | None = None
    student_text = ""
    if audio and audio.filename:
        try:
            student_wav = _ensure_wav_linear16(_save_wav_upload(audio))
            student_text = pipeline.transcribe_audio(
                student_wav,
                language_code=_tutor_asr_code(state.get("asr_language", "English")),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not transcribe: {exc}")

    try:
        if mode == "waiting_for_greeting":
            if not student_text:
                return {
                    "session_id": session_id,
                    "mode": mode,
                    "status": "Please answer the greeting using Ask / Answer.",
                    "student_text": "",
                    "audio_url": None,
                    "lesson_status": _lesson_status(state),
                }

            greet_reply = _build_greeting_reply(
                pipeline=pipeline,
                student_text=student_text,
                language=state.get("language", "English"),
                chapter=state.get("chapter", "this chapter"),
            )
            seg_text, next_idx, q_idx = _build_tutor_segment(chunks, idx)
            state["last_segment_start_idx"] = idx
            # Split kickoff into two clips so the browser can play the short
            # greeting line immediately and chain the (possibly long) first
            # teaching segment.  Both lookups are content-hashed → hit the
            # cache populated at session start.
            greet_audio = _synth_cached(pipeline, greet_reply, session_id, "greet")
            segment_audio = (
                _synth_cached(pipeline, seg_text, session_id, "segment")
                if seg_text
                else None
            )

            if q_idx is not None:
                state.update(mode="waiting_for_answer", current_idx=q_idx, idx=q_idx)
            else:
                state.update(mode="narrating", idx=next_idx)
                if state["idx"] >= total:
                    state["mode"] = "done"
            return {
                "session_id": session_id,
                "mode": state["mode"],
                "status": "Greeting done. Lesson kicked off.",
                "student_text": student_text,
                "audio_url": _tutor_audio_url(session_id, greet_audio),
                "segment_audio_url": (
                    _tutor_audio_url(session_id, segment_audio)
                    if segment_audio is not None
                    else None
                ),
                "lesson_status": _lesson_status(state),
            }

        if mode == "narrating":
            # If student pressed Send during narration, answer their free-form doubt.
            if student_text:
                context = pipeline.retrieve_context(student_text)
                reply = pipeline.answer_question(
                    question=student_text,
                    language=state.get("language", "English"),
                    context_chunks=context,
                )
                audio_path = _synth(pipeline, reply, session_id, "askreply")
                return {
                    "session_id": session_id,
                    "mode": "narrating",
                    "status": f"You asked: {student_text}",
                    "student_text": student_text,
                    "audio_url": _tutor_audio_url(session_id, audio_path),
                    "lesson_status": _lesson_status(state),
                }

            seg_text, next_idx, q_idx = _build_tutor_segment(chunks, idx)
            state["last_segment_start_idx"] = idx
            if not seg_text:
                state["mode"] = "done"
                return {
                    "session_id": session_id,
                    "mode": "done",
                    "status": "Lesson completed.",
                    "audio_url": None,
                    "lesson_status": _lesson_status(state),
                }
            # Segment text is deterministic → content-hashed cache lookup.
            audio_path = _synth_cached(pipeline, seg_text, session_id, "segment")
            if q_idx is not None:
                state.update(mode="waiting_for_answer", current_idx=q_idx, idx=q_idx)
                return {
                    "session_id": session_id,
                    "mode": "waiting_for_answer",
                    "status": "Question asked — tap Ask / Answer to reply.",
                    "audio_url": _tutor_audio_url(session_id, audio_path),
                    "lesson_status": _lesson_status(state),
                }
            state.update(mode="narrating", idx=next_idx)
            if state["idx"] >= total:
                state["mode"] = "done"
            return {
                "session_id": session_id,
                "mode": state["mode"],
                "status": "Segment narrated.",
                "audio_url": _tutor_audio_url(session_id, audio_path),
                "lesson_status": _lesson_status(state),
            }

        if mode == "waiting_for_answer":
            if not student_text:
                return {
                    "session_id": session_id,
                    "mode": mode,
                    "status": "No answer heard. Tap Ask / Answer and try again.",
                    "student_text": "",
                    "audio_url": None,
                    "lesson_status": _lesson_status(state),
                }
            current_idx = int(state.get("current_idx") or idx)
            chunk = chunks[current_idx]
            evaluation = pipeline.evaluate_student_answer(
                question=chunk.get("question", ""),
                expected_answer=chunk.get("answer", ""),
                student_answer=student_text,
                context_text=chunk.get("teacher_explanation", ""),
                language=state.get("language", "English"),
            )
            is_correct = bool(evaluation.get("is_correct", False))
            feedback = (evaluation.get("feedback") or "").strip() or "Thanks for your answer."
            # On the correct path the LLM's feedback line rarely adds
            # anything the student doesn't already know, so we skip live
            # TTS of it and fall back to a cached canned ack — the
            # feedback string itself still appears in the chat bubble
            # (via `evaluation.feedback` in the JSON response).  This
            # turns the "correct answer" round-trip from ~13 s to ~5 s.
            # On the wrong path we DO TTS the LLM explanation because it
            # carries the actual correction — no cheap substitute there.
            if is_correct:
                feedback_text = "Correct! Good job. Let us continue."
            else:
                feedback_text = (
                    f"{feedback} The correct answer is: {chunk.get('answer', '')}. "
                    "Let us move to the next concept."
                )
                state["struggled"].append({
                    "page_no": chunk["page_no"],
                    "chunk_no": chunk["chunk_no"],
                    "question": chunk.get("question", ""),
                    "student_answer": student_text,
                })
            state["mode"] = "narrating"
            state["idx"] = current_idx + 1
            state["current_idx"] = None

            next_idx = int(state["idx"])
            if next_idx >= total:
                state["mode"] = "done"
                final_text = f"{feedback_text} Great job! You completed the lesson."
                audio_path = _synth(pipeline, final_text, session_id, "final")
                return {
                    "session_id": session_id,
                    "mode": "done",
                    "status": "Lesson completed.",
                    "student_text": student_text,
                    "evaluation": {"is_correct": is_correct, "feedback": feedback},
                    "audio_url": _tutor_audio_url(session_id, audio_path),
                    "lesson_status": _lesson_status(state),
                }

            seg_text, seg_next_idx, seg_q_idx = _build_tutor_segment(chunks, next_idx)
            state["last_segment_start_idx"] = next_idx

            # ── Split audio into (feedback | segment) so the client plays the
            # short feedback line immediately and chains the segment in the
            # background — the segment WAV is almost always a cache hit
            # because of the background pre-synth kicked off at session start.
            # For the correct path the feedback is ALSO a cache hit (canned
            # ack), collapsing the whole turn to cached-only audio.
            feedback_audio = _synth_cached(
                pipeline, feedback_text, session_id, "feedback"
            )
            segment_audio: Path | None = None
            if seg_text:
                segment_audio = _synth_cached(
                    pipeline, seg_text, session_id, "segment"
                )

            if seg_q_idx is not None:
                state.update(mode="waiting_for_answer", current_idx=seg_q_idx, idx=seg_q_idx)
            else:
                state.update(mode="narrating", idx=seg_next_idx)
                if state["idx"] >= total:
                    state["mode"] = "done"
            return {
                "session_id": session_id,
                "mode": state["mode"],
                "status": "Feedback + next segment delivered.",
                "student_text": student_text,
                "evaluation": {"is_correct": is_correct, "feedback": feedback},
                "audio_url": _tutor_audio_url(session_id, feedback_audio),
                "segment_audio_url": (
                    _tutor_audio_url(session_id, segment_audio)
                    if segment_audio is not None
                    else None
                ),
                "lesson_status": _lesson_status(state),
            }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Tutor step failed: {exc}")

    return {
        "session_id": session_id,
        "mode": state.get("mode", "idle"),
        "status": "Nothing to do.",
        "audio_url": None,
        "lesson_status": _lesson_status(state),
    }


@app.post("/api/tutor/pause")
async def tutor_pause(request: Request):
    body = await request.json()
    state = _require_session(str(body.get("session_id") or ""))
    paused = not bool(state.get("is_paused", False))
    state["is_paused"] = paused
    return {
        "session_id": state["session_id"],
        "paused": paused,
        "mode": state.get("mode"),
        "lesson_status": _lesson_status(state),
    }


@app.get("/api/tutor/audio/{session_id}/{filename}")
async def tutor_audio(session_id: str, filename: str):
    # Path-traversal guard: reject any funny business in either segment.
    if "/" in session_id or ".." in session_id or "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Bad path")
    path = TUTOR_OUTPUT_ROOT / session_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path, media_type="audio/wav", filename=filename)


# ── Resource feedback API (must precede the catch-all proxy below) ────────
@app.post("/api/resource/feedback")
async def resource_feedback(
    request: Request,
    aitutor_sid: Annotated[str | None, Cookie()] = None,
):
    """Record a student's like / dislike and return updated counts + rating."""
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")

    chapter = str(payload.get("chapter") or "").strip()
    url = str(payload.get("url") or "").strip()
    action = str(payload.get("action") or "").strip().lower()
    if not chapter or not url:
        raise HTTPException(status_code=400, detail="`chapter` and `url` are required")
    if action not in ("like", "dislike"):
        raise HTTPException(status_code=400, detail="`action` must be 'like' or 'dislike'")
    if not aitutor_sid:
        raise HTTPException(status_code=401, detail="Sign in before voting.")

    fb = _load_feedback()
    entry = _entry_for(fb, chapter, url)
    voters = entry.setdefault("voters", {})
    previous = voters.get(aitutor_sid, "")

    if previous == action:
        # Clicking the same vote again → undo it.
        voters.pop(aitutor_sid, None)
        if action == "like":
            entry["likes"] = max(0, int(entry.get("likes", 0)) - 1)
        else:
            entry["dislikes"] = max(0, int(entry.get("dislikes", 0)) - 1)
        new_vote = ""
    else:
        # Switch vote or first vote.
        if previous == "like":
            entry["likes"] = max(0, int(entry.get("likes", 0)) - 1)
        elif previous == "dislike":
            entry["dislikes"] = max(0, int(entry.get("dislikes", 0)) - 1)
        if action == "like":
            entry["likes"] = int(entry.get("likes", 0)) + 1
        else:
            entry["dislikes"] = int(entry.get("dislikes", 0)) + 1
        voters[aitutor_sid] = action
        new_vote = action

    _save_feedback(fb)

    # Curator score for rating math — look it up from the curated file.
    curator_score = 0.0
    raw = _load_resources_raw()
    for r in raw.get(chapter, {}).get("resources", []):
        if r.get("url") == url:
            curator_score = float(r.get("score", 0) or 0)
            break

    return {
        "chapter": chapter,
        "url": url,
        "likes": entry["likes"],
        "dislikes": entry["dislikes"],
        "my_vote": new_vote,
        "rating": _rating(curator_score, entry["likes"], entry["dislikes"]),
    }


# ── Learn API proxy ────────────────────────────────────────────────────────
# The browser talks only to this server (port 3000). /api/* is proxied to the
# FastAPI learn_api service (port 8000) so we avoid CORS and keep a clean UX.
@app.api_route(
    "/api/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def learn_api_proxy(path: str, request: Request):
    target_url = f"{LEARN_API_URL}/{path}"
    # forward query string
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    # forward method, headers, body — strip host header so upstream picks its own
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    body = await request.body()

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            upstream = await client.request(
                method=request.method,
                url=target_url,
                headers=fwd_headers,
                content=body,
            )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream learn_api unreachable: {exc}") from exc

    # strip hop-by-hop / length headers that Starlette will re-compute
    excluded = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    passthrough_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in excluded}
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=passthrough_headers,
        media_type=upstream.headers.get("content-type"),
    )


@app.get("/app")
async def goto_app():
    return RedirectResponse(url=GRADIO_URL, status_code=status.HTTP_302_FOUND)


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8080")),
        reload=True,
    )
