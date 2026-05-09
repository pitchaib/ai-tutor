"""FastAPI backend for Learn UI page-wise tutor flow."""

from __future__ import annotations

import os
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(os.getenv("AITUTOR_ROOT", "/home/bp/AiTutor"))
DEFAULT_PDF_PATH = Path(
    os.getenv(
        "PDF_PATH",
        "/home/bp/AiTutor/Board/State/Tamil_Nadu/English/Physics/books/"
        "Class_12_Physics_English_Volume_1_2024_Edition-www.tntextbooks.in.pdf",
    )
)
SESSION_JSON = Path(
    os.getenv(
        "SESSION_JSON",
        str(PROJECT_ROOT / "modules/teacher_module/outputs/teaching_session.json"),
    )
)
TEACHER_CACHE_JSON = Path(
    os.getenv(
        "TEACHER_CACHE_JSON",
        str(PROJECT_ROOT / "Board/State/Tamil_Nadu/English/Physics/chapter_page_cache.json"),
    )
)
MCQ_CACHE_JSON = Path(
    os.getenv(
        "MCQ_CACHE_JSON",
        str(PROJECT_ROOT / "Board/State/Tamil_Nadu/English/Physics/chapter_page_mcq_cache.json"),
    )
)
DEFAULT_PAGE_NO = int(os.getenv("DEFAULT_PAGE_NO", "1"))
# Vertex endpoint config comes from configs/vertex.env (sourced by start.sh)
# or from environment variables exported by the caller.
# No hardcoded endpoints — rotating the endpoint only requires editing vertex.env.
VERTEX_ENDPOINT_URL = (os.getenv("VERTEX_ENDPOINT_URL") or "").strip()
VERTEX_API_ENDPOINT = (os.getenv("VERTEX_API_ENDPOINT") or "").strip() or None
VERTEX_PROJECT_ID = (os.getenv("VERTEX_PROJECT_ID") or "").strip() or None
VERTEX_LOCATION = (os.getenv("VERTEX_LOCATION") or "").strip() or None
VERTEX_ENDPOINT_ID = (os.getenv("VERTEX_ENDPOINT_ID") or "").strip() or None


# Reuse existing teacher pipeline directly.
import sys

TEACHER_SRC = PROJECT_ROOT / "modules/teacher_module/src"
if str(TEACHER_SRC) not in sys.path:
    sys.path.insert(0, str(TEACHER_SRC))
ASSESS_SRC = PROJECT_ROOT / "modules/assessment_module/src"
if str(ASSESS_SRC) not in sys.path:
    sys.path.insert(0, str(ASSESS_SRC))

from teacher_pdf_pipeline import (  # noqa: E402
    explain_session_section,
    load_session,
    load_vertex_endpoint_client,
    plan_page_session,
    save_session,
)
from assessment_mcq_pipeline import get_or_generate_chapter_mcqs  # noqa: E402
import fitz  # noqa: E402


class PageRequest(BaseModel):
    chapter_name: str = Field(..., min_length=1)
    page_no: int = Field(..., ge=1)
    max_new_tokens: int = Field(2048, ge=128, le=8192)


class NavigateRequest(BaseModel):
    chapter_name: str = Field(..., min_length=1)
    current_page: int = Field(..., ge=1)
    action: Literal["previous", "next"]
    max_new_tokens: int = Field(2048, ge=128, le=8192)


class LearnResponse(BaseModel):
    chapter_name: str
    page_no: int
    section_id: int
    title: str
    content: str
    teacher_explanation: str
    teacher_explanation_tamil: str
    teacher_explanation_audio_english_path: str
    teacher_explanation_audio_tamil_path: str
    status: str
    source: Literal["cache", "generated"]


# Selection-based contextual chat (textbook page) --------------------------
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=4000)


class TextbookChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    selected_text: str | None = Field(default=None, max_length=6000)
    full_context: str = Field(default="", max_length=20000)
    chapter_name: str = Field(default="", max_length=200)
    page_no: int | None = None
    language: Literal["English", "Tamil", "Mix"] = "English"
    history: list[ChatTurn] = Field(default_factory=list)
    max_new_tokens: int = Field(default=600, ge=64, le=2048)


class TextbookChatResponse(BaseModel):
    answer: str
    used_selection: bool
    # "on_page"      → fully grounded in selection or full page context
    # "beyond_page"  → related topic, needs some general knowledge to answer
    # "off_topic"    → unrelated to the page; polite redirect with a short reply
    # "unknown"      → model didn't emit a scope marker (fallback)
    source: Literal["on_page", "beyond_page", "off_topic", "unknown"] = "unknown"
    model: str = "gemma-via-vertex"


app = FastAPI(title="AI Personal Tutor Learn API", version="1.0.0")


def _ensure_session() -> dict[str, Any]:
    if SESSION_JSON.exists():
        session = load_session(SESSION_JSON)
        if isinstance(session, dict) and isinstance(session.get("chapters"), list):
            return session
    return {"pdf_path": str(DEFAULT_PDF_PATH), "chapters": [{"sections": []}]}


def _save_session(session: dict[str, Any]) -> None:
    save_session(session, SESSION_JSON)


def _normalize_sections(session: dict[str, Any]) -> list[dict[str, Any]]:
    chapters = session.setdefault("chapters", [])
    if not chapters:
        chapters.append({"sections": []})
    ch0 = chapters[0]
    secs = ch0.setdefault("sections", [])
    secs.sort(key=lambda s: (int(s.get("page_no", 0)), int(s.get("section_id", 0))))
    for idx, sec in enumerate(secs, start=1):
        sec["section_id"] = idx
    return secs


def _find_section_by_page(session: dict[str, Any], page_no: int) -> dict[str, Any] | None:
    secs = _normalize_sections(session)
    for sec in secs:
        if int(sec.get("page_no", 0)) == page_no:
            return sec
    return None


def _merge_page_session(session: dict[str, Any], page_session: dict[str, Any]) -> dict[str, Any]:
    target_secs = _normalize_sections(session)
    new_secs = page_session.get("chapters", [{}])[0].get("sections", [])
    if not new_secs:
        return session

    new_page = int(new_secs[0].get("page_no", 0))
    target_secs[:] = [s for s in target_secs if int(s.get("page_no", 0)) != new_page]
    target_secs.extend(new_secs)
    _normalize_sections(session)
    return session


@lru_cache(maxsize=1)
def _get_endpoint_client():
    # Prefer the full console URL (easy copy/paste from the Vertex UI).
    if VERTEX_ENDPOINT_URL:
        return load_vertex_endpoint_client(
            endpoint_url=VERTEX_ENDPOINT_URL,
            api_endpoint=VERTEX_API_ENDPOINT,
        )
    # Fall back to split VERTEX_PROJECT_ID / VERTEX_LOCATION / VERTEX_ENDPOINT_ID.
    if VERTEX_PROJECT_ID and VERTEX_LOCATION and VERTEX_ENDPOINT_ID:
        return load_vertex_endpoint_client(
            project_id=VERTEX_PROJECT_ID,
            location=VERTEX_LOCATION,
            endpoint_id=VERTEX_ENDPOINT_ID,
            api_endpoint=VERTEX_API_ENDPOINT,
        )
    raise RuntimeError(
        "Vertex endpoint config missing. Populate configs/vertex.env with "
        "VERTEX_ENDPOINT_URL (or VERTEX_PROJECT_ID + VERTEX_LOCATION + "
        "VERTEX_ENDPOINT_ID) and VERTEX_API_ENDPOINT (for private endpoints), "
        "then run `./start.sh restart`."
    )


def _build_response(
    *,
    chapter_name: str,
    sec: dict[str, Any],
    source: Literal["cache", "generated"],
) -> LearnResponse:
    return LearnResponse(
        chapter_name=chapter_name,
        page_no=int(sec.get("page_no", 0)),
        section_id=int(sec.get("section_id", 0)),
        title=str(sec.get("title", "")),
        content=str(sec.get("content", "")),
        teacher_explanation=str(sec.get("teacher_explanation", "")),
        teacher_explanation_tamil=str(sec.get("teacher_explanation_tamil", "")),
        teacher_explanation_audio_english_path=str(
            sec.get("teacher_explanation_audio_english_path", "")
        ),
        teacher_explanation_audio_tamil_path=str(
            sec.get("teacher_explanation_audio_tamil_path", "")
        ),
        status=str(sec.get("status", "")),
        source=source,
    )


def _ensure_page_explanation(
    *,
    chapter_name: str,
    page_no: int,
    max_new_tokens: int,
) -> LearnResponse:
    session = _ensure_session()
    sec = _find_section_by_page(session, page_no)

    if sec and (sec.get("teacher_explanation") or "").strip():
        return _build_response(chapter_name=chapter_name, sec=sec, source="cache")

    try:
        client = _get_endpoint_client()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"LLM endpoint is not configured: {e}") from e

    if sec is None:
        page_session = plan_page_session(
            DEFAULT_PDF_PATH,
            chapter_name=chapter_name,
            page_no=page_no,
            endpoint_client=client,
            use_llm_page_segmentation=True,
        )
        session = _merge_page_session(session, page_session)
        _save_session(session)
        sec = _find_section_by_page(session, page_no)
        if sec is None:
            raise HTTPException(status_code=500, detail="Planned page section not found after merge.")

    # Explain (or re-explain) the section for this page.
    updated = explain_session_section(
        session,
        chapter_index=0,
        section_id=int(sec["section_id"]),
        endpoint_client=client,
        max_new_tokens=max_new_tokens,
    )
    _save_session(session)
    return _build_response(chapter_name=chapter_name, sec=updated, source="generated")


def _default_start_page() -> int:
    # Prefer chapter-specific start page from chapter dictionary cache.
    # Kept for backward compatibility callers without chapter name.
    return _default_start_page_for_chapter("Electrostatics")


def _default_start_page_for_chapter(chapter_name: str) -> int:
    chapter_key = (chapter_name or "").strip()
    if TEACHER_CACHE_JSON.exists():
        try:
            with open(TEACHER_CACHE_JSON, encoding="utf-8") as f:
                cache = json.load(f)
            pages = cache.get(chapter_key, {}).get("pages", {})
            page_nos = sorted(int(k) for k in pages.keys() if str(k).isdigit() and int(k) > 0)
            if page_nos:
                return page_nos[0]
        except Exception:
            pass

    # Fallback to session-known pages.
    session = _ensure_session()
    secs = _normalize_sections(session)
    if secs:
        page_nos = [int(s.get("page_no", 0)) for s in secs if int(s.get("page_no", 0)) > 0]
        if page_nos:
            return min(page_nos)
    return max(1, DEFAULT_PAGE_NO)


def _quiz_page_numbers(chapter_name: str, current_page: int) -> list[int]:
    start = _default_start_page_for_chapter(chapter_name)
    end = max(start, int(current_page))
    return list(range(start, end + 1))


def _get_or_generate_quiz_mcqs(
    *,
    chapter_name: str,
    current_page: int,
    questions_per_page: int = 3,
) -> list[dict[str, Any]]:
    result = get_or_generate_chapter_mcqs(
        chapter_name=chapter_name,
        teacher_cache_path=TEACHER_CACHE_JSON,
        mcq_cache_path=MCQ_CACHE_JSON,
        endpoint_client=_get_endpoint_client(),
        page_nos=_quiz_page_numbers(chapter_name, current_page),
        questions_per_page=questions_per_page,
        max_new_tokens=2800,
        force_regenerate=False,
    )
    out: list[dict[str, Any]] = []
    for page in result.get("pages", []):
        for m in page.get("mcqs", []):
            if isinstance(m, dict):
                out.append(m)
    return out


def _quiz_html(chapter_name: str, current_page: int, mcqs: list[dict[str, Any]]) -> str:
    if not mcqs:
        return (
            "<h2>No questions available</h2>"
            "<p>Teacher explanations may be missing for this chapter/page range.</p>"
        )
    items = []
    for idx, q in enumerate(mcqs, start=1):
        qid = str(q.get("question_id", f"q{idx}"))
        question = str(q.get("question", ""))
        options = q.get("options", {})
        options_html = []
        for key in ("A", "B", "C", "D"):
            val = str(options.get(key, ""))
            options_html.append(
                f"<label><input type='radio' name='{qid}' value='{key}' required> "
                f"<strong>{key}</strong>) {val}</label><br/>"
            )
        items.append(
            f"<div style='padding:14px;border:1px solid #dbe2ef;border-radius:10px;margin-bottom:12px;'>"
            f"<p><strong>Q{idx}.</strong> {question}</p>{''.join(options_html)}</div>"
        )
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Test My Understanding</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#f5f8ff; margin:0; }}
    .wrap {{ max-width:980px; margin:24px auto; background:#fff; padding:20px; border-radius:12px; box-shadow:0 8px 24px rgba(0,0,0,.08); }}
    .top {{ margin-bottom:16px; }}
    .btn {{ background:#1f6feb; color:#fff; border:none; padding:10px 18px; border-radius:8px; font-weight:700; cursor:pointer; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <h2>🧠 Test My Understanding</h2>
      <p>Chapter: <strong>{chapter_name}</strong> | Pages: <strong>{_quiz_page_numbers(chapter_name, current_page)[0]} - {current_page}</strong></p>
      <p>Total Questions: <strong>{len(mcqs)}</strong></p>
    </div>
    <!-- Relative action keeps the form working whether the page is served
         directly (learn_api at /quiz) OR via the HTML-frontend proxy at
         /api/quiz.  "/quiz/submit" would 404 through the proxy because the
         frontend only forwards /api/* to learn_api. -->
    <form method="post" action="quiz/submit">
      <input type="hidden" name="chapter_name" value="{chapter_name}"/>
      <input type="hidden" name="current_page" value="{current_page}"/>
      {''.join(items)}
      <button class="btn" type="submit">Submit Answers</button>
    </form>
  </div>
</body>
</html>
"""


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/learn/init")
def learn_init(chapter_name: str = "Electrostatics", max_new_tokens: int = 2048) -> LearnResponse:
    start_page = _default_start_page_for_chapter(chapter_name)
    return _ensure_page_explanation(
        chapter_name=chapter_name,
        page_no=start_page,
        max_new_tokens=max_new_tokens,
    )


@app.post("/learn/page")
def learn_page(req: PageRequest) -> LearnResponse:
    return _ensure_page_explanation(
        chapter_name=req.chapter_name.strip(),
        page_no=req.page_no,
        max_new_tokens=req.max_new_tokens,
    )


@app.post("/learn/navigate")
def learn_navigate(req: NavigateRequest) -> LearnResponse:
    step = -1 if req.action == "previous" else 1
    page_no = max(1, req.current_page + step)
    return _ensure_page_explanation(
        chapter_name=req.chapter_name.strip(),
        page_no=page_no,
        max_new_tokens=req.max_new_tokens,
    )


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    head = text[: limit - 40].rsplit(" ", 1)[0]
    return head + "\n\n…[truncated for length]"


def _render_history(history: list[ChatTurn], limit: int = 6) -> str:
    turns = history[-limit:]
    if not turns:
        return "(no prior turns)"
    lines = []
    for t in turns:
        who = "Student" if t.role == "user" else "Tutor"
        lines.append(f"{who}: {t.content.strip()}")
    return "\n".join(lines)


def _build_textbook_chat_prompt(req: TextbookChatRequest) -> str:
    selected = (req.selected_text or "").strip()
    used_sel = bool(selected)
    sel_block = (
        f"Selected excerpt (the student highlighted this — this is your FIRST source):\n"
        f"\"\"\"\n{_truncate(selected, 4000)}\n\"\"\""
        if used_sel
        else "Selected excerpt: (none — the student did not highlight any text; "
        "use the full page context as your primary source)"
    )
    full_ctx = _truncate(req.full_context, 12000)
    chapter = (req.chapter_name or "Unknown chapter").strip()
    page = req.page_no if req.page_no and req.page_no > 0 else "—"
    language = req.language

    return f"""You are an AI tutor helping a school student understand a textbook page.
Your job is to answer the student's question while staying faithful to the page they are reading.

# Chapter
{chapter}

# Current page
{page}

# Full page context
\"\"\"
{full_ctx if full_ctx else "(no full context supplied)"}
\"\"\"

# {sel_block}

# Prior conversation (most recent last)
{_render_history(req.history)}

# New student question
{req.question.strip()}

# How to decide where the answer comes from
Classify the question into ONE of three scopes and then answer it PROPERLY —
do NOT refuse or punt. Picking a scope is only about WHERE the facts come
from; either way the student must get a real, useful answer.

  A. on_page      → The selected excerpt OR the full page context literally
                    contains the information needed. Answer using ONLY that
                    supplied text. Pick this only when you can truly answer
                    the question from the page — not merely because the
                    question is on-topic.

  B. beyond_page  → The question is ON THE SAME SUBJECT as this chapter /
                    page (same concept family, e.g. another aspect of
                    electric charge, Coulomb's law, capacitors, etc.) but
                    the page does NOT literally contain the answer. Use
                    general physics knowledge to answer it anyway. Rules:
                      • Be factually correct; no invented numbers or names.
                      • Keep it school-level appropriate.
                      • Start with ONE sentence tying the question to what
                        IS on this page, then give the fuller answer.
                      • It is NOT acceptable to say "the page does not
                        explain this" and stop — you MUST actually answer.

  C. off_topic   → The question has NOTHING to do with this chapter
                    (sports, movies, unrelated school subjects, etc.).
                    Politely say so in 1–2 sentences and suggest asking a
                    question about the current chapter. No general answer.

Worked examples (for a page about electric charge, units and quantisation):
  Q: "What is the unit of electric charge?"            → on_page
     (the page literally says "SI unit of charge is the coulomb")
  Q: "Why do like charges repel?"                      → beyond_page
     (page only states that they repel, never the reason)
  Q: "How is charge measured in a lab?"                → beyond_page
     (page gives the unit, but says nothing about instruments/procedure —
      "unit of charge" and "how charge is measured" are DIFFERENT questions,
      so the unit alone is NOT an answer)
  Q: "What is Coulomb's law formula?"                  → beyond_page
     (on the same topic, but the formula itself isn't on this page; quote
      the standard school-physics form F = k·q₁·q₂ / r² clearly)
  Q: "Who won the 2011 cricket world cup?"             → off_topic

# Output format (STRICT)
Line 1 must be EXACTLY one of these tokens (no quotes, no extra words):
    [SOURCE:on_page]
    [SOURCE:beyond_page]
    [SOURCE:off_topic]
Line 2 must be blank.
From Line 3 onward write the answer in Markdown.

# Answer-writing rules
1. Keep the reply tight: 2–4 short paragraphs OR a small Markdown list.
   Use **bold** for key terms.
2. For scope A (on_page): ground every sentence in the supplied context.
3. For scope B (beyond_page): one linking sentence to the page, then a
   COMPLETE school-level answer. Never end with "the page does not say".
4. For scope C (off_topic): at most 2 sentences, redirect the student.
5. Standard school-physics facts (Coulomb's law, Ohm's law, common SI
   constants, definitions, etc.) ARE acceptable to state in beyond_page
   answers — do NOT self-censor them. What's forbidden is INVENTING facts:
   unknown author names, made-up constants, or fictional experiments. When
   a precise figure is genuinely not known, describe the concept
   qualitatively rather than making a number up.
6. Reply in {language}. If "Mix", use English with Tamil for key terms in
   parentheses.
7. Do NOT repeat the question back verbatim.
8. Output Markdown only (no HTML, no code fences around the full answer).

# Your answer (remember: line 1 is the [SOURCE:...] tag):"""


_SOURCE_TAG_RE = re.compile(
    r"^\s*\[?\s*SOURCE\s*[:=]\s*(on_page|beyond_page|off_topic)\s*\]?\s*$",
    re.IGNORECASE,
)


def _split_source_tag(raw: str) -> tuple[str, Literal["on_page", "beyond_page", "off_topic", "unknown"]]:
    """Pull the `[SOURCE:xxx]` first-line marker out of the model reply.

    Returns (cleaned_answer, source). Falls back to ``unknown`` if the
    marker is missing or malformed so we never crash on a rogue reply.
    """
    if not raw:
        return "", "unknown"
    lines = raw.splitlines()
    # Find the first non-empty line — Gemma sometimes adds a leading blank.
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        return raw.strip(), "unknown"

    m = _SOURCE_TAG_RE.match(lines[idx])
    if not m:
        return raw.strip(), "unknown"

    scope = m.group(1).lower()
    # Drop the marker line AND a single blank line immediately after, if any.
    rest = lines[idx + 1 :]
    if rest and not rest[0].strip():
        rest = rest[1:]
    cleaned = "\n".join(rest).strip()
    return cleaned, scope  # type: ignore[return-value]


@app.post("/textbook/chat", response_model=TextbookChatResponse)
def textbook_chat(req: TextbookChatRequest) -> TextbookChatResponse:
    """Selection-based contextual chat.

    Gemma answers using a tiered source policy: first the selected excerpt,
    then the full page context, and — when the question is on-topic but not
    literally on the page — a short general-knowledge answer flagged as
    ``beyond_page`` in the response. Completely unrelated questions come back
    tagged ``off_topic`` with a polite redirect.
    """
    try:
        client = _get_endpoint_client()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"LLM endpoint is not configured: {e}") from e

    prompt = _build_textbook_chat_prompt(req)
    try:
        raw = client.generate_text(prompt, max_new_tokens=req.max_new_tokens)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Gemma call failed: {e}") from e

    answer, source = _split_source_tag((raw or "").strip())
    if not answer:
        answer = (
            "I could not generate a reply for that. Try rephrasing the question, "
            "or select a smaller, more specific passage."
        )
        source = "unknown"
    return TextbookChatResponse(
        answer=answer,
        used_selection=bool((req.selected_text or "").strip()),
        source=source,
    )


@app.get("/learn/page_image")
def learn_page_image(page_no: int) -> FileResponse:
    if page_no < 1:
        raise HTTPException(status_code=400, detail="page_no must be >= 1")
    if not DEFAULT_PDF_PATH.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {DEFAULT_PDF_PATH}")

    out_dir = PROJECT_ROOT / "modules/ui_module/outputs/page_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"page_{page_no}.png"
    if out_path.exists():
        return FileResponse(str(out_path), media_type="image/png")

    doc = fitz.open(DEFAULT_PDF_PATH)
    try:
        idx = page_no - 1
        if idx < 0 or idx >= len(doc):
            raise HTTPException(status_code=404, detail=f"page_no={page_no} out of PDF range")
        page = doc[idx]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
        pix.save(str(out_path))
    finally:
        doc.close()
    return FileResponse(str(out_path), media_type="image/png")


@app.get("/learn/quiz_link")
def learn_quiz_link(chapter_name: str, current_page: int) -> dict[str, Any]:
    page = max(1, int(current_page))
    url = f"/quiz?chapter_name={chapter_name}&current_page={page}"
    return {"quiz_url": url, "chapter_name": chapter_name, "current_page": page}


@app.get("/quiz", response_class=HTMLResponse)
def quiz_page(chapter_name: str, current_page: int, questions_per_page: int = 3) -> HTMLResponse:
    mcqs = _get_or_generate_quiz_mcqs(
        chapter_name=chapter_name.strip(),
        current_page=max(1, int(current_page)),
        questions_per_page=max(1, int(questions_per_page)),
    )
    html = _quiz_html(chapter_name.strip(), max(1, int(current_page)), mcqs)
    return HTMLResponse(content=html)


@app.post("/quiz/submit", response_class=HTMLResponse)
async def quiz_submit(request: Request) -> HTMLResponse:
    form = await request.form()
    chapter_name = str(form.get("chapter_name", "Electrostatics")).strip()
    current_page = max(1, int(form.get("current_page", 1)))
    mcqs = _get_or_generate_quiz_mcqs(chapter_name=chapter_name, current_page=current_page, questions_per_page=3)
    answer_map = {str(m.get("question_id")): str(m.get("answer", "")).upper() for m in mcqs}

    total = len(mcqs)
    correct = 0
    for qid, ans in answer_map.items():
        picked = str(form.get(qid, "")).upper()
        if picked and picked == ans:
            correct += 1

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Quiz Result</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#f5f8ff; margin:0; }}
    .wrap {{ max-width:780px; margin:36px auto; background:#fff; padding:24px; border-radius:12px; box-shadow:0 8px 24px rgba(0,0,0,.08); text-align:center; }}
    .score {{ font-size:32px; font-weight:700; color:#15429b; }}
    a.btn {{ display:inline-block; margin-top:16px; padding:10px 16px; background:#1f6feb; color:#fff; text-decoration:none; border-radius:8px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h2>✅ Quiz Submitted</h2>
    <p>Chapter: <strong>{chapter_name}</strong> | Pages: <strong>{_quiz_page_numbers(chapter_name, current_page)[0]} - {current_page}</strong></p>
    <p class="score">{correct} / {total}</p>
    <p>You answered <strong>{correct}</strong> correctly out of <strong>{total}</strong>.</p>
    <a class="btn" href="../quiz?chapter_name={chapter_name}&current_page={current_page}">Try Again</a>
  </div>
</body>
</html>
"""
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("learn_api:app", host="127.0.0.1", port=8000, reload=False)

