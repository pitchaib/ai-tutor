"""Interactive lesson page: narrate chunks, ask questions, evaluate answers."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import gradio as gr

import sys

PROJECT_ROOT = Path(os.getenv("AITUTOR_ROOT", "/home/bp/AiTutor"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_runtime_dependencies() -> None:
    need = {
        "google.genai": "google-genai",
        "google.cloud.speech": "google-cloud-speech",
        "google.cloud.aiplatform_v1": "google-cloud-aiplatform",
    }
    missing = []
    for module_name, pkg in need.items():
        try:
            __import__(module_name)
        except Exception:
            missing.append(pkg)
    if not missing:
        return
    cmd = [sys.executable, "-m", "pip", "install", *sorted(set(missing))]
    subprocess.run(cmd, check=True)


@lru_cache(maxsize=1)
def _pipeline():
    _ensure_runtime_dependencies()
    from modules.voice_qa_module.src.voice_qa_pipeline import (
        VoiceQAPipeline,
        VoiceQAPipelineConfig,
    )

    # GCP project and region are read from environment variables (VERTEX_PROJECT_ID,
    # VERTEX_LOCATION) which are sourced from configs/vertex.env by start.sh.
    # Do NOT hardcode project IDs or regions here.
    cfg = VoiceQAPipelineConfig(
        dictionary_json=PROJECT_ROOT
        / "modules/teacher_module/outputs/chunk_summary_dictionary.json",
        output_dir=PROJECT_ROOT / "outputs/voice_qa",
    )
    return VoiceQAPipeline(cfg)


def _asr_code(choice: str) -> str:
    val = (choice or "").strip().lower()
    if val.startswith("ta"):
        return "ta-IN"
    return "en-US"


def _to_tmp_wav(path: Path) -> str:
    tmp = tempfile.NamedTemporaryFile(prefix="voice_qa_answer_", suffix=".wav", delete=False)
    tmp.close()
    shutil.copyfile(path, tmp.name)
    return tmp.name


def _save_b64_wav(audio_b64: str | None) -> Path:
    payload = (audio_b64 or "").strip()
    if not payload:
        raise ValueError("No microphone audio captured.")
    if payload.startswith("data:"):
        payload = payload.split(",", 1)[-1]
    raw = base64.b64decode(payload)
    tmp = tempfile.NamedTemporaryFile(prefix="voice_qa_question_", suffix=".wav", delete=False)
    tmp.write(raw)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _flatten_chunks() -> list[dict]:
    data = _pipeline().dictionary
    out: list[dict] = []
    for page in data.get("pages", []):
        page_no = int(page.get("page_no", 0))
        for idx, chunk in enumerate(page.get("teacher_explanation_chunks", []), start=1):
            out.append(
                {
                    "page_no": page_no,
                    "chunk_no": idx,
                    "teacher_explanation": str(chunk.get("teacher_explanation", "")).strip(),
                    "summary": str(chunk.get("summary", "")).strip(),
                    "question": str(chunk.get("question", "")).strip(),
                    "answer": str(chunk.get("answer", "")).strip(),
                }
            )
    out.sort(key=lambda c: (c["page_no"], c["chunk_no"]))
    return out


@dataclass
class LessonAudioPrefetcher:
    session_id: str
    chunks: list[dict]
    answer_language: str

    def __post_init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.futures: dict[int, Future] = {}
        self.base_dir = PROJECT_ROOT / "outputs/voice_qa/lesson" / self.session_id
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _text_for_chunk(self, idx: int) -> str:
        c = self.chunks[idx]
        text = c["teacher_explanation"]
        if c["question"]:
            text += f"\n\nNow answer this question:\n{c['question']}"
        return text

    def _path_for_chunk(self, idx: int) -> Path:
        c = self.chunks[idx]
        return self.base_dir / f"page_{c['page_no']:03d}_chunk_{c['chunk_no']:03d}.wav"

    def _submit(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.chunks) or idx in self.futures:
            return
        out_path = self._path_for_chunk(idx)
        text = self._text_for_chunk(idx)
        self.futures[idx] = self.executor.submit(_pipeline().synthesize_answer, text, out_path)

    def warmup(self, start_idx: int) -> None:
        self._submit(start_idx)
        self._submit(start_idx + 1)

    def get(self, idx: int) -> Path:
        self._submit(idx)
        result = self.futures[idx].result()
        self._submit(idx + 2)
        return result

    def close(self) -> None:
        self.executor.shutdown(wait=False)


_LESSON_PREFETCHERS: dict[str, LessonAudioPrefetcher] = {}


def _intro_text(chapter_name: str, language: str) -> str:
    if (language or "").lower() == "tamil":
        return f"Good morning. Today we are going to study the chapter {chapter_name}. Let us begin."
    return f"Good morning. Today we are going to study the chapter {chapter_name}. Let us begin."


def _greeting_prompt(language: str) -> str:
    if (language or "").lower() == "tamil":
        return "Good morning. Before we start, how are you? How is your day?"
    return "Good morning. Before we start, how are you? How is your day?"


def _build_greeting_reply(
    *,
    student_text: str,
    chapter_name: str,
    language: str,
    pipeline,
) -> str:
    """Create natural conversational greeting reply via LLM."""
    context = pipeline.retrieve_context(chapter_name, top_k=3)
    prompt = (
        "You are a warm and friendly AI tutor greeting a student before lesson.\n\n"
        f"Student said: {student_text}\n"
        f"Chapter to teach: {chapter_name}\n"
        f"Preferred language: {language}\n\n"
        "Rules:\n"
        "1) Reply naturally in 2-3 short sentences.\n"
        "2) Show empathy for student's mood.\n"
        "3) Invite them into the lesson positively.\n"
        "4) Do NOT repeat their full sentence verbatim.\n"
        "5) Plain text only."
    )
    out = (pipeline.endpoint_client.generate_text(prompt, max_new_tokens=180) or "").strip()
    if out:
        return out
    return f"Thanks for sharing. Let us learn {chapter_name} together, step by step."


def _lesson_status(state: dict) -> str:
    total = int(state.get("total_chunks", 0))
    idx = int(state.get("idx", 0))
    mode = state.get("mode", "idle")
    paused = bool(state.get("is_paused", False))
    paused_tag = " | Paused" if paused else ""
    if mode == "waiting_for_greeting":
        return "Mode: waiting_for_greeting | Please answer the greeting with Ask/Answer."
    if mode == "done":
        return f"Lesson completed. Reviewed {total} chunks."
    if idx < total:
        c = state["chunks"][idx]
        return (
            f"Mode: {mode}{paused_tag} | Page {c['page_no']} Chunk {c['chunk_no']} | "
            f"Progress: {idx + 1}/{total}"
        )
    return f"Mode: {mode}{paused_tag} | Progress: {idx}/{total}"


def _build_narration_segment(chunks: list[dict], start_idx: int) -> tuple[str, int, int | None]:
    """
    Build one narration segment from start_idx up to the next question boundary.
    Returns:
      - segment_text
      - next_idx (first chunk after segment)
      - question_idx (index of chunk that has question, if included)
    """
    if start_idx >= len(chunks):
        return "", start_idx, None

    parts: list[str] = []
    i = start_idx
    question_idx: int | None = None
    while i < len(chunks):
        c = chunks[i]
        txt = (c.get("teacher_explanation", "") or "").strip()
        if txt:
            parts.append(txt)
        q = (c.get("question", "") or "").strip()
        if q:
            parts.append(f"Now answer this question: {q}")
            question_idx = i
            i += 1
            break
        i += 1

    return "\n\n".join(parts).strip(), i, question_idx


def start_lesson(
    chapter_name: str, answer_language: str
) -> tuple[dict, str | None, str, gr.update, str, str, str, gr.update]:
    chunks = _flatten_chunks()
    if not chunks:
        return (
            {},
            None,
            "No chunk data found.",
            gr.update(value="Ask/Answer"),
            "idle",
            "",
            "false",
            gr.update(value="Pause Lesson"),
        )

    session_id = f"{int(time.time() * 1000)}_{uuid4().hex[:8]}"
    state = {
        "session_id": session_id,
        "chapter_name": (chapter_name or "Electrostatics").strip(),
        "answer_language": answer_language or "English",
        "chunks": chunks,
        "idx": 0,
        "total_chunks": len(chunks),
        "mode": "waiting_for_greeting",
        "is_paused": False,
        "current_idx": None,
        "wrong_counts": {},
        "struggled": [],
    }

    prefetch = LessonAudioPrefetcher(
        session_id=session_id,
        chunks=chunks,
        answer_language=answer_language or "English",
    )
    prefetch.warmup(0)
    _LESSON_PREFETCHERS[session_id] = prefetch

    greet_path = PROJECT_ROOT / "outputs/voice_qa/lesson" / session_id / "greeting.wav"
    greet_audio = _pipeline().synthesize_answer(
        _greeting_prompt(state["answer_language"]),
        greet_path,
    )
    status = (
        "Greeting played. Please answer using Ask/Answer. "
        "After that, lesson narration will start automatically.\n\n"
        + _lesson_status(state)
    )
    return (
        state,
        _to_tmp_wav(greet_audio),
        status,
        gr.update(value="Ask/Answer"),
        "waiting_for_greeting",
        "",
        "false",
        gr.update(value="Pause Lesson"),
    )


def toggle_pause_resume(state: dict | None) -> tuple[dict, str, str, gr.update, str, gr.update]:
    if not state:
        return {}, "Start lesson first.", "false", gr.update(value="Pause Lesson"), "idle", gr.update()
    if state.get("mode") == "done":
        state["is_paused"] = False
        return state, "Lesson already completed.", "false", gr.update(value="Pause Lesson"), "done", gr.update()
    paused = bool(state.get("is_paused", False))
    state["is_paused"] = not paused
    if state["is_paused"]:
        prev_mode = str(state.get("mode", "idle"))
        state["_mode_before_pause"] = prev_mode
        if prev_mode == "narrating":
            # Hard-stop and later replay from segment checkpoint instead of skipping ahead.
            state["_resume_replay_pending"] = True
            state["_resume_replay_idx"] = int(state.get("_last_segment_start_idx", state.get("idx", 0)))
        state["mode"] = "paused"
        return (
            state,
            "Lesson paused. Click Resume Lesson to continue.",
            "true",
            gr.update(value="Resume Lesson"),
            "paused",
            gr.update(value=None, autoplay=False),
        )
    resumed_mode = str(state.get("_mode_before_pause", "narrating"))
    if resumed_mode == "paused":
        resumed_mode = "narrating"
    status = "Lesson resumed."
    if resumed_mode == "narrating" and bool(state.get("_resume_replay_pending", False)):
        state["idx"] = int(state.get("_resume_replay_idx", state.get("idx", 0)))
        state["_resume_replay_pending"] = False
        status = "Lesson resumed from the paused point."
    state["mode"] = resumed_mode
    return state, status, "false", gr.update(value="Pause Lesson"), resumed_mode, gr.update()


def run_lesson_step(
    audio_b64: str | None,
    state: dict | None,
    chapter_name: str,
    answer_language: str,
    asr_language: str,
) -> tuple[dict, str | None, str, gr.update, str, str]:
    if not state:
        return {}, None, "Start lesson first.", gr.update(value="Ask/Answer"), "idle", ""

    session_id = state.get("session_id", "")
    mode = state.get("mode", "idle")
    chunks = state.get("chunks", [])
    idx = int(state.get("idx", 0))
    total = int(state.get("total_chunks", 0))
    p = _pipeline()

    if mode == "done" or idx >= total:
        state["mode"] = "done"
        return state, None, _lesson_status(state), gr.update(value="Ask/Answer"), "done", ""

    if bool(state.get("is_paused", False)):
        return (
            state,
            None,
            "Lesson is paused. Click Resume Lesson to continue.",
            gr.update(value="Ask/Answer"),
            mode,
            "",
        )

    if mode == "waiting_for_greeting":
        if not (audio_b64 or "").strip():
            return (
                state,
                None,
                "Please answer the greeting using Ask/Answer.",
                gr.update(value="Ask/Answer"),
                "waiting_for_greeting",
                "",
            )
        try:
            in_path = _save_b64_wav(audio_b64)
            student_text = p.transcribe_audio(in_path, language_code=_asr_code(asr_language))
            chapter_name_local = state.get("chapter_name", "this chapter")
            greet_reply = _build_greeting_reply(
                student_text=student_text,
                chapter_name=chapter_name_local,
                language=answer_language or "English",
                pipeline=p,
            )
            greet_reply = f"{greet_reply} {_intro_text(chapter_name_local, answer_language or 'English')}"
        except Exception as exc:  # noqa: BLE001
            return (
                state,
                None,
                f"Could not process greeting response: {exc}",
                gr.update(value="Ask/Answer"),
                "waiting_for_greeting",
                "",
            )

        # Deterministic kickoff: include first chunk immediately after greeting
        # so lesson always starts right away even if frontend misses one auto-advance event.
        if idx < total:
            seg_text, next_idx, q_idx = _build_narration_segment(chunks, idx)
            state["_last_segment_start_idx"] = idx
            kickoff_text = f"{greet_reply}\n\nLet us begin the lesson.\n\n{seg_text}".strip()
            kickoff_path = (
                PROJECT_ROOT
                / "outputs/voice_qa/lesson"
                / session_id
                / f"greeting_kickoff_{int(time.time() * 1000)}.wav"
            )
            kickoff_audio = p.synthesize_answer(kickoff_text, kickoff_path)

            if q_idx is not None:
                state["mode"] = "waiting_for_answer"
                state["current_idx"] = q_idx
                state["idx"] = q_idx
                status = (
                    "Greeting completed. I covered the first lesson segment and asked a question.\n\n"
                    + _lesson_status(state)
                )
                return (
                    state,
                    _to_tmp_wav(kickoff_audio),
                    status,
                    gr.update(value="Ask/Answer"),
                    "waiting_for_answer",
                    "",
                )

            state["idx"] = next_idx
            state["mode"] = "narrating"
            if state["idx"] >= total:
                state["mode"] = "done"
                status = "Greeting completed and full lesson segment finished."
                return (
                    state,
                    _to_tmp_wav(kickoff_audio),
                    status,
                    gr.update(value="Ask/Answer"),
                    "done",
                    "",
                )
            status = (
                "Greeting completed and lesson started. Continuing automatically.\n\n"
                + _lesson_status(state)
            )
            return (
                state,
                _to_tmp_wav(kickoff_audio),
                status,
                gr.update(value="Ask/Answer"),
                "narrating",
                "",
            )

        # Fallback: no chunks available after greeting.
        done_path = PROJECT_ROOT / "outputs/voice_qa/lesson" / session_id / "greeting_done.wav"
        done_audio = p.synthesize_answer(greet_reply, done_path)
        state["mode"] = "done"
        status = "Greeting completed, but no lesson chunks are available."
        return (
            state,
            _to_tmp_wav(done_audio),
            status,
            gr.update(value="Ask/Answer"),
            "done",
            "",
        )

    prefetch = _LESSON_PREFETCHERS.get(session_id)
    if prefetch is None:
        prefetch = LessonAudioPrefetcher(session_id=session_id, chunks=chunks, answer_language=answer_language or "English")
        prefetch.warmup(idx)
        _LESSON_PREFETCHERS[session_id] = prefetch

    if mode == "narrating":
        # If student pressed Ask/Answer during narration, treat it as free-form doubt.
        if (audio_b64 or "").strip():
            try:
                in_path = _save_b64_wav(audio_b64)
                student_q = p.transcribe_audio(in_path, language_code=_asr_code(asr_language))
                context = p.retrieve_context(student_q)
                answer = p.answer_question(
                    question=student_q,
                    language=answer_language or "English",
                    context_chunks=context,
                )
            except Exception as exc:  # noqa: BLE001
                return state, None, f"Could not answer question: {exc}", gr.update(value="Ask/Answer"), "narrating", ""

            freeform_path = (
                PROJECT_ROOT
                / "outputs/voice_qa/lesson"
                / session_id
                / f"student_ask_{int(time.time() * 1000)}.wav"
            )
            freeform_audio = p.synthesize_answer(answer, freeform_path)
            status = (
                f"You asked: {student_q}\n\n"
                "I answered your doubt. Lesson narration will continue automatically."
            )
            return (
                state,
                _to_tmp_wav(freeform_audio),
                status,
                gr.update(value="Ask/Answer"),
                "narrating",
                "",
            )

        seg_text, next_idx, q_idx = _build_narration_segment(chunks, idx)
        state["_last_segment_start_idx"] = idx
        if not seg_text:
            state["mode"] = "done"
            return state, None, "Lesson completed.", gr.update(value="Ask/Answer"), "done", ""

        seg_path = (
            PROJECT_ROOT
            / "outputs/voice_qa/lesson"
            / session_id
            / f"segment_{int(time.time() * 1000)}.wav"
        )
        seg_audio = p.synthesize_answer(seg_text, seg_path)

        if q_idx is not None:
            state["mode"] = "waiting_for_answer"
            state["current_idx"] = q_idx
            state["idx"] = q_idx
            c = chunks[q_idx]
            status = (
                f"Question asked for page {c['page_no']} chunk {c['chunk_no']}. "
                "Press Ask/Answer and speak your answer."
            )
            return state, _to_tmp_wav(seg_audio), status, gr.update(value="Ask/Answer"), "waiting_for_answer", ""

        state["idx"] = next_idx
        if state["idx"] >= total:
            state["mode"] = "done"
            done_text = "Great job! You completed the lesson."
            done_path = PROJECT_ROOT / "outputs/voice_qa/lesson" / session_id / "lesson_done.wav"
            done_audio = p.synthesize_answer(done_text, done_path)
            return state, _to_tmp_wav(done_audio), _lesson_status(state), gr.update(value="Ask/Answer"), "done", ""

        return state, _to_tmp_wav(seg_audio), _lesson_status(state), gr.update(value="Ask/Answer"), "narrating", ""

    if mode == "waiting_for_answer":
        if not (audio_b64 or "").strip():
            return state, None, "No answer captured. Press Ask/Answer and speak.", gr.update(value="Ask/Answer"), "waiting_for_answer", ""

        current_idx = int(state.get("current_idx", idx))
        chunk = chunks[current_idx]
        key = f"{chunk['page_no']}:{chunk['chunk_no']}"

        try:
            in_path = _save_b64_wav(audio_b64)
            student_text = p.transcribe_audio(in_path, language_code=_asr_code(asr_language))
        except Exception as exc:  # noqa: BLE001
            return state, None, f"Could not transcribe answer: {exc}", gr.update(value="Ask/Answer"), "waiting_for_answer", ""

        result = p.evaluate_student_answer(
            question=chunk.get("question", ""),
            expected_answer=chunk.get("answer", ""),
            student_answer=student_text,
            context_text=chunk.get("teacher_explanation", ""),
            language=answer_language or "English",
        )
        is_correct = bool(result.get("is_correct", False))
        feedback = str(result.get("feedback", "")).strip() or "Thanks for your answer."

        # Single-attempt policy:
        # ask question once, evaluate once, then move forward.
        if is_correct:
            state["mode"] = "narrating"
            state["idx"] = current_idx + 1
            state["current_idx"] = None
            feedback_text = f"{feedback} Correct! Good job. Let's continue."
        else:
            feedback_text = (
                f"{feedback} "
                f"The correct answer is: {chunk.get('answer', '')}. "
                "Let us move to the next concept."
            )
            state["struggled"].append(
                {
                    "page_no": chunk["page_no"],
                    "chunk_no": chunk["chunk_no"],
                    "question": chunk.get("question", ""),
                    "student_answer": student_text,
                }
            )
            state["mode"] = "narrating"
            state["idx"] = current_idx + 1
            state["current_idx"] = None

        # Deterministic continuation:
        # include feedback + next narration segment in same audio so flow never stalls after Q&A.
        next_idx = int(state.get("idx", 0))
        if next_idx >= total:
            state["mode"] = "done"
            final_text = f"{feedback_text} Great job! You completed the lesson."
            final_path = (
                PROJECT_ROOT
                / "outputs/voice_qa/lesson"
                / session_id
                / f"feedback_done_{int(time.time() * 1000)}.wav"
            )
            final_audio = p.synthesize_answer(final_text, final_path)
            return (
                state,
                _to_tmp_wav(final_audio),
                "Lesson completed.",
                gr.update(value="Ask/Answer"),
                "done",
                "",
            )

        seg_text, seg_next_idx, seg_q_idx = _build_narration_segment(chunks, next_idx)
        state["_last_segment_start_idx"] = next_idx
        combined_text = feedback_text
        if seg_text:
            combined_text = f"{feedback_text}\n\nLet us continue.\n\n{seg_text}"

        combined_path = (
            PROJECT_ROOT
            / "outputs/voice_qa/lesson"
            / session_id
            / f"feedback_continue_{int(time.time() * 1000)}.wav"
        )
        combined_audio = p.synthesize_answer(combined_text, combined_path)

        if seg_q_idx is not None:
            state["mode"] = "waiting_for_answer"
            state["current_idx"] = seg_q_idx
            state["idx"] = seg_q_idx
            c = chunks[seg_q_idx]
            status = (
                f"Question asked for page {c['page_no']} chunk {c['chunk_no']}. "
                "Press Ask/Answer and speak your answer."
            )
            return (
                state,
                _to_tmp_wav(combined_audio),
                status,
                gr.update(value="Ask/Answer"),
                "waiting_for_answer",
                "",
            )

        state["idx"] = seg_next_idx
        if state["idx"] >= total:
            state["mode"] = "done"
            status = "Lesson completed."
            lesson_mode = "done"
        else:
            state["mode"] = "narrating"
            status = _lesson_status(state)
            lesson_mode = "narrating"
        return (
            state,
            _to_tmp_wav(combined_audio),
            status,
            gr.update(value="Ask/Answer"),
            lesson_mode,
            "",
        )

    return state, None, f"Unexpected mode: {mode}", gr.update(value="Ask/Answer"), "idle", ""


VOICE_JS = """
() => {
  const askEl = document.getElementById("ask_btn");
  const askBtn = askEl ? (askEl.tagName === "BUTTON" ? askEl : askEl.querySelector("button")) : null;
  const hiddenBox = document.querySelector("#audio_b64 textarea, textarea#audio_b64, #audio_b64 input, input#audio_b64");
  const modeBox = document.querySelector("#lesson_mode textarea, textarea#lesson_mode, #lesson_mode input, input#lesson_mode");
  const triggerEl = document.getElementById("process_btn");
  const triggerBtn = triggerEl ? (triggerEl.tagName === "BUTTON" ? triggerEl : triggerEl.querySelector("button")) : null;
  const statusWrap = document.getElementById("status_md");
  const wave = document.getElementById("record_wave");

  const setStatus = (msg) => {
    if (!statusWrap) return;
    const p = statusWrap.querySelector("p");
    if (p) p.textContent = msg;
    else statusWrap.textContent = msg;
  };
  const mode = (modeBox && modeBox.value ? modeBox.value : "idle").trim();
  const paused = () => !!window.__lessonUserPaused;

  if (!askBtn || !hiddenBox || !triggerBtn) {
    setStatus("UI wiring issue: refresh page once.");
    return [];
  }
  if (window.__voiceAskRunning) return [];
  if (paused()) {
    setStatus("Lesson is paused. Click Resume Lesson to continue.");
    return [];
  }
  if (mode === "done") {
    setStatus("Lesson already completed.");
    return [];
  }

  window.__voiceAskRunning = true;
  askBtn.classList.add("ask-recording");
  askBtn.classList.remove("ask-processing");
  if (wave) wave.classList.add("active");
  if (mode === "waiting_for_greeting") {
    setStatus("Listening... answer the greeting.");
  } else if (mode === "waiting_for_answer") {
    setStatus("Listening... speak your answer.");
  } else {
    setStatus("Listening... ask your doubt.");
  }

  const SILENCE_MS = 3000;
  const NO_SPEECH_MS = 7000;
  const FORCE_STOP_MS = 10000;
  const BASE_THRESHOLD = 0.008;

  navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioCtx();
    audioCtx.resume();
    const source = audioCtx.createMediaStreamSource(stream);
    const processor = audioCtx.createScriptProcessor(4096, 1, 1);
    const samples = [];
    let spoken = false;
    let silenceFor = 0;
    let finished = false;
    let lastTick = Date.now();
    const startedAt = Date.now();
    let noiseFloor = BASE_THRESHOLD;
    let calibrating = true;
    const calibrationValues = [];

    processor.onaudioprocess = (e) => {
      if (finished) return;
      const input = e.inputBuffer.getChannelData(0);
      samples.push(new Float32Array(input));

      let sum = 0.0;
      for (let i = 0; i < input.length; i += 1) sum += input[i] * input[i];
      const rms = Math.sqrt(sum / input.length);
      const now = Date.now();
      const dt = now - lastTick;
      lastTick = now;

      if (calibrating) {
        calibrationValues.push(rms);
        if (now - startedAt > 1000) {
          const avg = calibrationValues.length ? calibrationValues.reduce((a, b) => a + b, 0) / calibrationValues.length : BASE_THRESHOLD;
          noiseFloor = Math.max(BASE_THRESHOLD, avg * 2.5);
          calibrating = false;
          setStatus("Listening... speak your answer.");
        } else {
          setStatus("Calibrating mic noise...");
        }
      }

      const threshold = Math.max(BASE_THRESHOLD, noiseFloor);
      if (!calibrating && rms > threshold) {
        spoken = true;
        silenceFor = 0;
      } else if (spoken) {
        silenceFor += dt;
      }
      if ((spoken && silenceFor >= SILENCE_MS) || (!spoken && (now - startedAt) >= NO_SPEECH_MS) || (now - startedAt) >= FORCE_STOP_MS) {
        stopAndSend();
      }
    };

    source.connect(processor);
    processor.connect(audioCtx.destination);

    function merge(buffers) {
      let len = 0;
      for (const b of buffers) len += b.length;
      const out = new Float32Array(len);
      let offset = 0;
      for (const b of buffers) { out.set(b, offset); offset += b.length; }
      return out;
    }

    function floatTo16BitPCM(view, offset, input) {
      for (let i = 0; i < input.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, input[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      }
    }

    function writeString(view, offset, str) {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    }

    function encodeWav(float32, sampleRate) {
      const buffer = new ArrayBuffer(44 + float32.length * 2);
      const view = new DataView(buffer);
      writeString(view, 0, "RIFF");
      view.setUint32(4, 36 + float32.length * 2, true);
      writeString(view, 8, "WAVE");
      writeString(view, 12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeString(view, 36, "data");
      view.setUint32(40, float32.length * 2, true);
      floatTo16BitPCM(view, 44, float32);
      return buffer;
    }

    function stopAndSend() {
      if (finished) return;
      finished = true;
      try {
        processor.disconnect();
        source.disconnect();
        stream.getTracks().forEach((t) => t.stop());
      } catch (_) {}

      const mono = merge(samples);
      const wav = encodeWav(mono, audioCtx.sampleRate || 44100);
      const bytes = new Uint8Array(wav);
      let binary = "";
      for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
      const b64 = btoa(binary);

      hiddenBox.value = b64;
      hiddenBox.dispatchEvent(new Event("input", { bubbles: true }));
      hiddenBox.dispatchEvent(new Event("change", { bubbles: true }));
      askBtn.classList.remove("ask-recording");
      askBtn.classList.add("ask-processing");
      if (wave) wave.classList.remove("active");
      setStatus("Processing your voice...");
      triggerBtn.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
      setTimeout(() => askBtn.classList.remove("ask-processing"), 2500);
      window.__voiceAskRunning = false;
    }
  }).catch((err) => {
    setStatus("Microphone permission error: " + String(err));
    askBtn.classList.remove("ask-recording");
    askBtn.classList.remove("ask-processing");
    if (wave) wave.classList.remove("active");
    window.__voiceAskRunning = false;
  });
  return [];
}
"""

INIT_JS = """
() => {
  const triggerEl = document.getElementById("process_btn");
  const triggerBtn = triggerEl ? (triggerEl.tagName === "BUTTON" ? triggerEl : triggerEl.querySelector("button")) : null;
  const hiddenBox = document.querySelector("#audio_b64 textarea, textarea#audio_b64, #audio_b64 input, input#audio_b64");
  const modeBox = document.querySelector("#lesson_mode textarea, textarea#lesson_mode, #lesson_mode input, input#lesson_mode");
  const anim = document.getElementById("ai_voice_anim");

  const modeVal = () => (modeBox && modeBox.value ? modeBox.value.trim() : "idle");
  const getAudioEl = () => document.querySelector("#lesson_audio audio");
  const isPaused = () => !!window.__lessonUserPaused;
  const setAnimPlaying = (playing) => {
    if (!anim) return;
    if (playing) anim.classList.add("playing");
    else anim.classList.remove("playing");
  };
  const findPauseButton = () => {
    const host = document.getElementById("pause_btn");
    if (!host) return null;
    return host.tagName === "BUTTON" ? host : host.querySelector("button");
  };
  const setPauseLabel = (paused) => {
    const btn = findPauseButton();
    if (!btn) return;
    btn.textContent = paused ? "Resume Lesson" : "Pause Lesson";
  };

  const pauseAudio = () => {
    const el = getAudioEl();
    if (!el) return;
    try { el.pause(); } catch (e) {}
    setAnimPlaying(false);
  };
  const resumeAudio = () => {
    const el = getAudioEl();
    if (!el) return;
    const p = el.play();
    if (p && typeof p.catch === "function") p.catch(() => {});
  };

  const triggerAdvance = () => {
    if (!triggerBtn || !hiddenBox) return;
    hiddenBox.value = "";
    hiddenBox.dispatchEvent(new Event("input", { bubbles: true }));
    hiddenBox.dispatchEvent(new Event("change", { bubbles: true }));
    triggerBtn.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
  };

  // Client-side pause button: controls the <audio> element directly (like a native media player).
  const bindPauseButton = () => {
    const btn = findPauseButton();
    if (!btn || btn.dataset.lessonPauseBound === "1") return;
    btn.dataset.lessonPauseBound = "1";
    btn.addEventListener("click", () => {
      const nextPaused = !isPaused();
      window.__lessonUserPaused = nextPaused;
      setPauseLabel(nextPaused);
      if (nextPaused) {
        pauseAudio();
      } else {
        resumeAudio();
        const el = getAudioEl();
        const hasClip = !!(el && el.currentSrc && !el.ended);
        if (!hasClip && modeVal() === "narrating") {
          window.__lessonLastAdvanceAt = Date.now();
          triggerAdvance();
        }
      }
    }, true);
  };

  const bindAudio = () => {
    const audioEl = getAudioEl();
    if (!audioEl || audioEl.dataset.lessonBound === "1") return;
    audioEl.dataset.lessonBound = "1";

    const updateAnim = () => setAnimPlaying(!audioEl.paused && !audioEl.ended);
    const tryAutoplay = () => {
      if (isPaused()) return;
      if (modeVal() !== "narrating") return;
      if (audioEl.paused) {
        const p = audioEl.play();
        if (p && typeof p.catch === "function") p.catch(() => {});
      }
    };
    audioEl.addEventListener("play", updateAnim);
    audioEl.addEventListener("pause", updateAnim);
    audioEl.addEventListener("loadedmetadata", tryAutoplay);
    audioEl.addEventListener("canplay", tryAutoplay);
    audioEl.addEventListener("canplaythrough", tryAutoplay);
    audioEl.addEventListener("loadeddata", tryAutoplay);
    audioEl.addEventListener("ended", () => {
      updateAnim();
      if (isPaused()) return;
      if (modeVal() !== "narrating") return;
      window.__lessonAdvanceDoneForSrc = window.__lessonAdvanceDoneForSrc || {};
      const src = audioEl.currentSrc || "__no_src__";
      if (window.__lessonAdvanceDoneForSrc[src]) return;
      window.__lessonAdvanceDoneForSrc[src] = true;
      window.__lessonLastAdvanceAt = Date.now();
      triggerAdvance();
    });
    updateAnim();
    if (isPaused()) pauseAudio();
    else tryAutoplay();
  };

  bindAudio();
  bindPauseButton();
  if (window.__lessonAudioBindTimer) clearInterval(window.__lessonAudioBindTimer);
  window.__lessonAudioBindTimer = setInterval(() => { bindAudio(); bindPauseButton(); }, 400);

  // Auto-advance watchdog: advances to next chunk after current audio ends during narration.
  if (window.__lessonAdvanceWatchdog) clearInterval(window.__lessonAdvanceWatchdog);
  window.__lessonAdvanceWatchdog = setInterval(() => {
    if (isPaused()) return;
    if (modeVal() !== "narrating") return;
    const now = Date.now();
    const lastAdv = window.__lessonLastAdvanceAt || 0;
    const inCooldown = now - lastAdv < 1800;
    const audioEl = getAudioEl();
    if (!audioEl) {
      if (!inCooldown) {
        window.__lessonLastAdvanceAt = now;
        triggerAdvance();
      }
      return;
    }
    window.__lessonAdvanceDoneForSrc = window.__lessonAdvanceDoneForSrc || {};
    const src = audioEl.currentSrc || "__no_src__";
    if (window.__lessonLastSrc !== src) {
      window.__lessonLastSrc = src;
      if (audioEl.paused && !isPaused()) {
        const p = audioEl.play();
        if (p && typeof p.catch === "function") p.catch(() => {});
      }
    }
    const started = audioEl.currentTime > 0.2;
    const endedLike = audioEl.ended || (audioEl.paused && started && audioEl.readyState >= 2);
    if (!endedLike) return;
    if (inCooldown) return;
    if (window.__lessonAdvanceDoneForSrc[src]) return;
    window.__lessonAdvanceDoneForSrc[src] = true;
    window.__lessonLastAdvanceAt = now;
    triggerAdvance();
  }, 900);
  return [];
}
"""


CUSTOM_CSS = """
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
  width: 120px;
  height: 120px;
  border-radius: 60px;
  margin: 10px auto 12px;
  position: relative;
  background: radial-gradient(circle at 30% 30%, #a5c8ff, #3b6cc9);
  box-shadow: 0 0 0 0 rgba(59,108,201,0.45);
}
#ai_voice_anim::before,
#ai_voice_anim::after {
  content: "";
  position: absolute;
  inset: 22px;
  border-radius: 50%;
  border: 3px solid rgba(255, 255, 255, 0.75);
}
#ai_voice_anim::after {
  inset: 36px;
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
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="AI Tutor Lesson Mode", css=CUSTOM_CSS) as demo:
        gr.Markdown("# AI Tutor Lesson Mode")
        gr.Markdown(
            "Start lesson to hear chunk-by-chunk teaching. "
            "When a question comes, use Ask/Answer to record your response."
        )

        with gr.Row():
            chapter = gr.Dropdown(
                label="Chapter",
                choices=["Electrostatics", "Current Electricity", "Magnetism", "Optics"],
                value="Electrostatics",
            )
            answer_language = gr.Radio(
                label="Tutor Language",
                choices=["English", "Tamil", "Mix"],
                value="English",
            )
            asr_language = gr.Radio(
                label="Answer ASR Language",
                choices=["English (en-US)", "Tamil (ta-IN)"],
                value="English (en-US)",
            )

        with gr.Row():
            start_btn = gr.Button("Start Lesson", variant="secondary")
            pause_btn = gr.Button("Pause Lesson", variant="secondary", elem_id="pause_btn")
            ask_btn = gr.Button("Ask/Answer", variant="primary", elem_id="ask_btn")
        gr.HTML(
            "<div id='record_wave'>"
            "<span></span><span></span><span></span><span></span><span></span><span></span>"
            "</div>"
        )
        gr.HTML("<div id='ai_voice_anim' title='AI voice playing'></div>")

        lesson_state = gr.State(value={})
        hidden_audio_b64 = gr.Textbox(value="", label="", elem_id="audio_b64")
        lesson_mode = gr.Textbox(value="idle", label="", elem_id="lesson_mode")
        lesson_paused = gr.Textbox(value="false", label="", elem_id="lesson_paused")
        process_btn = gr.Button("Process", elem_id="process_btn")

        answer_audio = gr.Audio(label="Lesson Audio", interactive=False, autoplay=True, elem_id="lesson_audio")
        status = gr.Markdown("Ready. Click Start Lesson.", elem_id="status_md")

        start_btn.click(
            fn=start_lesson,
            inputs=[chapter, answer_language],
            outputs=[
                lesson_state,
                answer_audio,
                status,
                ask_btn,
                lesson_mode,
                hidden_audio_b64,
                lesson_paused,
                pause_btn,
            ],
        )
        process_event = process_btn.click(
            fn=run_lesson_step,
            inputs=[hidden_audio_b64, lesson_state, chapter, answer_language, asr_language],
            outputs=[lesson_state, answer_audio, status, ask_btn, lesson_mode, hidden_audio_b64],
        )
        pause_btn.click(
            fn=toggle_pause_resume,
            inputs=[lesson_state],
            outputs=[lesson_state, status, lesson_paused, pause_btn, lesson_mode, answer_audio],
            queue=False,
            cancels=[process_event],
        )
        ask_btn.click(fn=None, inputs=[], outputs=[], js=VOICE_JS)
        demo.load(fn=None, inputs=[], outputs=[], js=INIT_JS)

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch(server_name="127.0.0.1", server_port=7861)

