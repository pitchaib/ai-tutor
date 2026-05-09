"""End-to-end voice QA pipeline: ASR -> retrieval -> Gemma -> TTS."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

# Project root resolved from the AITUTOR_ROOT environment variable.
# Set AITUTOR_ROOT to the absolute path of this repository before running.
PROJECT_ROOT = Path(os.getenv("AITUTOR_ROOT", str(Path(__file__).resolve().parents[4])))
_logger = logging.getLogger(__name__)

# ── Module-level HTTP client & credential cache ──────────────────────────
# Historically every dedicated-endpoint call opened a fresh TLS connection
# via urllib + refreshed ADC credentials.  Reusing both across requests
# saves ~150-300 ms of TLS handshake and ~50-100 ms of token refresh per
# call — material when a single tutor turn hits the endpoint twice.
_HTTP_CLIENT: Any = None
_HTTP_CLIENT_LOCK = threading.Lock()

_CREDENTIALS: Any = None
_CREDENTIALS_LOCK = threading.Lock()


def _get_http_client():
    """Return a singleton httpx.Client with HTTP/2 + keep-alive."""
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None:
        return _HTTP_CLIENT
    with _HTTP_CLIENT_LOCK:
        if _HTTP_CLIENT is None:
            try:
                import httpx  # lazy so we don't hard-require httpx
            except ImportError as e:
                raise ImportError(
                    "httpx is required for Vertex endpoint calls."
                ) from e
            _HTTP_CLIENT = httpx.Client(
                http2=False,
                timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
                limits=httpx.Limits(
                    max_connections=8, max_keepalive_connections=4, keepalive_expiry=300.0
                ),
            )
    return _HTTP_CLIENT


def _get_credentials_and_token() -> tuple[Any, str]:
    """Cached ADC credentials; refresh only when the access token has expired."""
    global _CREDENTIALS
    import google.auth
    from google.auth.transport.requests import Request as GoogleAuthRequest

    with _CREDENTIALS_LOCK:
        if _CREDENTIALS is None:
            _CREDENTIALS, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        # `.expired` is False when the token still has >= 5 min of life left.
        if not _CREDENTIALS.valid or _CREDENTIALS.expired:
            _CREDENTIALS.refresh(GoogleAuthRequest())
        token = _CREDENTIALS.token or ""
    if not token:
        raise RuntimeError("Could not obtain ADC access token for Vertex endpoint call.")
    return _CREDENTIALS, token


@dataclass
class VertexEndpointConfig:
    project_id: str
    location: str
    endpoint_id: str
    api_endpoint: str | None = None


def parse_vertex_endpoint_url(endpoint_url: str) -> VertexEndpointConfig:
    parsed = urlparse(endpoint_url)
    m = re.search(r"/locations/([^/]+)/endpoints/([^/?#]+)", parsed.path)
    if not m:
        raise ValueError(f"Could not parse location/endpoint from URL: {endpoint_url}")
    location = m.group(1)
    endpoint_id = m.group(2)
    qs = parse_qs(parsed.query or "")
    project = (qs.get("project") or [""])[0]
    if not project:
        raise ValueError(
            "Project id not found in endpoint URL query. Add '?project=<project-id>' to URL."
        )
    return VertexEndpointConfig(project_id=project, location=location, endpoint_id=endpoint_id)


class VertexEndpointClient:
    def __init__(self, config: VertexEndpointConfig):
        self.config = config
        self._client = None
        self._endpoint_path = None

    def _is_dedicated_prediction_host(self) -> bool:
        host = (self.config.api_endpoint or "").strip().lower()
        return host.endswith(".prediction.vertexai.goog")

    def _predict_via_dedicated_rest(
        self, payload: dict[str, Any], max_new_tokens: int
    ) -> dict[str, Any]:
        """POST to the dedicated prediction host using a pooled HTTPS client.

        Both the httpx client AND the ADC credentials are cached module-wide
        (see `_get_http_client` / `_get_credentials_and_token`) so repeat
        calls skip the TLS handshake + token refresh.
        """
        try:
            _, token = _get_credentials_and_token()
        except ImportError as e:
            raise ImportError(
                "google-auth is required for dedicated endpoint REST calls."
            ) from e

        host = (self.config.api_endpoint or "").strip()
        url = (
            f"https://{host}/v1/projects/{self.config.project_id}/locations/"
            f"{self.config.location}/endpoints/{self.config.endpoint_id}:predict"
        )
        body = {
            "instances": [payload],
            "parameters": {
                "maxOutputTokens": int(max_new_tokens),
                "max_new_tokens": int(max_new_tokens),
                "temperature": 0.2,
                "top_p": 0.9,
            },
        }
        client = _get_http_client()
        resp = client.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        resp.raise_for_status()
        return resp.json()

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from google.cloud import aiplatform_v1
        except Exception as e:
            raise ImportError(
                "google-cloud-aiplatform is required for Vertex endpoint calls."
            ) from e
        api_host = self.config.api_endpoint or f"{self.config.location}-aiplatform.googleapis.com"
        self._client = aiplatform_v1.PredictionServiceClient(
            client_options={"api_endpoint": api_host}
        )
        self._endpoint_path = self._client.endpoint_path(
            project=self.config.project_id,
            location=self.config.location,
            endpoint=self.config.endpoint_id,
        )

    @staticmethod
    def _to_value(payload: dict[str, Any]):
        from google.protobuf import struct_pb2
        from google.protobuf.json_format import ParseDict

        out = struct_pb2.Value()
        ParseDict(payload, out)
        return out

    @staticmethod
    def _prediction_to_text(pred: Any) -> str:
        if isinstance(pred, dict):
            choices = pred.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    msg = first.get("message")
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        if isinstance(content, str) and content.strip():
                            return content.strip()
                    content = first.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
        if isinstance(pred, dict) and "stringValue" in pred:
            return str(pred.get("stringValue", "")).strip()
        if isinstance(pred, str):
            return pred.strip()
        if isinstance(pred, dict):
            for key in ("generated_text", "prediction", "text", "output_text", "content"):
                v = pred.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return str(pred).strip()

    def generate_text(self, prompt: str, max_new_tokens: int = 128) -> str:
        attempts = (
            {
                "@requestFormat": "chatCompletions",
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                "max_tokens": int(max_new_tokens),
            },
            {"prompt": prompt},
            {"inputs": prompt},
            {"text": prompt},
        )
        last_err: Exception | None = None
        if self._is_dedicated_prediction_host():
            for payload in attempts:
                try:
                    data = self._predict_via_dedicated_rest(payload, max_new_tokens)
                    preds = data.get("predictions")
                    if preds is None:
                        return ""
                    if isinstance(preds, dict):
                        return self._prediction_to_text(preds)
                    if isinstance(preds, list):
                        if not preds:
                            return ""
                        return self._prediction_to_text(preds[0])
                    return self._prediction_to_text(preds)
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    continue
            raise RuntimeError(
                "Dedicated endpoint predict failed for all payload shapes "
                f"{attempts}. Last error: {last_err}"
            ) from last_err

        self._ensure_client()
        from google.protobuf.json_format import MessageToDict

        parameters = self._to_value(
            {
                "maxOutputTokens": int(max_new_tokens),
                "max_new_tokens": int(max_new_tokens),
                "temperature": 0.2,
                "top_p": 0.9,
            }
        )
        response = None
        for payload in attempts:
            instances = [self._to_value(payload)]
            try:
                response = self._client.predict(
                    endpoint=self._endpoint_path,
                    instances=instances,
                    parameters=parameters,
                )
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        if response is None:
            raise RuntimeError(
                "Vertex endpoint predict failed for all payload shapes "
                f"{attempts}. Last error: {last_err}"
            ) from last_err
        if not response.predictions:
            return ""
        pred_dict = MessageToDict(response.predictions[0])
        return self._prediction_to_text(pred_dict)


def load_vertex_endpoint_client(
    *,
    endpoint_url: str | None = None,
    project_id: str | None = None,
    location: str | None = None,
    endpoint_id: str | None = None,
    api_endpoint: str | None = None,
) -> VertexEndpointClient:
    if endpoint_url:
        cfg = parse_vertex_endpoint_url(endpoint_url)
    else:
        if not (project_id and location and endpoint_id):
            raise ValueError("Provide endpoint_url OR all of project_id/location/endpoint_id.")
        cfg = VertexEndpointConfig(
            project_id=project_id,
            location=location,
            endpoint_id=endpoint_id,
            api_endpoint=api_endpoint,
        )
    if endpoint_url and api_endpoint:
        cfg.api_endpoint = api_endpoint
    return VertexEndpointClient(cfg)


def _env(name: str, default: str | None = None) -> str | None:
    """Read env var, returning default for missing/empty values."""
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip()
    return val if val else default


@dataclass
class VoiceQAPipelineConfig:
    # Defaults pulled from environment — sourced from configs/vertex.env by start.sh.
    project_id: str = field(default_factory=lambda: _env("VERTEX_PROJECT_ID", "") or "")
    location: str = field(default_factory=lambda: _env("VERTEX_LOCATION", "us-central1") or "us-central1")
    dictionary_json: Path = (
        PROJECT_ROOT / "modules/teacher_module/outputs/chunk_summary_dictionary.json"
    )
    output_dir: Path = PROJECT_ROOT / "outputs/voice_qa"
    # If VERTEX_ENDPOINT_URL is set, it wins; otherwise load_vertex_endpoint_client
    # is given project/location/endpoint_id individually.
    vertex_endpoint_url: str = field(default_factory=lambda: _env("VERTEX_ENDPOINT_URL", "") or "")
    vertex_endpoint_id: str = field(default_factory=lambda: _env("VERTEX_ENDPOINT_ID", "") or "")
    vertex_api_endpoint: str | None = field(default_factory=lambda: _env("VERTEX_API_ENDPOINT"))
    asr_endpoint: str = "speech.googleapis.com"
    tts_model: str = "gemini-2.5-flash-preview-tts"
    tts_voice: str = "Kore"
    # TTS lives on a managed publisher model; its regional availability is
    # independent of (and narrower than) our Gemma endpoint's region.
    # `gemini-2.5-flash-preview-tts` is not in asia-southeast1 as of 2026-04,
    # so we pin TTS to its own region (default us-central1) via env.
    tts_project: str = field(
        default_factory=lambda: _env("VERTEX_TTS_PROJECT") or _env("VERTEX_PROJECT_ID", "") or ""
    )
    tts_location: str = field(
        default_factory=lambda: _env("VERTEX_TTS_LOCATION", "us-central1") or "us-central1"
    )
    # Top-k for retrieval.  Lowered from 6 → 3 — a school-level answer almost
    # never benefits from more context than that, and the prompt prefill cost
    # is roughly linear in this value.
    retrieval_top_k: int = 3
    asr_chunk_ms: int = 100
    # ASR model.  "latest_short" is tuned for utterances < ~60 s (typical
    # student answers) and is noticeably faster than "default" + streaming.
    asr_model: str = "latest_short"


class VoiceQAPipeline:
    def __init__(self, config: VoiceQAPipelineConfig | None = None) -> None:
        self.config = config or VoiceQAPipelineConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.config.dictionary_json.exists():
            raise FileNotFoundError(
                f"Dictionary file not found: {self.config.dictionary_json}"
            )
        with open(self.config.dictionary_json, encoding="utf-8") as f:
            self.dictionary = json.load(f)

        # Prefer VERTEX_ENDPOINT_URL; fall back to split project/location/endpoint id.
        if self.config.vertex_endpoint_url:
            self.endpoint_client = load_vertex_endpoint_client(
                endpoint_url=self.config.vertex_endpoint_url,
                api_endpoint=self.config.vertex_api_endpoint,
            )
        elif self.config.project_id and self.config.location and self.config.vertex_endpoint_id:
            self.endpoint_client = load_vertex_endpoint_client(
                project_id=self.config.project_id,
                location=self.config.location,
                endpoint_id=self.config.vertex_endpoint_id,
                api_endpoint=self.config.vertex_api_endpoint,
            )
        else:
            raise RuntimeError(
                "Vertex endpoint config missing. Populate configs/vertex.env "
                "(VERTEX_ENDPOINT_URL or VERTEX_PROJECT_ID + VERTEX_LOCATION + "
                "VERTEX_ENDPOINT_ID, plus VERTEX_API_ENDPOINT for private endpoints)."
            )

        try:
            from google.cloud import speech as speech_mod
            from google import genai as genai_mod
            from google.genai import types as genai_types_mod
        except Exception as exc:
            raise ImportError(
                "Missing required packages for voice pipeline. "
                "Install `google-cloud-speech` and `google-genai` in your active env."
            ) from exc

        self._speech = speech_mod
        self._genai_types = genai_types_mod
        self.speech_client = speech_mod.SpeechClient(
            client_options={"api_endpoint": self.config.asr_endpoint}
        )
        self.genai_client = genai_mod.Client(
            vertexai=True,
            project=self.config.project_id,
            location=self.config.location,
        )
        # Separate client for TTS — routed to a region that hosts the
        # preview TTS model (see tts_location docstring).
        tts_project = self.config.tts_project or self.config.project_id
        tts_location = self.config.tts_location or self.config.location
        if tts_project == self.config.project_id and tts_location == self.config.location:
            self.genai_tts_client = self.genai_client
        else:
            self.genai_tts_client = genai_mod.Client(
                vertexai=True,
                project=tts_project,
                location=tts_location,
            )

    # ---------- ASR ----------
    def _wav_metadata(self, path: Path) -> dict[str, int]:
        with wave.open(str(path), "rb") as wf:
            return {
                "channels": wf.getnchannels(),
                "sample_rate_hz": wf.getframerate(),
                "sample_width_bytes": wf.getsampwidth(),
            }

    def _wav_pcm_chunks(self, path: Path):
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frames_per_chunk = max(
                1, int(sample_rate * (self.config.asr_chunk_ms / 1000.0))
            )
            bytes_per_frame = channels * sample_width

            while True:
                data = wf.readframes(frames_per_chunk)
                if not data:
                    break
                if len(data) % bytes_per_frame != 0:
                    data = data[: len(data) - (len(data) % bytes_per_frame)]
                if data:
                    yield data

    def transcribe_audio(self, wav_path: Path, language_code: str = "en-US") -> str:
        """Synchronous ASR on a pre-recorded WAV.

        Uses `recognize()` (not streaming) because we already have the full
        audio on disk — streaming adds overhead with zero early-result
        benefit.  The "latest_short" model is tuned for utterances < 60 s,
        which matches a tutor-mode student turn.
        """
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        meta = self._wav_metadata(wav_path)
        if meta["sample_width_bytes"] != 2:
            raise ValueError(
                f"Expected LINEAR16 WAV; got sample_width={meta['sample_width_bytes']}"
            )

        audio_bytes = wav_path.read_bytes()
        config = self._speech.RecognitionConfig(
            encoding=self._speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=meta["sample_rate_hz"],
            language_code=language_code,
            audio_channel_count=meta["channels"],
            enable_automatic_punctuation=True,
            model=self.config.asr_model,
        )
        audio = self._speech.RecognitionAudio(content=audio_bytes)
        response = self.speech_client.recognize(config=config, audio=audio)

        finals: list[str] = []
        for result in getattr(response, "results", []) or []:
            if not result.alternatives:
                continue
            text = result.alternatives[0].transcript.strip()
            if text:
                finals.append(text)
        transcript = " ".join(finals).strip()
        if not transcript:
            raise RuntimeError("ASR returned empty transcript.")
        return transcript

    # ---------- Retrieval ----------
    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9']+", (text or "").lower()))

    def retrieve_context(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        k = top_k or self.config.retrieval_top_k
        q_tokens = self._tokenize(question)
        scored: list[tuple[int, dict[str, Any]]] = []

        for page in self.dictionary.get("pages", []):
            page_no = int(page.get("page_no", 0))
            for idx, chunk in enumerate(page.get("teacher_explanation_chunks", []), start=1):
                text = str(chunk.get("teacher_explanation", ""))
                summ = str(chunk.get("summary", ""))
                ques = str(chunk.get("question", ""))
                ans = str(chunk.get("answer", ""))
                blob = f"{text} {summ} {ques} {ans}"
                tokens = self._tokenize(blob)
                overlap = len(q_tokens & tokens)
                if overlap > 0:
                    scored.append(
                        (
                            overlap,
                            {
                                "page_no": page_no,
                                "chunk_id": idx,
                                "teacher_explanation": text,
                                "summary": summ,
                                "question": ques,
                                "answer": ans,
                            },
                        )
                    )

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:k]]

    # ---------- LLM ----------
    def answer_question(
        self,
        *,
        question: str,
        language: str = "English",
        context_chunks: list[dict[str, Any]],
        max_new_tokens: int = 220,
    ) -> str:
        context_text = "\n\n".join(
            [
                (
                    f"[Page {c['page_no']} | Chunk {c['chunk_id']}]\n"
                    f"Explanation: {c['teacher_explanation']}\n"
                    f"Summary: {c['summary']}\n"
                    f"Related QA: Q={c['question']} A={c['answer']}"
                )
                for c in context_chunks
            ]
        )

        prompt = f"""
You are a helpful AI tutor.

User question:
{question}

Preferred answer language: {language}

Use this textbook context first:
{context_text}

Instructions:
1) Answer clearly and directly.
2) Prefer context-based answer; use external knowledge only if needed.
3) If using external knowledge, keep it short and aligned with textbook level.
4) Output plain answer text only (no markdown, no labels).
""".strip()

        out = self.endpoint_client.generate_text(prompt, max_new_tokens=max_new_tokens)
        out = (out or "").strip()
        if not out:
            raise RuntimeError("LLM returned empty answer.")
        return out

    def evaluate_student_answer(
        self,
        *,
        question: str,
        expected_answer: str,
        student_answer: str,
        context_text: str = "",
        language: str = "English",
        max_new_tokens: int = 160,
    ) -> dict[str, Any]:
        """Evaluate whether a student's answer is correct and provide feedback."""
        prompt = f"""
You are an expert 12th-standard tutor evaluating a spoken student answer.

Question:
{question}

Expected answer:
{expected_answer}

Student answer:
{student_answer}

Reference context:
{context_text}

Preferred language for feedback: {language}

Rules:
1) Judge conceptually, not exact wording.
2) If student captures the key idea, mark correct.
3) Return valid JSON only with exactly these keys:
   - is_correct (boolean)
   - feedback (string)
   - reason (string)
4) Keep feedback short, encouraging, and clear for school students.
5) If incorrect, explain what is missing.
""".strip()

        raw = (self.endpoint_client.generate_text(prompt, max_new_tokens=max_new_tokens) or "").strip()
        if not raw:
            return {
                "is_correct": False,
                "feedback": "I could not evaluate your answer clearly. Please try once more.",
                "reason": "empty_model_output",
            }

        try:
            start = raw.find("{")
            end = raw.rfind("}")
            candidate = raw[start : end + 1] if start != -1 and end != -1 and end > start else raw
            data = json.loads(candidate)
            is_correct = bool(data.get("is_correct", False))
            feedback = str(data.get("feedback", "")).strip()
            reason = str(data.get("reason", "")).strip()
            if not feedback:
                feedback = "Good attempt. Let us review this concept once more."
            return {"is_correct": is_correct, "feedback": feedback, "reason": reason}
        except Exception:
            text = raw.lower()
            is_correct = any(k in text for k in ("correct", "well done", "good job"))
            fallback = (
                "Correct! Good job."
                if is_correct
                else "Not fully correct. Let me explain the key idea and you can try again."
            )
            return {"is_correct": is_correct, "feedback": fallback, "reason": "non_json_model_output"}

    # ---------- TTS ----------
    def _audio_part_to_bytes(self, part) -> bytes:
        data = part.inline_data.data
        if isinstance(data, bytes):
            return data
        if isinstance(data, str):
            return base64.b64decode(data)
        raise TypeError(f"Unexpected audio data type: {type(data)}")

    def _parse_audio_mime(self, mime_type: str) -> tuple[bool, int, int]:
        mime = (mime_type or "").lower().strip()
        if "wav" in mime or "x-wav" in mime:
            return True, 24000, 1
        sample_rate = 24000
        channels = 1
        if "rate=" in mime:
            try:
                sample_rate = int(mime.split("rate=")[1].split(";")[0].split(",")[0])
            except Exception:
                pass
        if "channels=" in mime:
            try:
                channels = int(mime.split("channels=")[1].split(";")[0].split(",")[0])
            except Exception:
                pass
        return False, sample_rate, channels

    def _write_pcm_as_wav(
        self, pcm_bytes: bytes, out_path: Path, sample_rate: int, channels: int
    ) -> None:
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(max(1, channels))
            wf.setsampwidth(2)
            wf.setframerate(max(8000, sample_rate))
            wf.writeframes(pcm_bytes)

    def synthesize_answer(
        self, answer_text: str, out_path: Path, voice_name: str | None = None
    ) -> Path:
        voice = voice_name or self.config.tts_voice
        cfg = self._genai_types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=self._genai_types.SpeechConfig(
                voice_config=self._genai_types.VoiceConfig(
                    prebuilt_voice_config=self._genai_types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        )
        response = self.genai_tts_client.models.generate_content(
            model=self.config.tts_model,
            contents=answer_text,
            config=cfg,
        )
        part = response.candidates[0].content.parts[0]
        audio_bytes = self._audio_part_to_bytes(part)
        mime_type = getattr(part.inline_data, "mime_type", "") or ""

        out_path.parent.mkdir(parents=True, exist_ok=True)
        is_wav, sample_rate, channels = self._parse_audio_mime(mime_type)
        if is_wav or audio_bytes[:4] == b"RIFF":
            out_path.write_bytes(audio_bytes)
        else:
            self._write_pcm_as_wav(
                audio_bytes, out_path, sample_rate=sample_rate, channels=channels
            )
        return out_path

    # ---------- Warm-up ----------
    def warmup(self) -> None:
        """Hit the LLM endpoint and the TTS model with tiny payloads so the
        first real tutor turn doesn't pay for cold-start token refresh /
        connection setup.  Exceptions are logged and swallowed — warm-up is
        strictly best-effort.
        """
        try:
            self.endpoint_client.generate_text("Reply with OK.", max_new_tokens=8)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("LLM warm-up failed (non-fatal): %s", exc)
        try:
            cfg = self._genai_types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=self._genai_types.SpeechConfig(
                    voice_config=self._genai_types.VoiceConfig(
                        prebuilt_voice_config=self._genai_types.PrebuiltVoiceConfig(
                            voice_name=self.config.tts_voice
                        )
                    )
                ),
            )
            self.genai_tts_client.models.generate_content(
                model=self.config.tts_model, contents="Hi.", config=cfg
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning("TTS warm-up failed (non-fatal): %s", exc)

    # ---------- Public ----------
    def run(
        self,
        *,
        input_audio_path: str | Path,
        output_audio_path: str | Path,
        asr_language_code: str = "en-US",
        answer_language: str = "English",
        retrieval_top_k: int | None = None,
    ) -> Path:
        in_path = Path(input_audio_path)
        out_path = Path(output_audio_path)

        question = self.transcribe_audio(in_path, language_code=asr_language_code)
        context = self.retrieve_context(question, top_k=retrieval_top_k)
        answer = self.answer_question(
            question=question,
            language=answer_language,
            context_chunks=context,
        )
        return self.synthesize_answer(answer, out_path)

