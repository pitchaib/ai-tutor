from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

import fitz
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_CHOICES = {
    "26B_A4B": "google/gemma-4-26B-A4B",
    "E4B": "google/gemma-4-E4B",
}


@dataclass
class ChapterSpan:
    chapter_name: str
    start_page: int
    end_page: int


@dataclass
class VertexEndpointConfig:
    project_id: str
    location: str
    endpoint_id: str
    api_endpoint: str | None = None


def parse_vertex_endpoint_url(endpoint_url: str) -> VertexEndpointConfig:
    """
    Parse a Vertex endpoint console URL into config fields.
    Example:
    https://console.cloud.google.com/vertex-ai/online-prediction/locations/us-central1/endpoints/123...?project=my-proj
    """
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
    """
    Lightweight wrapper for Vertex AI endpoint predict().
    Assumes endpoint accepts prompt-like input and returns generated text in predictions.
    """

    def __init__(self, config: VertexEndpointConfig):
        self.config = config
        self._client = None
        self._endpoint_path = None

    def _is_dedicated_prediction_host(self) -> bool:
        host = (self.config.api_endpoint or "").strip().lower()
        return host.endswith(".prediction.vertexai.goog")

    def _predict_via_dedicated_rest(self, payload: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
        try:
            import google.auth
            from google.auth.transport.requests import Request as GoogleAuthRequest
        except Exception as e:
            raise ImportError(
                "google-auth is required for dedicated endpoint REST calls."
            ) from e

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(GoogleAuthRequest())
        token = credentials.token
        if not token:
            raise RuntimeError(
                "Could not obtain ADC access token for Vertex endpoint call."
            )

        host = (self.config.api_endpoint or "").strip()
        url = (
            f"https://{host}/v1/projects/{self.config.project_id}/locations/"
            f"{self.config.location}/endpoints/{self.config.endpoint_id}:predict"
        )
        body = json.dumps(
            {
                "instances": [payload],
                "parameters": {
                    "maxOutputTokens": int(max_new_tokens),
                    "max_new_tokens": int(max_new_tokens),
                    "temperature": 0.2,
                    "top_p": 0.9,
                },
            }
        ).encode("utf-8")
        req = Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        with urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from google.cloud import aiplatform_v1
        except Exception as e:
            raise ImportError(
                "google-cloud-aiplatform is required for Vertex endpoint calls. "
                "Install requirements again in your notebook kernel."
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
            # chatCompletions-style response
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
        if isinstance(pred, dict) and "structValue" in pred:
            fields = pred.get("structValue", {}).get("fields", {})
            flat = {
                k: (v.get("stringValue") if isinstance(v, dict) else v)
                for k, v in fields.items()
            }
            return VertexEndpointClient._prediction_to_text(flat)
        if isinstance(pred, str):
            return pred.strip()
        if isinstance(pred, dict):
            for key in (
                "generated_text",
                "prediction",
                "text",
                "output_text",
                "content",
            ):
                v = pred.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            cands = pred.get("candidates")
            if isinstance(cands, list) and cands:
                first = cands[0]
                if isinstance(first, str):
                    return first.strip()
                if isinstance(first, dict):
                    part = first.get("content")
                    if isinstance(part, str) and part.strip():
                        return part.strip()
                    parts = first.get("parts")
                    if isinstance(parts, list):
                        txt = " ".join(
                            p.get("text", "").strip()
                            for p in parts
                            if isinstance(p, dict) and p.get("text")
                        ).strip()
                        if txt:
                            return txt
        return str(pred).strip()

    def generate_text(self, prompt: str, max_new_tokens: int = 128) -> str:
        attempts = (
            {
                "@requestFormat": "chatCompletions",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
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
                    # Some endpoints return predictions as a dict (chatCompletions),
                    # others return a list of prediction items.
                    if isinstance(preds, dict):
                        txt = self._prediction_to_text(preds)
                        if txt:
                            return txt
                        return ""
                    if isinstance(preds, list):
                        if not preds:
                            return ""
                        return self._prediction_to_text(preds[0])
                    # Last-resort parse for unusual shapes.
                    txt = self._prediction_to_text(preds)
                    if txt:
                        return txt
                    return ""
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
                msg = str(e)
                if (
                    "Reauthentication is needed" in msg
                    or "application-default login" in msg
                    or "Getting metadata from plugin failed" in msg
                ):
                    raise RuntimeError(
                        "Vertex authentication failed (ADC expired/missing). "
                        "Run: `gcloud auth application-default login` and then retry. "
                        "If using a service account, set GOOGLE_APPLICATION_CREDENTIALS "
                        "to the key JSON path before starting Jupyter."
                    ) from e
                if (
                    "Private endpoint via Private Service Connect cannot be accessed" in msg
                    or "Private Service Connect" in msg
                ):
                    raise RuntimeError(
                        "This Vertex endpoint is PRIVATE (PSC). Public domain calls are blocked. "
                        "Use a PSC private DNS/API host from inside the connected VPC "
                        "(set api_endpoint in load_vertex_endpoint_client), or deploy a public endpoint."
                    ) from e
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
            raise ValueError(
                "Provide endpoint_url OR all of project_id/location/endpoint_id."
            )
        cfg = VertexEndpointConfig(
            project_id=project_id,
            location=location,
            endpoint_id=endpoint_id,
            api_endpoint=api_endpoint,
        )
    if endpoint_url and api_endpoint:
        cfg.api_endpoint = api_endpoint
    return VertexEndpointClient(cfg)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def looks_like_toc_page(text: str) -> bool:
    """
    Heuristic: table-of-contents / index pages list many units/chapters with page numbers.
    Body pages should not be treated as chapter start when only the title substring matches here.
    """
    if not text or len(text.strip()) < 40:
        return False
    head = text[:3000].lower()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    if re.search(r"\bcontents?\b", head) or re.search(r"\bindex\b", head[:800]):
        return True
    # Dot leaders (..... 12) common in TOC
    if re.search(r"\.{4,}", text):
        return True
    # Many lines ending with a page number (narrow line, trailing digits)
    trailing_num = sum(
        1
        for ln in lines[:80]
        if len(ln) < 140 and re.search(r"[\s\.…]{2,}\d{1,4}\s*$", ln)
    )
    if len(lines) >= 6 and trailing_num >= 4:
        return True
    # Dense list of numbered sections (1., 2., 1.1, etc.) typical of syllabus/index
    numbered = sum(1 for ln in lines[:60] if re.match(r"^\d+(\.\d+)*\s+\S", ln))
    if len(lines) >= 10 and numbered >= 6:
        return True
    return False


def looks_like_toc_by_chapter_list(page_text: str, chapter_names: list[str]) -> bool:
    """True if several requested chapter titles appear on the same page (typical index / contents)."""
    hay = normalize_text(page_text)
    hits = sum(1 for c in chapter_names if normalize_text(c) and normalize_text(c) in hay)
    return hits >= 2


def page_mentions_chapter_only_as_toc_lines(text: str, chapter_name: str) -> bool:
    """
    True when the chapter title appears only in TOC-style lines (dot leaders / trailing page no.),
    not as a real section heading + body. Used when you request a single chapter name.
    """
    key = (chapter_name.strip().split() or [""])[0].lower()
    if len(key) < 3:
        return False
    lines_with_key: list[str] = []
    for ln in text.splitlines():
        if key in ln.lower():
            lines_with_key.append(ln)
    if not lines_with_key:
        return False

    def line_looks_toc_entry(ln: str) -> bool:
        s = ln.strip()
        if re.search(r"\.{3,}", ln) or "…" in ln:
            return True
        if len(s) < 130 and re.search(r"\d{1,4}\s*$", s):
            return True
        return False

    return all(line_looks_toc_entry(ln) for ln in lines_with_key)


def scan_toc_end_index(page_texts: list[str], max_scan: int = 80) -> int:
    """
    Index of the first page after the initial TOC block (0-based).
    If no TOC is detected, returns 0 (search whole PDF — caller can still use min_page).
    """
    n = min(len(page_texts), max_scan)
    if n == 0:
        return 0
    i = 0
    while i < n and looks_like_toc_page(page_texts[i]):
        i += 1
    # If early pages look like cover (very little text), skip them
    while i < n and len(page_texts[i].strip()) < 80:
        i += 1
    return i


def extract_pdf_page_texts(pdf_path: str | Path) -> list[str]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    def _page_text_reading_order(page: fitz.Page) -> str:
        """
        Extract page text with a stable reading order.
        For two-column layouts, force left column first, then right column.
        """
        blocks = page.get_text("blocks")
        cleaned: list[tuple[float, float, float, float, str]] = []
        for b in blocks:
            if not isinstance(b, (tuple, list)) or len(b) < 5:
                continue
            x0, y0, x1, y1, txt = b[:5]
            if not isinstance(txt, str):
                continue
            t = txt.strip()
            if not t:
                continue
            cleaned.append((float(x0), float(y0), float(x1), float(y1), t))

        if not cleaned:
            return page.get_text("text")

        page_w = float(page.rect.width)
        mid = page_w * 0.5
        gutter = page_w * 0.02
        left = [b for b in cleaned if b[0] <= (mid - gutter)]
        right = [b for b in cleaned if b[0] >= (mid + gutter)]
        center = [b for b in cleaned if b not in left and b not in right]

        # Detect likely two-column page by separated x-start clusters.
        two_col = False
        if len(left) >= 2 and len(right) >= 2:
            left_x = sorted(b[0] for b in left)
            right_x = sorted(b[0] for b in right)
            med_left = left_x[len(left_x) // 2]
            med_right = right_x[len(right_x) // 2]
            if (med_right - med_left) >= (page_w * 0.22):
                two_col = True

        if two_col:
            left_sorted = sorted(left, key=lambda b: (b[1], b[0]))
            right_sorted = sorted(right, key=lambda b: (b[1], b[0]))
            center_sorted = sorted(center, key=lambda b: (b[1], b[0]))
            ordered = left_sorted + right_sorted + center_sorted
        else:
            ordered = sorted(cleaned, key=lambda b: (b[1], b[0]))

        return "\n".join(b[4] for b in ordered).strip()

    doc = fitz.open(path)
    pages = [_page_text_reading_order(page) for page in doc]
    doc.close()
    return pages


def find_chapter_page(
    page_texts: list[str],
    chapter_name: str,
    start_search_page: int = 0,
    *,
    skip_toc_pages: bool = True,
    min_page: int = 0,
    all_chapter_names: list[str] | None = None,
) -> int:
    needle = normalize_text(chapter_name)
    if not needle:
        raise ValueError("chapter_name must not be empty")

    toc_end = scan_toc_end_index(page_texts) if skip_toc_pages else 0
    low = max(start_search_page, min_page, toc_end)

    for idx in range(low, len(page_texts)):
        if skip_toc_pages and looks_like_toc_page(page_texts[idx]):
            continue
        if (
            skip_toc_pages
            and all_chapter_names
            and len(all_chapter_names) >= 2
            and looks_like_toc_by_chapter_list(page_texts[idx], all_chapter_names)
        ):
            continue
        hay = normalize_text(page_texts[idx])
        if needle not in hay:
            continue
        if skip_toc_pages and page_mentions_chapter_only_as_toc_lines(
            page_texts[idx], chapter_name
        ):
            continue
        return idx

    # Fallback: TOC heuristic may miss some books — retry without TOC skip
    if skip_toc_pages:
        return find_chapter_page(
            page_texts,
            chapter_name,
            start_search_page=start_search_page,
            skip_toc_pages=False,
            min_page=min_page,
            all_chapter_names=None,
        )
    return -1


def locate_chapter_ranges(
    page_texts: list[str],
    chapter_names: Iterable[str],
    *,
    skip_toc_pages: bool = True,
    min_page: int = 0,
) -> list[ChapterSpan]:
    chapter_names = [c.strip() for c in chapter_names if c and c.strip()]
    if not chapter_names:
        raise ValueError("chapter_names must contain at least one chapter")

    starts: list[tuple[str, int]] = []
    cursor = 0
    for chapter in chapter_names:
        page_idx = find_chapter_page(
            page_texts,
            chapter,
            start_search_page=cursor,
            skip_toc_pages=skip_toc_pages,
            min_page=min_page,
            all_chapter_names=chapter_names,
        )
        if page_idx == -1:
            raise ValueError(f"Could not locate chapter in PDF: {chapter}")
        starts.append((chapter, page_idx))
        cursor = page_idx + 1

    def find_next_chapter_start_page(
        page_texts_local: list[str], start_page: int, current_chapter: str
    ) -> int:
        """
        Best-effort detection of the next *top-level* chapter start page.
        Used when the caller provides only one chapter name.
        """
        current_norm = normalize_text(current_chapter)
        for i in range(start_page + 1, len(page_texts_local)):
            page = page_texts_local[i]
            if looks_like_toc_page(page):
                continue
            lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
            top = lines[:30]
            for ln in top:
                s = _normalize_line_for_heading(ln)
                if not s:
                    continue
                # Top-level chapter styles: "2 Current Electricity", "CHAPTER 2 ...", etc.
                top_level = bool(
                    re.match(r"^\d+\s+[A-Z][A-Za-z0-9,()'’\- ]{3,80}$", s)
                    or re.match(r"^CHAPTER\s+\d+\b", s, re.IGNORECASE)
                )
                if not top_level:
                    continue
                # Avoid matching subsection lines and the same chapter title
                if re.match(r"^\d+\.\d+", s):
                    continue
                if current_norm and current_norm in normalize_text(s):
                    continue
                return i
        return -1

    spans: list[ChapterSpan] = []
    for i, (chapter, start_page) in enumerate(starts):
        if i < len(starts) - 1:
            end_page = starts[i + 1][1] - 1
        else:
            # Single-chapter request: infer chapter end from next top-level heading.
            next_start = find_next_chapter_start_page(page_texts, start_page, chapter)
            end_page = (next_start - 1) if next_start != -1 else (len(page_texts) - 1)
        spans.append(ChapterSpan(chapter_name=chapter, start_page=start_page, end_page=end_page))

    return spans


def extract_chapter_text(page_texts: list[str], span: ChapterSpan) -> str:
    return "\n".join(page_texts[span.start_page : span.end_page + 1]).strip()


def _looks_like_pdf_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    low = s.lower()
    if ".indd" in low:
        return True
    if re.match(r"^\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2}$", s):
        return True
    if re.match(r"^\d{1,4}$", s):
        return True
    if re.match(r"^unit\s+\d+\b", low) and len(s.split()) <= 6:
        return True
    return False


def _is_bullet_line(line: str) -> bool:
    s = line.strip()
    return bool(
        re.match(
            r"^(?:[•\-\*\u2022]|\d+[\.\)]|[ivxlcdm]+\)|\([ivxlcdm]+\))\s+",
            s,
            re.IGNORECASE,
        )
    )


def _page_blocks(page_text: str) -> list[dict[str, Any]]:
    """
    Split one page into ordered paragraph-like blocks.
    Bullet runs are grouped as one block.
    """
    lines = [ln.rstrip() for ln in page_text.splitlines()]
    blocks: list[dict[str, Any]] = []
    cur: list[str] = []
    cur_is_bullet = False

    def flush():
        nonlocal cur, cur_is_bullet
        txt = "\n".join(cur).strip()
        if txt:
            blocks.append({"text": txt, "is_bullet": cur_is_bullet})
        cur = []
        cur_is_bullet = False

    for raw in lines:
        s = raw.strip()
        if _looks_like_pdf_noise_line(s):
            continue
        if not s:
            flush()
            continue
        is_bullet = _is_bullet_line(s)
        if is_bullet and cur and not cur_is_bullet:
            flush()
        if not is_bullet and cur_is_bullet and cur:
            # continuation line for a bullet item/list
            cur.append(s)
            continue
        if not cur:
            cur_is_bullet = is_bullet
            cur = [s]
        else:
            cur.append(s)
    flush()
    return blocks


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """
    Best-effort parser for LLM JSON output (possibly wrapped in markdown fences).
    """
    if not text:
        return None
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else None
    except Exception:
        pass
    # fallback: first JSON object in text
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        val = json.loads(m.group(0))
        return val if isinstance(val, dict) else None
    except Exception:
        return None


def _llm_page_blocks(
    *,
    endpoint_client: VertexEndpointClient,
    chapter_name: str,
    page_no: int,
    page_text: str,
    max_new_tokens: int = 1600,
) -> list[dict[str, Any]]:
    """
    Ask LLM to segment one page into coherent paragraph blocks.
    Falls back to heuristic parser if JSON is invalid.
    """
    # Keep prompt size reasonable while preserving enough context for segmentation.
    clipped = page_text.strip()
    if len(clipped) > 14000:
        clipped = clipped[:14000]
    prompt = f"""You are a document-structure assistant.
Task: Segment the following textbook page into coherent teaching blocks.

Rules:
1) Work only with this page content.
2) Remove print/noise lines (timestamps, .indd lines, standalone page numbers, repeated headers).
3) IMPORTANT: At the beginning of chapter pages, ignore and EXCLUDE:
   - "Learning Objectives" / "Objectives" headings
   - all bullet points listed under those headings
   - chapter "Summary", outcomes, or overview bullets/lines
   These should never appear in output blocks.
4) KEEP all actual concept/explanation paragraphs that come after those headings.
5) Prefer 1 paragraph per block; merge very tiny adjacent fragments.
6) Return strict JSON only (no markdown):
{{
  "blocks": [
    {{
      "start_hint": "<first meaningful line fragment>",
      "end_hint": "<last line fragment>",
      "text": "<full block text>"
    }}
  ]
}}

Chapter: {chapter_name}
Page number: {page_no}

Page text:
{clipped}
"""
    out = endpoint_client.generate_text(prompt, max_new_tokens=max_new_tokens)
    parsed = _extract_json_object(out)
    if not parsed or "blocks" not in parsed or not isinstance(parsed["blocks"], list):
        return _page_blocks(page_text)
    blocks: list[dict[str, Any]] = []
    for b in parsed["blocks"]:
        if not isinstance(b, dict):
            continue
        txt = str(b.get("text", "")).strip()
        txt = _remove_objective_bullets(txt)
        if not txt:
            continue
        blocks.append({"text": txt, "is_bullet": _is_bullet_line(txt.splitlines()[0].strip())})
    return blocks if blocks else _page_blocks(page_text)


def _is_beginning_noise_block(text: str) -> bool:
    low = normalize_text(text)
    needles = (
        "learning objective",
        "learning objectives",
        "learning outcome",
        "learning outcomes",
        "summary",
        "in this unit",
        "chapter outline",
        "let us learn",
    )
    if any(n in low for n in needles):
        return True
    # Very short uppercase banners at chapter start often headings only.
    raw = text.strip()
    if len(raw.split()) <= 6 and raw.isupper():
        return True
    return False


def _line_is_beginning_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    low = normalize_text(s)
    if not low:
        return True
    # Treat only explicit heading-like objective/summary lines as skippable.
    explicit = (
        "learning objective",
        "learning objectives",
        "learning outcome",
        "learning outcomes",
        "summary",
        "chapter outline",
        "let us learn",
    )
    if any(k in low for k in explicit):
        return True
    if re.match(r"^in this unit\b", low) and len(low.split()) <= 12:
        return True
    return False


def _trim_beginning_noise_prefix(text: str) -> str:
    """
    Remove only the heading/noise prefix lines at chapter start, keeping intro prose.
    """
    lines = [ln for ln in text.splitlines()]
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            i += 1
            continue
        if _looks_like_pdf_noise_line(ln) or _line_is_beginning_noise(ln):
            i += 1
            continue
        # First meaningful non-noise line reached.
        break
    return "\n".join(lines[i:]).strip()


def _is_objective_heading_line(line: str) -> bool:
    low = normalize_text(line)
    if not low:
        return False
    return bool(
        re.match(r"^(learning\s+)?objectives?\b", low)
        or re.match(r"^learning\s+outcomes?\b", low)
    )


def _remove_objective_bullets(text: str) -> str:
    """
    Post-processing guard:
    remove objective/outcome headings and bullet-list lines under them.
    """
    lines = text.splitlines()
    out: list[str] = []
    in_objective = False

    for raw in lines:
        s = raw.strip()
        if not s:
            if not in_objective:
                out.append(raw)
            continue

        if _is_objective_heading_line(s):
            in_objective = True
            continue

        if in_objective:
            if _is_bullet_line(s):
                continue
            if len(s.split()) <= 12 and not s.endswith("."):
                continue
            in_objective = False
            out.append(raw)
            continue

        out.append(raw)

    return "\n".join(out).strip()


def _drop_incomplete_tail_sentence(text: str) -> str:
    """
    If text ends with an incomplete sentence fragment, trim it away.
    Keeps content up to the last clear sentence-ending punctuation.
    """
    s = (text or "").strip()
    if not s:
        return ""
    if re.search(r"[.!?][\"\')\]]?\s*$", s):
        return s

    matches = list(re.finditer(r"[.!?][\"\')\]]?", s))
    if not matches:
        return s
    cut = matches[-1].end()
    return s[:cut].strip()


def _extract_introduction_content(page_text: str) -> str:
    """
    Deterministically extract INTRODUCTION section body from a page, if present.
    Returns empty string when not found.
    """
    lines = [ln.strip() for ln in page_text.splitlines()]
    if not lines:
        return ""

    # Drop obvious print/footer noise first.
    cleaned_lines = [ln for ln in lines if ln and not _looks_like_pdf_noise_line(ln)]
    if not cleaned_lines:
        return ""

    intro_idx = -1
    for i, ln in enumerate(cleaned_lines):
        if re.match(r"^(?:\d+(?:\.\d+)*)?\s*introduction\b", ln, re.IGNORECASE):
            intro_idx = i
            break
        if re.match(r"^\d+(?:\.\d+)*$", ln) and i + 1 < len(cleaned_lines):
            if re.match(r"^introduction\b", cleaned_lines[i + 1], re.IGNORECASE):
                intro_idx = i + 1
                break
        if "introduction" in normalize_text(ln).split():
            intro_idx = i
            break

    if intro_idx < 0:
        return ""

    body: list[str] = []
    for ln in cleaned_lines[intro_idx + 1 :]:
        if _is_objective_heading_line(ln):
            continue
        # Stop at likely next section heading.
        if re.match(r"^\d+(?:\.\d+)+\s+[A-Z][A-Z\s\-]{2,}$", ln):
            break
        if re.match(r"^[A-Z][A-Z\s\-]{6,}$", ln) and len(ln.split()) <= 6:
            break
        body.append(ln)

    text = "\n".join(body).strip()
    if not text:
        return ""
    return f"Introduction : {text}"


def _is_intro_objective_page(page_text: str) -> bool:
    """
    Heuristic for chapter introduction page where objective bullets should be removed.
    """
    low = normalize_text(page_text)
    has_objective = (
        "learning objective" in low
        or "learning objectives" in low
        or "learning outcome" in low
        or "learning outcomes" in low
        or "in this unit" in low
    )
    has_intro = "introduction" in low or bool(re.search(r"\b1\.1\b", low))
    return has_objective and has_intro


def _looks_like_objective_bullet_block(text: str) -> bool:
    """
    Detect objective-only bullet blocks where the heading might be missing.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False

    bullet_lines = [ln for ln in lines if _is_bullet_line(ln)]
    if len(bullet_lines) < 3:
        return False
    if len(bullet_lines) < max(3, int(0.6 * len(lines))):
        return False

    # Objective bullets are usually short noun phrases, not full sentences.
    phrase_like = sum(1 for ln in bullet_lines if len(ln.split()) <= 12 and not ln.endswith("."))
    if phrase_like < max(3, int(0.7 * len(bullet_lines))):
        return False

    low = normalize_text(text)
    objective_terms = (
        "historical background",
        "role of",
        "concept of",
        "calculation of",
        "electrostatic potential",
        "electric field",
        "coulomb law",
        "superposition principle",
    )
    return any(t in low for t in objective_terms)


def _is_low_information_block(text: str) -> bool:
    s = text.strip()
    if not s:
        return True
    if s in {"+", "-", "=", "±"}:
        return True
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return True
    first = lines[0]
    if re.match(r"^unit\s+\d+\b", first.lower()) and len(lines) <= 3:
        return True
    if len(first) <= 3 and sum(ch.isalpha() for ch in first) <= 2:
        if len(lines) <= 3 or not any(len(ln.split()) >= 5 for ln in lines[1:]):
            return True
    if len(lines) <= 2 and all(_looks_like_math_or_noise(ln) for ln in lines):
        return True
    alpha = sum(ch.isalpha() for ch in s)
    if alpha < 25:
        return True
    return False


def _best_block_title_seed(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        n = re.sub(r"\s+", " ", ln).strip()
        if len(n) < 4:
            continue
        if re.match(r"^unit\s+\d+\b", n.lower()):
            continue
        if _looks_like_math_or_noise(n):
            continue
        return n
    return re.sub(r"\s+", " ", (lines[0] if lines else "Block")).strip()


def _merge_small_blocks(
    blocks: list[dict[str, Any]],
    *,
    small_words_threshold: int = 40,
    max_merge_blocks: int = 5,
) -> list[dict[str, Any]]:
    """
    Keep one-paragraph blocks by default.
    Merge tiny adjacent non-bullet blocks (up to max_merge_blocks) for better context.
    """
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        words = len(b["text"].split())
        if b["is_bullet"] or words >= small_words_threshold:
            out.append(b)
            i += 1
            continue
        merged = [b["text"]]
        j = i + 1
        while j < len(blocks) and len(merged) < max_merge_blocks:
            nb = blocks[j]
            if nb["is_bullet"]:
                break
            merged.append(nb["text"])
            if sum(len(x.split()) for x in merged) >= small_words_threshold:
                j += 1
                break
            j += 1
        out.append({"text": "\n\n".join(merged).strip(), "is_bullet": False})
        i = j
    return out


def chapter_sections_pagewise(
    page_texts: list[str],
    span: ChapterSpan,
    *,
    endpoint_client: VertexEndpointClient | None = None,
    use_llm_page_segmentation: bool = False,
    only_page_no: int | None = None,
    min_body_chars: int = 80,
) -> list[dict[str, Any]]:
    """
    Build sequential page-wise sections:
    - page by page
    - one paragraph at a time
    - bullet points grouped together
    - skip summary/objectives at beginning of chapter
    """
    sections: list[dict[str, Any]] = []
    section_id = 1
    seen_real_content = False

    for page_idx in range(span.start_page, span.end_page + 1):
        page_no = page_idx + 1
        if only_page_no is not None and page_no != only_page_no:
            continue
        if endpoint_client is not None and use_llm_page_segmentation:
            page_blocks = _llm_page_blocks(
                endpoint_client=endpoint_client,
                chapter_name=span.chapter_name,
                page_no=page_no,
                page_text=page_texts[page_idx],
            )
        else:
            page_blocks = _page_blocks(page_texts[page_idx])
        page_blocks = _merge_small_blocks(page_blocks, max_merge_blocks=5)
        page_block_index = 0
        for block in page_blocks:
            txt = block["text"].strip()
            if (page_idx - span.start_page) <= 1 and not seen_real_content:
                # Do not drop entire mixed blocks; trim objective/summary prefix only.
                txt = _trim_beginning_noise_prefix(txt)
                if not txt:
                    continue
                # Guard: objective-only bullet list may survive if heading was lost by LLM.
                if _looks_like_objective_bullet_block(txt):
                    continue
            if len(txt) < min_body_chars:
                continue
            if _is_low_information_block(txt):
                continue
            if (page_idx - span.start_page) <= 1 and not seen_real_content:
                first_line = next((ln.strip() for ln in txt.splitlines() if ln.strip()), "")
                # Skip only if the remaining block itself is still a pure heading/noise block.
                if first_line and _line_is_beginning_noise(first_line) and len(txt.split()) < 35:
                    continue
            seen_real_content = True
            page_block_index += 1
            seed = _best_block_title_seed(txt)
            if len(seed) > 60:
                seed = seed[:57] + "..."
            title = f"Page {page_no} - Block {page_block_index}: {seed}"
            sections.append(
                {
                    "section_id": section_id,
                    "title": title,
                    "content": txt,
                    "status": "pending",
                    "teacher_explanation": None,
                    "page_no": page_no,
                }
            )
            section_id += 1
    return sections


def _normalize_line_for_heading(line: str) -> str:
    s = line.strip()
    s = re.sub(r"[\s\u2000-\u200f\u202f\ufeff]+", " ", s).strip()
    return s


def _looks_like_body_sentence(line: str) -> bool:
    """First line of a paragraph — not a real section heading."""
    s = line.strip()
    if len(s) > 95:
        return True
    if s.endswith(".") and len(s) > 30:
        return True
    low = s.lower()
    prefixes = (
        "the ",
        "in ",
        "this ",
        "a ",
        "an ",
        "when ",
        "if ",
        "consider ",
        "suppose ",
        "from ",
        "note ",
        "hence ",
        "therefore ",
        "since ",
        "by ",
        "calculate ",
        "compute ",
        "for ",
        "but ",
        "however ",
        "according ",
        "similarly ",
        "using ",
        "apply ",
        "what ",
        "how ",
        "why ",
        "which ",
        "let ",
        "with ",
        "without ",
        "most ",
        "some ",
        "many ",
        "each ",
        "every ",
        "two ",
        "three ",
        "earth ",
        "now ",
        "here ",
        "there ",
        "these ",
        "those ",
        "it ",
        "they ",
        "we ",
        "after ",
        "before ",
        "during ",
        "electricity ",
        "electromagnetism ",
        "benjamin ",
        "franklin ",
        "important ",
        "following ",
        "both ",
        "non ",
    )
    return any(low.startswith(p) for p in prefixes)


def _looks_like_math_or_noise(line: str) -> bool:
    s = line.strip()
    if s.startswith("="):
        return True
    if re.match(r"^[=+\-×*/().\s0-9A-Za-z^–−]+$", s):
        tokens = s.split()
        has_operator = any(op in s for op in ("=", "×", "+", "-", "/", "^"))
        if has_operator and len(tokens) <= 8:
            alpha_tokens = sum(1 for t in tokens if re.search(r"[A-Za-z]", t))
            if alpha_tokens <= 3:
                return True
    if len(s) <= 15:
        if re.match(r"^[A-Za-z]{1,4}\d+\s*$", s):
            return True
        if re.match(r"^[A-Za-z0-9∫∑×·±]{1,12}\s*$", s) and not re.search(
            r"[aeiouAEIOU]{2,}", s
        ):
            return True
        if len(s.split()) == 1 and len(s) <= 4 and s.isalpha() and s.islower():
            return True
    return False


def _is_strong_section_heading(line: str) -> bool:
    """
    Only treat lines as section boundaries if they look like real textbook headings:
    numbered subsections (1.1.1 …), EXAMPLE blocks, multi-word ALL CAPS banners, Solution.
    Deliberately ignores arbitrary Title Case lines (those are usually sentence starts in PDFs).
    """
    s = _normalize_line_for_heading(line)
    if not s or len(s) > 160:
        return False
    if _looks_like_math_or_noise(s):
        return False
    if _looks_like_body_sentence(s):
        return False

    # 1.1, 1.1.1, 1.3.2 … (Tamil Nadu / NCERT style)
    m_num = re.match(r"^(\d+\.\d+(?:\.\d+)?)\s+(.+)$", s)
    if m_num:
        rest = m_num.group(2).strip()
        if not rest or len(s.split()) > 16:
            return False
        if rest[0].isdigit() or rest[0] in "=+-×*/^":
            return False
        if not re.search(r"[A-Za-z]", rest):
            return False
        if _looks_like_math_or_noise(rest):
            return False
        return True

    # EXAMPLE 1.13
    if re.match(r"^EXAMPLE\s+[\d\.]+\s*", s, re.IGNORECASE):
        return len(s) < 100

    # Multi-word ALL CAPS section banners (SUMMARY, CONCEPT MAP, …)
    words = s.split()
    if len(words) >= 2 and s.isupper() and 8 <= len(s) <= 72:
        return True

    return False


def _section_display_title(first_line: str) -> str:
    """Human-readable title: prefer the numbered heading line, trimmed."""
    s = _normalize_line_for_heading(first_line)
    if len(s) > 120:
        return s[:117] + "..."
    return s


def split_chapter_into_sections(chapter_text: str, min_body_chars: int = 80) -> list[dict[str, Any]]:
    """
    Split one chapter's plain text into subsections with titles.
    Titles come only from strong headings (numbered subsections, EXAMPLE, etc.), not body sentences.
    Each item: section_id, title, content (full text for that block), status 'pending'.
    """
    lines = [ln.rstrip() for ln in chapter_text.splitlines()]
    chunks: list[tuple[str, list[str]]] = []
    current_title = "Introduction"
    current_lines: list[str] = []

    def flush():
        nonlocal current_title, current_lines
        body = "\n".join(current_lines).strip()
        if len(body) >= min_body_chars:
            chunks.append((current_title, current_lines.copy()))
        current_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_lines:
                current_lines.append("")
            continue
        if _is_strong_section_heading(stripped) and current_lines:
            flush()
            current_title = _section_display_title(stripped)
            current_lines = [stripped]
        else:
            if not current_lines and not chunks:
                current_title = (
                    _section_display_title(stripped)
                    if _is_strong_section_heading(stripped)
                    else "Introduction"
                )
            current_lines.append(stripped)

    flush()

    if not chunks and chapter_text.strip():
        chunks.append(("Full chapter", chapter_text.strip().splitlines()))

    out: list[dict[str, Any]] = []
    for i, (title, clines) in enumerate(chunks, start=1):
        content = "\n".join(clines).strip()
        if len(content) < min_body_chars:
            continue
        out.append(
            {
                "section_id": i,
                "title": title[:500],
                "content": content,
                "status": "pending",
                "teacher_explanation": None,
            }
        )

    # Coherence pass: merge tiny fragments into neighboring sections.
    merged: list[dict[str, Any]] = []
    min_words_for_standalone = 70
    for sec in out:
        wc = len(sec["content"].split())
        if wc >= min_words_for_standalone or not merged:
            merged.append(sec)
            continue
        merged[-1]["content"] = (
            merged[-1]["content"].rstrip() + "\n\n" + sec["content"].lstrip()
        ).strip()
        # Keep earlier title, but include tiny subsection label in text itself.
        merged[-1]["teacher_explanation"] = None
    for i, sec in enumerate(merged, start=1):
        sec["section_id"] = i
    return merged


def split_sections(chapter_text: str) -> list[str]:
    """Legacy: content-only chunks (for older callers)."""
    detailed = split_chapter_into_sections(chapter_text)
    out = [d["content"] for d in detailed]
    if not out and chapter_text.strip():
        t = chapter_text.strip()
        if len(t) > 60:
            return [t]
    return [s for s in out if len(s) > 60]


def load_model(model_key: str):
    if model_key not in MODEL_CHOICES:
        valid = ", ".join(MODEL_CHOICES.keys())
        raise ValueError(f"Invalid model_key '{model_key}'. Choose one of: {valid}")

    model_id = MODEL_CHOICES[model_key]
    # Helps reduce CUDA allocator fragmentation on constrained VRAM machines.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    return model_id, tokenizer, model


def generate_teacher_explanation(
    chapter_name: str,
    section_text: str,
    *,
    endpoint_client: VertexEndpointClient | None = None,
    section_title: str | None = None,
    previous_context: str | None = None,
    max_new_tokens: int = 2048,
) -> str:
    if endpoint_client is None:
        raise ValueError("endpoint_client is required. Local fallback is disabled.")

    def _strip_prompt_echo(text: str, prompt_text: str) -> str:
        t = (text or "").strip()
        if not t:
            return ""
        matches = list(re.finditer(r"(?is)\boutput\s*:\s*", t))
        if matches:
            t = t[matches[-1].end() :].strip()
        if t.startswith(prompt_text.strip()):
            t = t[len(prompt_text.strip()) :].strip()
        # If model echoed instructions/text blocks, keep likely explanation tail.
        if "\nText:\n" in t:
            t = t.split("\nText:\n", 1)[-1].strip()
        if "\nSubsection source text:\n" in t:
            t = t.split("\nSubsection source text:\n", 1)[-1].strip()
        # If introduction heading exists, cut from there.
        m_intro = re.search(r"(?is)\bintroduction\s*:", t)
        if m_intro and m_intro.start() > 0:
            t = t[m_intro.start() :].strip()
        return t

    def _looks_low_quality(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True
        words = re.findall(r"[A-Za-z]+", t)
        if len(words) < 60:
            return True
        low = t.lower()
        if "prompt:" in low:
            return True
        if "output:" in low:
            return True
        if "strict output rules" in low:
            return True
        if low.count(" la ") >= 4:
            return True
        # Repeated nonsense pattern seen from endpoint failures.
        if "examplesexamples" in low or "lexamples" in low:
            return True
        uniq_ratio = len(set(w.lower() for w in words)) / max(len(words), 1)
        if uniq_ratio < 0.28:
            return True
        return False

    def _dynamic_fallback_explanation(source_text: str, title: str | None) -> str:
        src = re.sub(r"\s+", " ", (source_text or "")).strip()
        if not src:
            return (
                "Introduction\n"
                "This section could not be generated from the model response. "
                "Please retry once for a richer explanation.\n\n"
                "Quick revision:\n"
                "- The source text for this section was empty.\n"
                "- Re-run this section to get a full explanation.\n"
            ).strip()

        raw_title = (title or "").strip()
        topic = raw_title
        topic = re.sub(r"^Page\s+\d+\s*-\s*Block\s+\d+\s*:\s*", "", topic, flags=re.IGNORECASE)
        topic = re.sub(r"^Page\s+\d+\s*-\s*Cleaned\s+content\s*:?\s*", "", topic, flags=re.IGNORECASE)
        if not topic:
            topic = "this topic"

        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", src)
            if s.strip() and len(s.strip().split()) >= 5
        ]
        if not sentences:
            sentences = [src]

        # Keep it concise-but-complete, and deterministic.
        core = sentences[:6]
        p1 = " ".join(core[:2]).strip()
        p2 = " ".join(core[2:4]).strip()
        p3 = " ".join(core[4:6]).strip()
        paras = [p for p in [p1, p2, p3] if p]

        bullets = []
        for s in core[:4]:
            b = s
            if len(b) > 180:
                b = b[:177].rstrip() + "..."
            bullets.append(b)

        out_lines = [
            "Introduction",
            f"This subsection is about {topic}.",
        ]
        if paras:
            out_lines.append(" ".join(paras[:1]))
        if len(paras) >= 2:
            out_lines.append(paras[1])
        if len(paras) >= 3:
            out_lines.append(paras[2])
        out_lines.append("Quick revision:")
        for b in bullets:
            out_lines.append(f"- {b}")

        return "\n\n".join(out_lines[:4]) + ("\n\n" if len(out_lines) > 4 else "") + "\n".join(out_lines[4:])

    # Enough context for the model; trim only if extremely long.
    trimmed = _drop_incomplete_tail_sentence(section_text.strip())
    sub = f"\nSubsection: {section_title}\n" if section_title else ""
    prev = ""
    if previous_context and previous_context.strip():
        prev_trimmed = re.sub(r"\s+", " ", previous_context.strip())
        if len(prev_trimmed) > 1200:
            prev_trimmed = prev_trimmed[:1200].rstrip() + "..."
        prev = (
            "\nPrevious context (for continuity only; focus mainly on current subsection):\n"
            f"{prev_trimmed}\n"
        )
    prompt = f"""You are an expert teacher for 12th standard students.

Chapter: {chapter_name}
{sub}
Teach ONLY the ideas in this subsection (do not jump ahead to later chapters).
Focus primarily on the CURRENT subsection text. Use previous context only to connect flow.
{prev}

Your explanation must:
- Start from basics: define terms, build intuition, then add detail.
- Use clear structure: several paragraphs with short headings or numbered steps where helpful.
- Use simple English; add one or two concrete examples (everyday or textbook-style) when they help.
- Connect to why this matters before formulas, if the text includes equations.
- Be thorough: a student who missed class should still follow.

End with a short "Quick revision" block: 4–6 bullet points covering the main ideas.

Subsection source text:
{trimmed}
"""
    out = endpoint_client.generate_text(prompt, max_new_tokens=max_new_tokens)
    cleaned = _strip_prompt_echo(out, prompt)
    if not _looks_low_quality(cleaned):
        return cleaned

    retry_prompt = f"""You are a physics teacher.
Explain this text clearly for a 12th standard student.

Strict output rules:
- Return only explanation text.
- Do not include 'Prompt:' or 'Output:'.
- Start with a short heading: Introduction
- Then 3-5 short paragraphs in simple English.
- End with 'Quick revision:' and 4 bullet points.

Text:
{trimmed}
"""
    out2 = endpoint_client.generate_text(retry_prompt, max_new_tokens=min(max_new_tokens, 1400))
    cleaned2 = _strip_prompt_echo(out2, retry_prompt)
    if not _looks_low_quality(cleaned2):
        return cleaned2

    # Dynamic deterministic fallback when endpoint returns malformed content.
    return _dynamic_fallback_explanation(trimmed, section_title)


def plan_teaching_session(
    pdf_path: str | Path,
    chapter_names: List[str],
    *,
    endpoint_client: VertexEndpointClient | None = None,
    use_llm_page_segmentation: bool = False,
    only_page_no: int | None = None,
    skip_toc_pages: bool = True,
    min_page: int = 0,
) -> dict[str, Any]:
    """
    Compatibility wrapper for direct page planning.
    Uses first chapter name + only_page_no and delegates to plan_page_session.
    """
    _ = skip_toc_pages
    _ = min_page
    _ = use_llm_page_segmentation

    if not chapter_names:
        raise ValueError("chapter_names must include at least one chapter name")
    chapter_name = next((c.strip() for c in chapter_names if c and c.strip()), "")
    if not chapter_name:
        raise ValueError("chapter_names must include at least one non-empty chapter name")
    if only_page_no is None:
        raise ValueError("only_page_no is required in direct page planning mode")

    return plan_page_session(
        pdf_path=pdf_path,
        chapter_name=chapter_name,
        page_no=only_page_no,
        endpoint_client=endpoint_client,
        use_llm_page_segmentation=True,
    )


def plan_page_session(
    pdf_path: str | Path,
    chapter_name: str,
    page_no: int,
    *,
    endpoint_client: VertexEndpointClient | None = None,
    use_llm_page_segmentation: bool = True,
) -> dict[str, Any]:
    """
    Direct-input page planner:
    - read only the requested page
    - print raw page text
    - ask LLM to return only cleaned page content
      (exclude learning objectives/objectives content)
    """
    if not chapter_name or not chapter_name.strip():
        raise ValueError("chapter_name must not be empty")
    page_texts = extract_pdf_page_texts(pdf_path)
    if page_no < 1 or page_no > len(page_texts):
        raise ValueError(
            f"page_no={page_no} is out of range. Valid PDF page range is 1..{len(page_texts)}"
        )
    if endpoint_client is None:
        raise ValueError("endpoint_client is required for plan_page_session")

    raw_page_text = page_texts[page_no - 1].strip()
    print(raw_page_text)
    is_intro_page = _is_intro_objective_page(raw_page_text)

    clipped = raw_page_text
    prompt = f"""You are cleaning one textbook page.
Return ONLY the actual concept/content from this page.

Rules:
1) Keep actual explanatory paragraph content from the page.
2) Do not add explanations, summaries, headings, JSON, markdown, or extra text.
3) Output plain cleaned page content only.
4) Do not repeat the prompt or the source page text.
"""
    if is_intro_page:
        prompt += """
5) This is the chapter-introduction page.
6) Remove "Learning Objectives", "Objectives", "Learning Outcomes" headings.
7) Remove every bullet/line under those objective headings.
8) If there is an INTRODUCTION heading, return only the INTRODUCTION paragraph content.
"""
    else:
        prompt += """
5) This is NOT the introduction page.
6) Keep numbered/bulleted points like (i), (ii), (iii) when they are actual content.
"""
    prompt += f"""

Chapter: {chapter_name.strip()}
Page number: {page_no}

Page text:
{clipped}
"""
    llm_raw = endpoint_client.generate_text(prompt, max_new_tokens=4096).strip()

    # Some endpoints may echo "Prompt: ... Output: ...". Keep only the actual output text.
    cleaned_content = llm_raw
    m_out = re.search(r"(?is)\boutput\s*:\s*", cleaned_content)
    if m_out:
        cleaned_content = cleaned_content[m_out.end() :].strip()
    if cleaned_content.startswith(prompt.strip()):
        cleaned_content = cleaned_content[len(prompt.strip()) :].strip()

    # Only intro page gets objective-heading bullet cleanup.
    if is_intro_page:
        cleaned_content = _remove_objective_bullets(cleaned_content).strip()
        cleaned_content = _trim_beginning_noise_prefix(cleaned_content).strip()
    cleaned_content = _drop_incomplete_tail_sentence(cleaned_content)

    intro_only = _extract_introduction_content(raw_page_text) if is_intro_page else ""
    if intro_only:
        cleaned_content = _drop_incomplete_tail_sentence(intro_only)

    if not cleaned_content:
        fallback_raw = raw_page_text
        if is_intro_page:
            fallback_raw = _trim_beginning_noise_prefix(_remove_objective_bullets(fallback_raw)).strip()
        cleaned_content = fallback_raw
        cleaned_content = _drop_incomplete_tail_sentence(cleaned_content)
    sections = [
        {
            "section_id": 1,
            "title": f"Page {page_no} - Cleaned content",
            "content": cleaned_content,
            "status": "pending",
            "teacher_explanation": None,
            "page_no": page_no,
        }
    ]
    return {
        "pdf_path": str(pdf_path),
        "chapters": [
            {
                "sections": sections,
            }
        ],
    }


def explain_session_section(
    session: dict[str, Any],
    chapter_index: int,
    section_id: int,
    endpoint_client: VertexEndpointClient | None = None,
    max_new_tokens: int = 2048,
) -> dict[str, Any]:
    """
    Generate teacher text for one section by section_id (1-based), mark it done, return updated section dict.
    """
    ch = session["chapters"][chapter_index]
    sections = ch["sections"]
    sec = next((s for s in sections if s["section_id"] == section_id), None)
    if sec is None:
        raise ValueError(f"No section_id={section_id} in chapter_index={chapter_index}")

    prev_context = ""
    prev = next((s for s in sections if s["section_id"] == section_id - 1), None)
    if prev is not None:
        prev_page = prev.get("page_no")
        prev_title = prev.get("title", "")
        prev_expl = (prev.get("teacher_explanation") or "").strip()
        prev_content = (prev.get("content") or "").strip()
        prev_seed = prev_expl if prev_expl else prev_content
        if prev_seed:
            prev_context = f"Previous section (page {prev_page}) - {prev_title}: {prev_seed}"

    explanation = generate_teacher_explanation(
        endpoint_client=endpoint_client,
        chapter_name=ch.get("chapter_name", "Selected topic"),
        section_text=sec["content"],
        section_title=sec.get("title"),
        previous_context=prev_context,
        max_new_tokens=max_new_tokens,
    )
    sec["teacher_explanation"] = explanation
    sec["status"] = "done"
    return sec


def save_session(session: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)


def load_session(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sections_status_markdown(session: dict[str, Any], chapter_index: int = 0) -> str:
    """Markdown table for notebook display: # | Title | Status | Words |"""
    ch = session["chapters"][chapter_index]
    lines = [
        "| # | Title | Status | Words |",
        "|---|--------|--------|-------|",
    ]
    for s in ch["sections"]:
        title = (s.get("title") or "")[:60].replace("|", "/")
        st = s.get("status", "pending")
        wc = len((s.get("content") or "").split())
        lines.append(f"| {s['section_id']} | {title} | {st} | {wc} |")
    return "\n".join(lines)


def chapter_ranges_markdown(session: dict[str, Any]) -> str:
    """Markdown table for detected chapter page ranges (1-based)."""
    lines = [
        "| Chapter | Start Page | End Page | Total Pages |",
        "|---------|------------|----------|-------------|",
    ]
    for ch in session.get("chapters", []):
        sp = int(ch.get("start_page", 0) or 0)
        ep = int(ch.get("end_page", 0) or 0)
        total = max(0, ep - sp + 1) if sp and ep else 0
        lines.append(f"| {ch.get('chapter_name', '')} | {sp} | {ep} | {total} |")
    return "\n".join(lines)


def build_chapter_teaching_notes(
    pdf_path: str | Path,
    chapter_names: List[str],
    endpoint_client: VertexEndpointClient | None = None,
    max_sections_per_chapter: int = 6,
    *,
    skip_toc_pages: bool = True,
    min_page: int = 0,
) -> dict:
    page_texts = extract_pdf_page_texts(pdf_path)
    spans = locate_chapter_ranges(
        page_texts,
        chapter_names,
        skip_toc_pages=skip_toc_pages,
        min_page=min_page,
    )

    result = {"pdf_path": str(pdf_path), "chapters": []}
    for span in spans:
        chapter_text = extract_chapter_text(page_texts, span)
        detailed = split_chapter_into_sections(chapter_text)[:max_sections_per_chapter]

        chapter_out = {
            "chapter_name": span.chapter_name,
            "start_page": span.start_page + 1,
            "end_page": span.end_page + 1,
            "sections": [],
        }

        for sec in detailed:
            explanation = generate_teacher_explanation(
                endpoint_client=endpoint_client,
                chapter_name=span.chapter_name,
                section_text=sec["content"],
                section_title=sec.get("title"),
            )
            chapter_out["sections"].append(
                {
                    "section_id": sec["section_id"],
                    "title": sec["title"],
                    "section_source_excerpt": sec["content"][:1200],
                    "teacher_explanation": explanation,
                }
            )

        result["chapters"].append(chapter_out)

    return result
