from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from teacher_pdf_pipeline import VertexEndpointClient


DIFFICULTIES = ("easy", "medium", "hard")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str | Path, default: Any) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_json_value(text: str) -> Any | None:
    if not text:
        return None
    raw = text.strip()
    if "```" in raw:
        start = raw.find("```")
        end = raw.rfind("```")
        if end > start:
            fenced = raw[start + 3 : end].strip()
            if fenced.lower().startswith("json"):
                fenced = fenced[4:].strip()
            raw = fenced
    try:
        return json.loads(raw)
    except Exception:
        pass
    # fallback: first {...} or first [...]
    obj_start = raw.find("{")
    obj_end = raw.rfind("}")
    if obj_start >= 0 and obj_end > obj_start:
        try:
            return json.loads(raw[obj_start : obj_end + 1])
        except Exception:
            pass
    arr_start = raw.find("[")
    arr_end = raw.rfind("]")
    if arr_start >= 0 and arr_end > arr_start:
        try:
            return json.loads(raw[arr_start : arr_end + 1])
        except Exception:
            pass
    return None


def _difficulty_sequence(total: int) -> list[str]:
    if total <= 0:
        return []
    # Gradual rise: easy -> medium -> hard
    n_easy = max(1, total // 3)
    n_medium = max(1, total // 3)
    n_hard = total - n_easy - n_medium
    if n_hard <= 0:
        n_hard = 1
        if n_medium > 1:
            n_medium -= 1
        else:
            n_easy = max(1, n_easy - 1)
    seq = (["easy"] * n_easy) + (["medium"] * n_medium) + (["hard"] * n_hard)
    return seq[:total]


def _normalize_option_map(opts: Any) -> dict[str, str] | None:
    if isinstance(opts, dict):
        out = {}
        for k in ("A", "B", "C", "D"):
            v = opts.get(k)
            if not isinstance(v, str) or not v.strip():
                return None
            out[k] = v.strip()
        return out
    if isinstance(opts, list) and len(opts) == 4 and all(isinstance(x, str) and x.strip() for x in opts):
        return {"A": opts[0].strip(), "B": opts[1].strip(), "C": opts[2].strip(), "D": opts[3].strip()}
    return None


def _normalize_answer(ans: Any) -> str | None:
    if isinstance(ans, str):
        t = ans.strip().upper()
        if t in {"A", "B", "C", "D"}:
            return t
    return None


def _normalize_mcqs(
    raw_mcqs: list[Any],
    *,
    page_no: int,
    source_section_ids: list[int],
    difficulty_seq: list[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw_mcqs):
        if not isinstance(item, dict):
            continue
        q = item.get("question")
        if not isinstance(q, str) or len(q.strip()) < 12:
            continue
        options = _normalize_option_map(item.get("options"))
        if options is None:
            continue
        ans = _normalize_answer(item.get("answer"))
        if ans is None:
            continue
        exp = item.get("explanation")
        explanation = exp.strip() if isinstance(exp, str) and exp.strip() else "Reasoning not provided."
        tag = item.get("concept_tag")
        concept_tag = tag.strip() if isinstance(tag, str) and tag.strip() else "conceptual-understanding"
        difficulty = difficulty_seq[min(i, len(difficulty_seq) - 1)] if difficulty_seq else "medium"
        out.append(
            {
                "question_id": f"p{page_no}_q{i + 1}",
                "difficulty": difficulty,
                "question": q.strip(),
                "options": options,
                "answer": ans,
                "explanation": explanation,
                "concept_tag": concept_tag,
                "source_section_ids": source_section_ids,
                "page_no": page_no,
            }
        )
    return out


def _build_prompt(
    *,
    chapter_name: str,
    page_no: int,
    source_text: str,
    total_questions: int,
    difficulty_seq: list[str],
) -> str:
    seq = ", ".join(difficulty_seq)
    return f"""You are an expert assessment designer for 12th-standard physics.
Create conceptual MCQs for the given chapter page.

Requirements:
1) Generate exactly {total_questions} MCQs.
2) Difficulty must progress in this exact order: {seq}.
3) Questions must test conceptual understanding (no pure formula plugging).
4) Each MCQ must have 4 options: A, B, C, D and exactly one correct answer.
5) Add a short explanation for why the answer is correct.
6) Keep language clear for class-12 students.
7) Return STRICT JSON only:
{{
  "mcqs": [
    {{
      "question": "...",
      "options": {{
        "A": "...",
        "B": "...",
        "C": "...",
        "D": "..."
      }},
      "answer": "A|B|C|D",
      "difficulty": "easy|medium|hard",
      "concept_tag": "...",
      "explanation": "..."
    }}
  ]
}}

Chapter: {chapter_name}
Page number: {page_no}

Teacher explanation source:
{source_text}
"""


def generate_mcqs_for_page(
    *,
    chapter_name: str,
    page_no: int,
    source_text: str,
    source_section_ids: list[int],
    endpoint_client: VertexEndpointClient,
    questions_per_page: int = 6,
    max_new_tokens: int = 2800,
) -> list[dict[str, Any]]:
    difficulty_seq = _difficulty_sequence(questions_per_page)
    prompt = _build_prompt(
        chapter_name=chapter_name,
        page_no=page_no,
        source_text=source_text,
        total_questions=questions_per_page,
        difficulty_seq=difficulty_seq,
    )
    raw = endpoint_client.generate_text(prompt, max_new_tokens=max_new_tokens).strip()
    parsed = _extract_json_value(raw)
    items: list[Any] = []
    if isinstance(parsed, dict) and isinstance(parsed.get("mcqs"), list):
        items = parsed["mcqs"]
    elif isinstance(parsed, list):
        items = parsed
    mcqs = _normalize_mcqs(
        items,
        page_no=page_no,
        source_section_ids=source_section_ids,
        difficulty_seq=difficulty_seq,
    )
    # One strict retry if count is short/invalid.
    if len(mcqs) < questions_per_page:
        retry_prompt = (
            prompt
            + "\nIMPORTANT RETRY: Your previous output was invalid. "
            "Return only valid JSON with exactly the requested number of MCQs."
        )
        raw2 = endpoint_client.generate_text(retry_prompt, max_new_tokens=max_new_tokens).strip()
        parsed2 = _extract_json_value(raw2)
        items2: list[Any] = []
        if isinstance(parsed2, dict) and isinstance(parsed2.get("mcqs"), list):
            items2 = parsed2["mcqs"]
        elif isinstance(parsed2, list):
            items2 = parsed2
        mcqs = _normalize_mcqs(
            items2,
            page_no=page_no,
            source_section_ids=source_section_ids,
            difficulty_seq=difficulty_seq,
        )
    return mcqs[:questions_per_page]


def get_or_generate_chapter_mcqs(
    *,
    chapter_name: str,
    teacher_cache_path: str | Path,
    mcq_cache_path: str | Path,
    endpoint_client: VertexEndpointClient,
    page_nos: list[int] | None = None,
    questions_per_page: int = 6,
    max_new_tokens: int = 2800,
    force_regenerate: bool = False,
) -> dict[str, Any]:
    teacher_db = _load_json(teacher_cache_path, default={})
    chapter_data = teacher_db.get(chapter_name, {})
    pages = chapter_data.get("pages", {})
    if not isinstance(pages, dict) or not pages:
        raise ValueError(f"No teacher cache pages found for chapter '{chapter_name}'")

    selected_pages = sorted(
        [int(k) for k in pages.keys() if str(k).isdigit() and (page_nos is None or int(k) in page_nos)]
    )
    if not selected_pages:
        raise ValueError("No matching pages found in teacher cache for requested chapter/page filter.")

    mcq_db = _load_json(mcq_cache_path, default={})
    chapter_bucket = mcq_db.setdefault(chapter_name, {"pages": {}})

    results: list[dict[str, Any]] = []
    for p in selected_pages:
        page_key = str(p)
        section_list = pages.get(page_key, {}).get("sections", [])
        source_sections = [s for s in section_list if (s.get("teacher_explanation") or "").strip()]
        if not source_sections:
            results.append(
                {
                    "page_no": p,
                    "status": "skipped_no_teacher_explanation",
                    "mcq_count": 0,
                    "from_cache": False,
                }
            )
            continue

        source_text = "\n\n".join((s.get("teacher_explanation") or "").strip() for s in source_sections)
        source_hash = _sha256(source_text)
        source_section_ids = [int(s.get("section_id", 0) or 0) for s in source_sections if s.get("section_id")]

        existing = chapter_bucket["pages"].get(page_key, {})
        cached_mcqs = existing.get("mcqs", [])
        is_cache_hit = (
            not force_regenerate
            and existing.get("source_hash") == source_hash
            and isinstance(cached_mcqs, list)
            and len(cached_mcqs) >= questions_per_page
        )

        if is_cache_hit:
            mcqs = cached_mcqs[:questions_per_page]
            status = "cache_hit"
        else:
            mcqs = generate_mcqs_for_page(
                chapter_name=chapter_name,
                page_no=p,
                source_text=source_text,
                source_section_ids=source_section_ids,
                endpoint_client=endpoint_client,
                questions_per_page=questions_per_page,
                max_new_tokens=max_new_tokens,
            )
            status = "generated" if mcqs else "generation_failed"
            chapter_bucket["pages"][page_key] = {
                "mcqs": mcqs,
                "source_hash": source_hash,
                "updated_at": _now_iso(),
                "model_info": {
                    "questions_per_page": questions_per_page,
                },
            }

        results.append(
            {
                "page_no": p,
                "status": status,
                "from_cache": is_cache_hit,
                "mcq_count": len(mcqs),
                "mcqs": mcqs,
            }
        )

    _save_json(mcq_cache_path, mcq_db)
    return {
        "chapter_name": chapter_name,
        "teacher_cache_path": str(teacher_cache_path),
        "mcq_cache_path": str(mcq_cache_path),
        "pages": results,
    }


def mcq_status_markdown(result: dict[str, Any]) -> str:
    lines = [
        "| Page | Status | Cache | MCQs |",
        "|------|--------|-------|------|",
    ]
    for p in result.get("pages", []):
        page_no = p.get("page_no", "")
        status = p.get("status", "")
        cache = "yes" if p.get("from_cache") else "no"
        count = p.get("mcq_count", 0)
        lines.append(f"| {page_no} | {status} | {cache} | {count} |")
    return "\n".join(lines)
