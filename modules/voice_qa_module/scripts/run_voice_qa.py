#!/usr/bin/env python3
"""CLI runner for end-to-end voice QA pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path("/home/bp/AiTutor")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.voice_qa_module.src.voice_qa_pipeline import (
    VoiceQAPipeline,
    VoiceQAPipelineConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ASR -> Gemma -> TTS pipeline.")
    parser.add_argument("--input-audio", required=True, help="Input WAV question path.")
    parser.add_argument("--output-audio", required=True, help="Output WAV answer path.")
    parser.add_argument(
        "--asr-language-code",
        default="en-US",
        help="ASR language code, e.g., en-US / ta-IN",
    )
    parser.add_argument(
        "--answer-language",
        default="English",
        help="Answer language for tutor response (English/Tamil/Mix).",
    )
    parser.add_argument(
        "--project-id",
        default="gcp-cap-dsml-core-dev",
        help="GCP project for Vertex Gemini TTS client.",
    )
    parser.add_argument(
        "--location",
        default="us-central1",
        help="GCP location for Vertex Gemini TTS client.",
    )
    parser.add_argument(
        "--dictionary-json",
        default="/home/bp/AiTutor/modules/teacher_module/outputs/chunk_summary_dictionary.json",
        help="Chunk-summary dictionary JSON path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="How many retrieved chunks to pass to Gemma.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = VoiceQAPipelineConfig(
        project_id=args.project_id,
        location=args.location,
        dictionary_json=Path(args.dictionary_json),
        retrieval_top_k=args.top_k,
    )
    pipeline = VoiceQAPipeline(cfg)
    out = pipeline.run(
        input_audio_path=args.input_audio,
        output_audio_path=args.output_audio,
        asr_language_code=args.asr_language_code,
        answer_language=args.answer_language,
        retrieval_top_k=args.top_k,
    )
    print(str(out.resolve()))


if __name__ == "__main__":
    main()

