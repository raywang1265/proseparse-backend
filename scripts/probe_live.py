"""Probe the live Cloud Run service for /health, /analyze, and /voice.

Usage (from repo root, with venv active):

    python scripts/probe_live.py

Reads ANALYSIS_API_KEY and ANALYSIS_BASE_URL from .env (see .env.example).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

BASE_URL = os.getenv("ANALYSIS_BASE_URL", "").rstrip("/")
API_KEY = os.getenv("ANALYSIS_API_KEY", "")

SAMPLE_PARAGRAPHS = [
    {
        "block": 0,
        "text": (
            "The lighthouse stood pale against the bruised sky, and the wind "
            "was carried inland like a rumor nobody wanted to repeat."
        ),
    },
    {
        "block": 1,
        "text": (
            'Mara pressed her palm to the cold glass and listened to the gulls '
            'shrieking over the breakwater. "You came back," she said, not '
            "turning around."
        ),
    },
    {
        "block": 2,
        "text": (
            '"I never really left." Thomas muttered, and the words were '
            "swallowed by the salt-thick air."
        ),
    },
    # Tagless follow-up with only one known speaker → should land on UNKNOWN.
    {
        "block": 3,
        "text": '"Still waiting for an answer."',
    },
]


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def paragraph_by_block(block: int) -> str:
    for p in SAMPLE_PARAGRAPHS:
        if p["block"] == block:
            return p["text"]
    return ""


def summarize_spans(spans: list[dict]) -> list[dict]:
    """Compact span summary with a UTF-16 slice preview (BMP-safe for samples)."""
    out: list[dict] = []
    for sp in spans:
        block = sp.get("block")
        span = sp.get("span") or []
        preview = None
        if isinstance(block, int) and len(span) == 2:
            text = paragraph_by_block(block)
            start, end = span
            preview = text[start:end]
        out.append({"block": block, "span": span, "preview": preview})
    return out


def main() -> None:
    if not BASE_URL:
        fail("ANALYSIS_BASE_URL missing — copy .env.example to .env")
    if not API_KEY or API_KEY == "replace-me":
        fail("ANALYSIS_API_KEY missing or still placeholder — set it in .env")

    headers = {"Authorization": f"Bearer {API_KEY}"}
    timeout = httpx.Timeout(120.0, connect=30.0)

    print(f"Probing {BASE_URL}\n")

    with httpx.Client(base_url=BASE_URL, timeout=timeout) as client:
        # --- health (no auth) ---
        r = client.get("/health")
        print(f"GET /health -> {r.status_code} {r.text}")
        if r.status_code != 200:
            fail("/health did not return 200")

        # --- analyze ---
        analyze_body = {
            "sessionId": "probe-live",
            "batchIndex": 0,
            "paragraphs": SAMPLE_PARAGRAPHS,
        }
        r = client.post("/analyze", json=analyze_body, headers=headers)
        print(f"\nPOST /analyze -> {r.status_code}")
        if r.status_code != 200:
            fail(f"/analyze error: {r.text}")
        analyze = r.json()
        print(
            json.dumps(
                {
                    "batchIndex": analyze.get("batchIndex"),
                    "resultCount": len(analyze.get("results", [])),
                    "results": [
                        {
                            "block": item["block"],
                            "passiveCount": len(item.get("passive", [])),
                            "activeCount": len(item.get("active", [])),
                            "tagCount": len(item.get("tags", [])),
                            "counts": item.get("counts"),
                        }
                        for item in analyze.get("results", [])
                    ],
                },
                indent=2,
            )
        )

        # --- voice (may cold-start MiniLM; allow longer timeout) ---
        voice_body = {
            "sessionId": "probe-live",
            "batchIndex": 0,
            "paragraphs": SAMPLE_PARAGRAPHS,
        }
        r = client.post("/voice", json=voice_body, headers=headers)
        print(f"\nPOST /voice -> {r.status_code}")
        if r.status_code != 200:
            fail(f"/voice error: {r.text}")
        voice = r.json()
        characters = voice.get("characters", [])
        summary = {
            "batchIndex": voice.get("batchIndex"),
            "characters": [
                {
                    "name": ch["name"],
                    "vectorDim": len(ch.get("vector", [])),
                    "stylometry": ch.get("stylometry"),
                    "uniqueLemmaCount": len(ch.get("uniqueLemmas", [])),
                    "sampleLemmas": ch.get("uniqueLemmas", [])[:8],
                    "spanCount": len(ch.get("spans") or []),
                    "spans": summarize_spans(ch.get("spans") or []),
                }
                for ch in characters
            ],
        }
        print(json.dumps(summary, indent=2))

        # Soft checks so an old revision without spans is obvious.
        if characters and any("spans" not in ch for ch in characters):
            fail(
                "/voice characters missing `spans` — deploy the revision "
                "that returns dialogue spans"
            )
        named = [c for c in characters if c.get("name") != "UNKNOWN"]
        if named and not any(c.get("spans") for c in named):
            fail("/voice returned no dialogue spans for named speakers")

    print("\nOK: health, /analyze, and /voice all succeeded.")


if __name__ == "__main__":
    main()
