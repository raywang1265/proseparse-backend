"""Tests for sensory prose analysis and /sensory endpoint."""

from __future__ import annotations

import os

# Auth is mandatory; set the key before the app lifespan runs (on TestClient enter).
os.environ.setdefault("ANALYSIS_API_KEY", "test-key")
AUTH = {"Authorization": "Bearer test-key"}

from typing import Any

import numpy as np
import pytest
import spacy
from fastapi.testclient import TestClient

import main as main_mod
from main import app
from sensory import analyze_sensory_paragraph, reset_anchor_cache
from sensory_lexicon import SENSES, get_lexicon
from voice import EMBEDDING_DIM

nlp = spacy.load("en_core_web_sm", disable=["ner"])
lexicon = get_lexicon()


class FakeEmbedder:
    """Deterministic stand-in for SentenceTransformer (no MiniLM download).

    Anchor texts from ``SENSE_ANCHORS`` get one-hot vectors on distinct dims so
    cosine classification is stable. Other sentences get a weak sight-leaning
    vector below the default threshold unless they contain sense keywords.
    """

    _KEYWORD_SENSE = {
        "visual": 0,
        "bright": 0,
        "color": 0,
        "shadow": 0,
        "sight": 0,
        "auditory": 1,
        "sound": 1,
        "noise": 1,
        "echo": 1,
        "hearing": 1,
        "tactile": 2,
        "texture": 2,
        "rough": 2,
        "soft": 2,
        "touch": 2,
        "feel": 2,
        "olfactory": 3,
        "smell": 3,
        "scent": 3,
        "odor": 3,
        "fragrance": 3,
        "aroma": 3,
        "reek": 3,
        "gustatory": 4,
        "taste": 4,
        "flavor": 4,
        "sweet": 4,
        "bitter": 4,
        "sour": 4,
        "salty": 4,
    }

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        out = []
        for s in sentences:
            vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            lower = s.lower()
            # Exact anchor strings → clean one-hot on that sense dim.
            from sensory import SENSE_ANCHORS

            matched_anchor = False
            for i, sense in enumerate(SENSES):
                if lower.strip() == SENSE_ANCHORS[sense].lower():
                    vec[i] = 1.0
                    matched_anchor = True
                    break
            if not matched_anchor:
                votes = [0.0] * 5
                for word, idx in self._KEYWORD_SENSE.items():
                    if word in lower:
                        votes[idx] += 1.0
                best = int(np.argmax(votes))
                if votes[best] > 0:
                    vec[best] = 0.9
                    vec[(best + 1) % 5] = 0.1
                else:
                    # Uniform vector → cosine vs any one-hot anchor ≈ 1/sqrt(dim) ≪ 0.35.
                    vec[:] = 1.0
            # L2-normalize to match normalize_embeddings=True.
            norm = float(np.linalg.norm(vec)) or 1.0
            out.append(vec / norm)
        return np.stack(out)


@pytest.fixture(autouse=True)
def _stub_sensory_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point /sensory at local spaCy + FakeEmbedder (no MiniLM download)."""
    fake = FakeEmbedder()
    reset_anchor_cache()
    monkeypatch.setattr(main_mod, "nlp", nlp)
    monkeypatch.setattr(main_mod, "embedder", fake)

    def _ensure_embedder() -> Any:
        return fake

    monkeypatch.setattr(main_mod, "ensure_embedder", _ensure_embedder)
    yield
    reset_anchor_cache()


def slice_utf16(text: str, start: int, end: int) -> str:
    """BMP-safe slice matching our sample texts."""
    return text[start:end]


def phrases(text: str, spans: list[list[int]]) -> list[str]:
    return [slice_utf16(text, s, e) for s, e in spans]


# --- Unit: analyze_sensory_paragraph -----------------------------------------


@pytest.mark.parametrize(
    "text,sense,must_contain",
    [
        (
            "The pungent aroma of burnt garlic filled the small kitchen.",
            "smell",
            ["pungent", "aroma"],
        ),
        (
            "A piercing scream echoed down the long empty corridor.",
            "sound",
            ["scream", "echoed"],
        ),
        (
            "The damp blanket felt unexpectedly soft against her skin.",
            "touch",
            ["damp", "soft"],
        ),
        (
            "The lemonade tasted sweet and faintly bitter on her tongue.",
            "taste",
            ["sweet", "bitter"],
        ),
        (
            "Bright light gleamed against the pale wall.",
            "sight",
            ["Bright", "gleamed"],
        ),
    ],
)
def test_clear_sensory_sentences(text: str, sense: str, must_contain: list[str]) -> None:
    result = analyze_sensory_paragraph(nlp(text), lexicon, FakeEmbedder())
    found = phrases(text, result[sense])
    joined = " ".join(found)
    for needle in must_contain:
        assert any(needle.lower() in p.lower() for p in found) or needle.lower() in joined.lower(), (
            f"expected {sense} trigger containing {needle!r}, got {found!r}"
        )
    assert result["counts"][sense] == len(result[sense])
    assert result["counts"][sense] >= 1


def test_non_sensory_sentence_yields_nothing() -> None:
    text = "She thought about the decision for a long time."
    result = analyze_sensory_paragraph(nlp(text), lexicon, FakeEmbedder())
    for sense in SENSES:
        assert result[sense] == [], f"{sense} unexpectedly non-empty: {result[sense]}"
        assert result["counts"][sense] == 0


def test_action_exposition_without_modifiers_skipped() -> None:
    text = "He walked toward the counter and asked for a cup of coffee."
    result = analyze_sensory_paragraph(nlp(text), lexicon, FakeEmbedder())
    total = sum(result["counts"][s] for s in SENSES)
    assert total == 0


def test_counts_match_array_lengths() -> None:
    text = (
        "The pungent aroma filled the kitchen. "
        "A scream echoed down the corridor. "
        "The damp cloth felt soft."
    )
    result = analyze_sensory_paragraph(nlp(text), lexicon, FakeEmbedder())
    for sense in SENSES:
        assert result["counts"][sense] == len(result[sense])


def test_no_cross_sense_span_overlap() -> None:
    text = (
        "The pungent aroma of sweet garlic filled the damp room "
        "while a loud scream echoed under bright light."
    )
    result = analyze_sensory_paragraph(nlp(text), lexicon, FakeEmbedder())
    occupied: list[tuple[int, int, str]] = []
    for sense in SENSES:
        for s, e in result[sense]:
            for os_, oe, other in occupied:
                assert not (s < oe and os_ < e), (
                    f"overlap between {sense} [{s},{e}] and {other} [{os_},{oe}]"
                )
            occupied.append((s, e, sense))


def test_debug_details_optional() -> None:
    text = "The pungent aroma filled the kitchen."
    plain = analyze_sensory_paragraph(nlp(text), lexicon, FakeEmbedder(), debug=False)
    assert "details" not in plain
    dbg = analyze_sensory_paragraph(nlp(text), lexicon, FakeEmbedder(), debug=True)
    assert "details" in dbg
    assert isinstance(dbg["details"], list)
    if dbg["details"]:
        d0 = dbg["details"][0]
        assert set(d0) >= {"span", "sense", "confidence", "tier"}
        assert d0["sense"] in SENSES


def test_tier1_without_embedder() -> None:
    text = "The pungent aroma filled the kitchen."
    result = analyze_sensory_paragraph(nlp(text), lexicon, embedder=None)
    assert result["counts"]["smell"] >= 1
    assert any("aroma" in p.lower() or "pungent" in p.lower() for p in phrases(text, result["smell"]))


# --- Endpoint ----------------------------------------------------------------


def test_sensory_endpoint_echoes_batch_and_blocks() -> None:
    payload = {
        "batchIndex": 4,
        "paragraphs": [
            {
                "block": 0,
                "text": "The pungent aroma of garlic filled the kitchen.",
            },
            {
                "block": 7,
                "text": "A loud scream echoed down the corridor.",
            },
        ],
    }
    with TestClient(app) as client:
        response = client.post("/sensory", json=payload, headers=AUTH)
        assert response.status_code == 200
        data = response.json()
        assert data["batchIndex"] == 4
        blocks = {item["block"] for item in data["results"]}
        assert blocks == {0, 7}
        by_block = {item["block"]: item for item in data["results"]}
        assert by_block[0]["counts"]["smell"] >= 1
        assert by_block[7]["counts"]["sound"] >= 1
        for item in data["results"]:
            for sense in SENSES:
                assert sense in item
                assert item["counts"][sense] == len(item[sense])
            assert item.get("details") is None


def test_sensory_endpoint_debug_flag() -> None:
    payload = {
        "batchIndex": 0,
        "debug": True,
        "paragraphs": [
            {"block": 0, "text": "The pungent aroma filled the kitchen."},
        ],
    }
    with TestClient(app) as client:
        data = client.post("/sensory", json=payload, headers=AUTH).json()
    assert data["results"][0]["details"] is not None


def test_sensory_requires_auth() -> None:
    payload = {
        "batchIndex": 0,
        "paragraphs": [{"block": 0, "text": "The pungent aroma filled the kitchen."}],
    }
    with TestClient(app) as client:
        assert client.post("/sensory", json=payload).status_code == 401
        wrong = {"Authorization": "Bearer nope"}
        assert client.post("/sensory", json=payload, headers=wrong).status_code == 401
        assert client.post("/sensory", json=payload, headers=AUTH).status_code == 200


def test_sensory_oversized_paragraph_rejected() -> None:
    from main import MAX_CHARS_PER_PARAGRAPH

    payload = {
        "batchIndex": 0,
        "paragraphs": [
            {"block": 0, "text": "a " * (MAX_CHARS_PER_PARAGRAPH // 2 + 10)},
        ],
    }
    with TestClient(app) as client:
        assert client.post("/sensory", json=payload, headers=AUTH).status_code == 413
