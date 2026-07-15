"""Tests for character voice extraction and /voice endpoint."""

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

from voice import (
    EMBEDDING_DIM,
    UNKNOWN,
    analyze_voice_chapter,
    attribute_quotes,
    compute_stylometry,
)
import main as main_mod
from main import app

nlp_ner = spacy.load("en_core_web_sm")


class FakeEmbedder:
    """Deterministic stand-in for SentenceTransformer (avoids downloading MiniLM)."""

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray:
        out = []
        for i, s in enumerate(sentences):
            vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            # Spread a few non-zero dims so vectors differ by content length/index.
            vec[i % EMBEDDING_DIM] = 1.0
            vec[(len(s) + i) % EMBEDDING_DIM] = 0.5
            out.append(vec)
        return np.stack(out)


@pytest.fixture(autouse=True)
def _stub_voice_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point /voice at local spaCy NER + FakeEmbedder (no MiniLM download)."""
    fake = FakeEmbedder()
    monkeypatch.setattr(main_mod, "nlp_ner", nlp_ner)
    monkeypatch.setattr(main_mod, "embedder", fake)

    def _ensure() -> tuple[Any, Any]:
        return nlp_ner, fake

    monkeypatch.setattr(main_mod, "ensure_voice_models", _ensure)


def test_attribute_explicit_named_speaker() -> None:
    text = (
        'Mara pressed her palm to the glass. "You came back," she said, '
        "not turning around."
    )
    doc = nlp_ner(text)
    assignments = attribute_quotes(doc)
    speakers = [a.speaker for a in assignments]
    assert "Mara" in speakers
    assert any("You came back" in a.dialogue for a in assignments)


def test_pronoun_lookback_zips_past_long_quote() -> None:
    """Long quoted speech must not push the earlier name out of lookback range."""
    long_speech = " ".join(
        [
            "I have walked these halls for years and counted every crack in the stone.",
            "The wind never sleeps, and neither do the gulls that circle the tower.",
            "You should have written. You should have warned me about the tide.",
            "Instead you vanished like fog and left me holding a cold lamp.",
            "The stairs remember your footsteps better than I remember your face.",
            "Tell me now: why did you return on a night like this, after all this silence?",
        ]
    )
    text = (
        f'Mara waited by the door. "{long_speech}" she said, not turning around.'
    )
    doc = nlp_ner(text)
    assignments = attribute_quotes(doc)
    assert any(a.speaker == "Mara" for a in assignments)
    # Sanity: the quoted span alone is far longer than the old 400-char window.
    assert len(long_speech) > 400


def test_attribute_split_sentence_tag() -> None:
    text = '"I never really left." Thomas muttered, and the wind howled.'
    doc = nlp_ner(text)
    assignments = attribute_quotes(doc)
    assert any(a.speaker == "Thomas" for a in assignments)


def test_rapid_fire_alternation() -> None:
    # Seed A=Thomas (most recent), B=Mara; untagged lines alternate B, A → Mara, Thomas.
    text = (
        '"Hello," Mara said.\n'
        '"Hi," Thomas replied.\n'
        '"How are you?"\n'
        '"Fine."\n'
    )
    doc = nlp_ner(text)
    assignments = attribute_quotes(doc)
    by_text = {a.dialogue.rstrip(",.?!"): a.speaker for a in assignments}
    assert by_text.get("Hello") == "Mara"
    assert by_text.get("Hi") == "Thomas"
    assert by_text.get("How are you") == "Mara"
    assert by_text.get("Fine") == "Thomas"


def test_single_speaker_tagless_goes_unknown() -> None:
    text = '"Only me here," Mara said.\n"Still talking."\n"And again."\n'
    doc = nlp_ner(text)
    assignments = attribute_quotes(doc)
    # First line attributed; later tagless lines lack a second seed → UNKNOWN.
    later = [a for a in assignments if "Still talking" in a.dialogue or "And again" in a.dialogue]
    assert later
    assert all(a.speaker == UNKNOWN for a in later)


def test_vocative_inside_quote_is_not_speaker() -> None:
    text = '"Run, Mara!" Thomas shouted.'
    doc = nlp_ner(text)
    assignments = attribute_quotes(doc)
    assert len(assignments) == 1
    assert assignments[0].speaker == "Thomas"
    assert "Mara" not in (assignments[0].speaker,)


def test_stylometry_raw_integer_tallies() -> None:
    sty = compute_stylometry(["I can't believe it!", "Really?"], nlp_ner)
    assert isinstance(sty["sentenceCount"], int)
    assert isinstance(sty["tokenCount"], int)
    assert isinstance(sty["wordCount"], int)
    assert isinstance(sty["charCount"], int)
    assert isinstance(sty["contractionCount"], int)
    assert sty["contractionCount"] >= 1
    assert sty["wordCount"] >= 1
    assert sty["wordCount"] <= sty["tokenCount"]
    assert isinstance(sty["punctuation"], dict)
    assert isinstance(sty["posCounts"], dict)
    assert isinstance(sty["uniqueLemmas"], list)
    assert all(isinstance(x, str) for x in sty["uniqueLemmas"])


def test_analyze_voice_chapter_vectors_are_384() -> None:
    text = (
        '"You came back," Mara said.\n'
        '"I never really left," Thomas replied.\n'
    )
    doc = nlp_ner(text)
    result = analyze_voice_chapter(doc, FakeEmbedder(), stylometry_nlp=nlp_ner)
    assert result
    for ch in result:
        assert len(ch["vector"]) == EMBEDDING_DIM
        assert isinstance(ch["stylometry"]["tokenCount"], int)
        assert isinstance(ch["uniqueLemmas"], list)


def test_voice_endpoint_returns_single_payload() -> None:
    payload = {
        "batchIndex": 2,
        "paragraphs": [
            {
                "block": 0,
                "text": '"You came back," Mara said, not turning around.',
            },
            {
                "block": 1,
                "text": '"I never really left," Thomas replied.',
            },
        ],
    }
    with TestClient(app) as client:
        response = client.post("/voice", json=payload, headers=AUTH)
        assert response.status_code == 200
        data = response.json()
        assert data["batchIndex"] == 2
        assert "characters" in data
        assert isinstance(data["characters"], list)
        names = {c["name"] for c in data["characters"]}
        # At least one named speaker; UNKNOWN may or may not appear.
        assert names & {"Mara", "Thomas"}
        for ch in data["characters"]:
            assert len(ch["vector"]) == EMBEDDING_DIM
            assert isinstance(ch["stylometry"]["sentenceCount"], int)
            assert isinstance(ch["uniqueLemmas"], list)
            assert "spans" in ch
            assert isinstance(ch["spans"], list)

        # Dialogue spans land on the correct paragraph blocks and slice quoted text.
        by_name = {c["name"]: c for c in data["characters"]}
        mara_spans = by_name["Mara"]["spans"]
        thomas_spans = by_name["Thomas"]["spans"]
        assert mara_spans and mara_spans[0]["block"] == 0
        assert thomas_spans and thomas_spans[0]["block"] == 1
        p0 = payload["paragraphs"][0]["text"]
        s, e = mara_spans[0]["span"]
        assert "You came back" in p0[s:e]


def test_unknown_speaker_has_spans_when_unattributed() -> None:
    payload = {
        "batchIndex": 0,
        "paragraphs": [
            {"block": 5, "text": '"Only me here," Mara said.'},
            {"block": 6, "text": '"Still talking to the void."'},
        ],
    }
    with TestClient(app) as client:
        data = client.post("/voice", json=payload, headers=AUTH).json()
    by_name = {c["name"]: c for c in data["characters"]}
    assert "UNKNOWN" in by_name
    unknown_spans = by_name["UNKNOWN"]["spans"]
    assert unknown_spans
    assert all(sp["block"] == 6 for sp in unknown_spans)
    p6 = payload["paragraphs"][1]["text"]
    s, e = unknown_spans[0]["span"]
    assert "Still talking" in p6[s:e]


def test_voice_requires_auth() -> None:
    payload = {
        "batchIndex": 0,
        "paragraphs": [{"block": 0, "text": '"Hi," Mara said.'}],
    }
    with TestClient(app) as client:
        assert client.post("/voice", json=payload).status_code == 401
        wrong = {"Authorization": "Bearer nope"}
        assert client.post("/voice", json=payload, headers=wrong).status_code == 401
        assert client.post("/voice", json=payload, headers=AUTH).status_code == 200


def test_voice_oversized_paragraph_rejected() -> None:
    from main import MAX_CHARS_PER_PARAGRAPH

    payload = {
        "batchIndex": 0,
        "paragraphs": [
            {"block": 0, "text": "a " * (MAX_CHARS_PER_PARAGRAPH // 2 + 10)},
        ],
    }
    with TestClient(app) as client:
        assert client.post("/voice", json=payload, headers=AUTH).status_code == 413
