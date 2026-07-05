"""Tests for prose analysis."""

from __future__ import annotations

import spacy
import pytest
from fastapi.testclient import TestClient

from analysis import analyze_paragraph, codepoint_index_to_utf16, to_utf16_span
from main import app

nlp = spacy.load("en_core_web_sm", disable=["ner"])

SAMPLE_PARAGRAPHS = [
    (
        0,
        "The lighthouse stood pale against the bruised sky, and the wind was carried inland like a rumor nobody wanted to repeat.",
        {"passive": ["was carried"], "tags": []},
    ),
    (
        1,
        'Mara pressed her palm to the cold glass and listened to the gulls shrieking over the breakwater. "You came back," she said, not turning around.',
        {"passive": [], "tags": ["said"]},
    ),
    (
        2,
        '"I never really left." Thomas muttered, and the words were swallowed by the salt-thick air.',
        {"passive": ["were swallowed"], "tags": ["muttered"]},
    ),
    (
        3,
        "The lamp above them flickered, throwing long amber teeth across the floor. Something had been broken here, years ago, and never repaired.",
        {"passive": ["had been broken"], "tags": []},
    ),
]


def slice_text(text: str, start: int, end: int) -> str:
    """Slice using UTF-16 code units like JavaScript (BMP-safe for our samples)."""
    return text[start:end]


@pytest.mark.parametrize("block,text,expected", SAMPLE_PARAGRAPHS)
def test_analyze_paragraph_spans(block: int, text: str, expected: dict) -> None:
    doc = nlp(text)
    result = analyze_paragraph(doc)

    for passive_phrase in expected["passive"]:
        assert any(
            slice_text(text, s, e) == passive_phrase for s, e in result["passive"]
        ), f"block {block}: missing passive span {passive_phrase!r}"

    for tag in expected["tags"]:
        assert any(slice_text(text, s, e) == tag for s, e in result["tags"]), (
            f"block {block}: missing tag span {tag!r}"
        )
        assert not any(slice_text(text, s, e) == tag for s, e in result["active"]), (
            f"block {block}: tag {tag!r} should not also be active"
        )

    assert result["counts"]["passive"] == len(result["passive"])
    assert result["counts"]["active"] == len(result["active"])


def test_passive_and_active_both_present() -> None:
    text = (
        "The lighthouse stood pale against the bruised sky, and the wind "
        "was carried inland like a rumor nobody wanted to repeat."
    )
    result = analyze_paragraph(nlp(text))
    assert result["counts"]["passive"] == 1
    assert result["counts"]["active"] >= 1
    assert any(slice_text(text, s, e) == "was carried" for s, e in result["passive"])
    assert any(slice_text(text, s, e) == "stood" for s, e in result["active"])


def actives(text: str, result: dict) -> list[str]:
    return [slice_text(text, s, e) for s, e in result["active"]]


def tags(text: str, result: dict) -> list[str]:
    return [slice_text(text, s, e) for s, e in result["tags"]]


# --- Approach A: syntactic dialogue tags (verbs NOT in the seed lexicon) -----


@pytest.mark.parametrize(
    "text,verb",
    [
        ('"I hate this," she breathed, barely audible.', "breathed"),
        ('"Stop," he growled.', "growled"),
        ('"Where?" she rasped.', "rasped"),
        ('She murmured, "Come here."', "murmured"),
    ],
)
def test_syntactic_tags_without_lexicon(text: str, verb: str) -> None:
    from analysis import ATTRIBUTION_VERBS

    doc = nlp(text)
    result = analyze_paragraph(doc)
    lemma = nlp(verb)[0].lemma_.lower()
    assert lemma not in ATTRIBUTION_VERBS, f"{verb!r} unexpectedly in seed lexicon"
    assert verb in tags(text, result), f"{verb!r} not detected as a tag"


def test_action_beat_adjacent_to_quote_is_not_a_tag() -> None:
    # "nodded" is an action beat, not attribution; the quote is a separate
    # fragment that "nodded" does not govern, so it must not be tagged.
    text = 'He nodded. "Fine."'
    result = analyze_paragraph(nlp(text))
    assert "nodded" not in tags(text, result)


# --- Finite-verb active detection -------------------------------------------


def test_bare_participles_and_infinitives_are_not_active() -> None:
    text = "The lamp flickered, throwing amber teeth, and nobody wanted to repeat it."
    result = analyze_paragraph(nlp(text))
    found = actives(text, result)
    assert "flickered" in found
    assert "wanted" in found
    assert "throwing" not in found  # bare participle
    assert "to repeat" not in found  # infinitive
    assert not any("repeat" in a for a in found)


def test_auxiliary_carried_nonfinite_verbs_are_active() -> None:
    for text, phrase in [
        ("He had been running for hours.", "had been running"),
        ("She will go home.", "will go"),
    ]:
        result = analyze_paragraph(nlp(text))
        assert phrase in actives(text, result), f"missing active {phrase!r}"


def test_in_dialogue_verbs_excluded_from_active() -> None:
    text = '"You came back," she said.'
    result = analyze_paragraph(nlp(text))
    assert "came" not in actives(text, result)  # inside the quote
    assert result["counts"]["active"] == len(result["active"])


def test_utf16_offsets_match_bmp_text() -> None:
    text = "hello"
    assert to_utf16_span(text, 0, 5) == [0, 5]
    assert codepoint_index_to_utf16(text, 2) == 2


def test_batch_endpoint_echoes_batch_index_and_blocks() -> None:
    with TestClient(app) as client:
        payload = {
            "batchIndex": 3,
            "paragraphs": [
                {"block": 0, "text": SAMPLE_PARAGRAPHS[0][1]},
                {"block": 5, "text": SAMPLE_PARAGRAPHS[1][1]},
            ],
        }
        response = client.post("/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["batchIndex"] == 3
        blocks = {item["block"] for item in data["results"]}
        assert blocks == {0, 5}
        for item in data["results"]:
            assert "passive" in item
            assert "active" in item
            assert "tags" in item
            assert item["counts"]["passive"] == len(item["passive"])
            assert item["counts"]["active"] == len(item["active"])


def test_oversized_paragraph_rejected() -> None:
    from main import MAX_CHARS_PER_PARAGRAPH

    with TestClient(app) as client:
        payload = {
            "batchIndex": 0,
            "paragraphs": [
                {"block": 0, "text": "a " * (MAX_CHARS_PER_PARAGRAPH // 2 + 10)},
            ],
        }
        response = client.post("/analyze", json=payload)
        assert response.status_code == 413


def test_health() -> None:
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
