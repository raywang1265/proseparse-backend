"""Tests for exposition classification and /exposition endpoint."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Callable

# Auth is mandatory; set the key before the app lifespan runs (on TestClient enter).
os.environ.setdefault("ANALYSIS_API_KEY", "test-key")
AUTH = {"Authorization": "Bearer test-key"}

import pytest
import torch
from fastapi.testclient import TestClient

import main as main_mod
from exposition import MAX_TOKENS, classify_paragraphs
from main import app


class FakeTokenizer:
    """Word-count stand-in: 1 token per word plus 2 special tokens."""

    def __init__(self) -> None:
        self.last_texts: list[str] = []

    def __call__(
        self,
        texts: list[str],
        truncation: bool = False,
        padding: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        self.last_texts = list(texts)
        ids_list: list[list[int]] = []
        for text in texts:
            n = max(1, len(text.split()) + (2 if add_special_tokens else 0))
            ids = list(range(n))
            if truncation and max_length is not None:
                ids = ids[:max_length]
            ids_list.append(ids)

        if return_tensors == "pt":
            max_len = max(len(ids) for ids in ids_list)
            if padding:
                padded = [ids + [0] * (max_len - len(ids)) for ids in ids_list]
                mask = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in ids_list]
            else:
                padded, mask = ids_list, [[1] * len(ids) for ids in ids_list]
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
            }
        return {"input_ids": ids_list}


class FakeModel:
    """Returns logits from a text callback; ``parameters()`` lives on CPU."""

    def __init__(
        self,
        tokenizer: FakeTokenizer,
        logit_fn: Callable[[str], list[float]],
        label2id: dict[str, int] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.logit_fn = logit_fn
        self.config = SimpleNamespace(
            label2id=label2id or {"indirect": 0, "direct": 1}
        )
        self._param = torch.zeros(1)

    def parameters(self):
        yield self._param

    def __call__(self, **_kwargs: Any) -> SimpleNamespace:
        logits = [self.logit_fn(text) for text in self.tokenizer.last_texts]
        return SimpleNamespace(logits=torch.tensor(logits, dtype=torch.float32))


def _stub_pair(
    logit_fn: Callable[[str], list[float]],
    label2id: dict[str, int] | None = None,
) -> tuple[FakeTokenizer, FakeModel]:
    tokenizer = FakeTokenizer()
    return tokenizer, FakeModel(tokenizer, logit_fn, label2id=label2id)


def test_label_resolved_by_name_not_index() -> None:
    """Swapped label2id: column 0 is `direct`. High score there must still be direct."""
    tokenizer, model = _stub_pair(
        lambda _text: [5.0, 0.0],
        label2id={"direct": 0, "indirect": 1},
    )
    result = classify_paragraphs(tokenizer, model, ["Any paragraph."])[0]
    assert result["label"] == "direct"
    assert result["pDirect"] > 0.9


def test_direct_share_rounding() -> None:
    # softmax([0, log(0.8123/0.1877)])[1] == 0.8123
    p = 0.8123
    logit_direct = float(torch.log(torch.tensor(p / (1.0 - p))))
    tokenizer, model = _stub_pair(lambda _text: [0.0, logit_direct])
    result = classify_paragraphs(tokenizer, model, ["Any paragraph."])[0]
    assert result["label"] == "direct"
    assert result["pDirect"] == pytest.approx(p, abs=1e-4)
    assert result["directShare"] == 81


def test_half_probability_is_direct() -> None:
    tokenizer, model = _stub_pair(lambda _text: [0.0, 0.0])
    result = classify_paragraphs(tokenizer, model, ["Tied logits."])[0]
    assert result["pDirect"] == pytest.approx(0.5)
    assert result["label"] == "direct"
    assert result["directShare"] == 50


def test_length_bucket_restores_input_order() -> None:
    """Short paragraph is batched first; results must still follow input order."""

    def logit_fn(text: str) -> list[float]:
        if text.startswith("SHORT"):
            return [0.0, 4.0]  # direct
        return [4.0, 0.0]  # indirect

    tokenizer, model = _stub_pair(logit_fn)
    long = "LONG " + ("word " * 40)
    short = "SHORT ok"
    results = classify_paragraphs(tokenizer, model, [long, short])
    assert results[0]["label"] == "indirect"
    assert results[1]["label"] == "direct"


def test_truncated_flag() -> None:
    tokenizer, model = _stub_pair(lambda _text: [4.0, 0.0])
    over = "word " * (MAX_TOKENS + 10)
    under = "A short paragraph."
    results = classify_paragraphs(tokenizer, model, [over, under])
    assert results[0]["truncated"] is True
    assert results[1]["truncated"] is False


@pytest.fixture
def stub_exposition(monkeypatch: pytest.MonkeyPatch) -> None:
    def logit_fn(text: str) -> list[float]:
        lowered = text.lower()
        if "was a kind man" in lowered or "had always been" in lowered:
            return [0.0, 3.0]
        return [3.0, 0.0]

    tokenizer, model = _stub_pair(logit_fn)

    def _ensure() -> tuple[Any, Any]:
        return tokenizer, model

    monkeypatch.setattr(main_mod, "ensure_exposition_model", _ensure)
    monkeypatch.setattr(main_mod, "exposition_tokenizer", tokenizer)
    monkeypatch.setattr(main_mod, "exposition_model", model)


def test_exposition_endpoint_echoes_batch_and_blocks(stub_exposition: None) -> None:
    payload = {
        "batchIndex": 6,
        "paragraphs": [
            {
                "block": 0,
                "text": "Thomas was a kind man who had always been afraid of the dark.",
            },
            {
                "block": 4,
                "text": "Mara pressed her palm to the glass and did not look back.",
            },
        ],
    }
    with TestClient(app) as client:
        response = client.post("/exposition", json=payload, headers=AUTH)
        assert response.status_code == 200
        data = response.json()
        assert data["batchIndex"] == 6
        by_block = {item["block"]: item for item in data["results"]}
        assert set(by_block) == {0, 4}
        assert by_block[0]["label"] == "direct"
        assert by_block[0]["directShare"] == round(100 * by_block[0]["pDirect"])
        assert by_block[4]["label"] == "indirect"
        assert by_block[4]["truncated"] is False


def test_exposition_requires_auth(stub_exposition: None) -> None:
    payload = {
        "batchIndex": 0,
        "paragraphs": [{"block": 0, "text": "The wind carried the rumor inland."}],
    }
    with TestClient(app) as client:
        assert client.post("/exposition", json=payload).status_code == 401
        wrong = {"Authorization": "Bearer nope"}
        assert client.post("/exposition", json=payload, headers=wrong).status_code == 401
        assert client.post("/exposition", json=payload, headers=AUTH).status_code == 200


def test_exposition_oversized_paragraph_rejected(stub_exposition: None) -> None:
    from main import MAX_CHARS_PER_PARAGRAPH

    payload = {
        "batchIndex": 0,
        "paragraphs": [
            {"block": 0, "text": "a " * (MAX_CHARS_PER_PARAGRAPH // 2 + 10)},
        ],
    }
    with TestClient(app) as client:
        assert client.post("/exposition", json=payload, headers=AUTH).status_code == 413


@pytest.mark.skipif(
    not os.getenv("EXPOSITION_INTEGRATION"),
    reason="set EXPOSITION_INTEGRATION=1 to load the real checkpoint",
)
def test_real_model_direct_vs_indirect() -> None:
    from exposition import load_classifier

    tokenizer, model = load_classifier()
    results = classify_paragraphs(
        tokenizer,
        model,
        [
            (
                "John was a kind and patient man who had always been afraid of "
                "the dark and missed his mother terribly every single day."
            ),
            "John's hand trembled on the latch. He did not look back.",
        ],
    )
    assert results[0]["label"] == "direct"
    assert results[1]["label"] == "indirect"
