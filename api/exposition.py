"""Direct vs indirect exposition classifier (DeBERTa-v3 fine-tune)."""

from __future__ import annotations

import os
from typing import Any

import torch

MODEL_ID = os.getenv("EXPOSITION_MODEL", "gu1npen/proseparse-exposition-finetune")
MODEL_REVISION = os.getenv("EXPOSITION_REVISION", "main")
MAX_TOKENS = 384
INFERENCE_BATCH = 8
DIRECT_LABEL = "direct"
INDIRECT_LABEL = "indirect"


def load_classifier() -> tuple[Any, Any]:
    """Load tokenizer + sequence-classification model; pin torch thread count."""
    threads = int(os.getenv("TORCH_NUM_THREADS", "2"))
    torch.set_num_threads(threads)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    model.eval()
    return tokenizer, model


def _direct_index(model: Any) -> int:
    """Column of the `direct` class — look up by name, never hardcode id 1."""
    config = getattr(model, "config", None)
    label2id = getattr(config, "label2id", None) or {}
    if DIRECT_LABEL in label2id:
        return int(label2id[DIRECT_LABEL])
    id2label = getattr(config, "id2label", None) or {}
    for idx, name in id2label.items():
        if name == DIRECT_LABEL:
            return int(idx)
    raise RuntimeError(
        f"model config is missing {DIRECT_LABEL!r}: "
        f"label2id={label2id!r} id2label={id2label!r}"
    )


def _result_from_p_direct(p_direct: float, truncated: bool) -> dict:
    return {
        "label": DIRECT_LABEL if p_direct >= 0.5 else INDIRECT_LABEL,
        "pDirect": p_direct,
        "directShare": round(100 * p_direct),
        "truncated": truncated,
    }


def classify_paragraphs(tokenizer: Any, model: Any, texts: list[str]) -> list[dict]:
    """Score each paragraph; return dicts in input order.

    Tokenizes once without truncation to get true lengths (and the `truncated`
    flag), sorts by length, then runs dynamically-padded sub-batches of
    ``INFERENCE_BATCH`` so padding waste stays low.
    """
    untruncated = tokenizer(
        texts,
        truncation=False,
        padding=False,
        add_special_tokens=True,
    )
    lengths = [len(ids) for ids in untruncated["input_ids"]]
    truncated_flags = [length > MAX_TOKENS for length in lengths]
    order = sorted(range(len(texts)), key=lambda i: lengths[i])

    direct_idx = _direct_index(model)
    device = next(model.parameters()).device
    results: list[dict | None] = [None] * len(texts)

    for start in range(0, len(order), INFERENCE_BATCH):
        batch_indices = order[start : start + INFERENCE_BATCH]
        batch_texts = [texts[i] for i in batch_indices]
        batch = tokenizer(
            batch_texts,
            truncation=True,
            max_length=MAX_TOKENS,
            padding=True,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, "to")}
        with torch.inference_mode():
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)

        for j, orig_i in enumerate(batch_indices):
            p_direct = float(probs[j, direct_idx].item())
            results[orig_i] = _result_from_p_direct(p_direct, truncated_flags[orig_i])

    assert all(r is not None for r in results)
    return results  # type: ignore[return-value]
