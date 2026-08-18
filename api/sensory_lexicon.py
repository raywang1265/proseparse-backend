"""Lazy loader for the compiled Lancaster sensory lexicon artifact."""

from __future__ import annotations

import gzip
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SENSES = ("sight", "sound", "touch", "smell", "taste")
SENSE_INDEX = {s: i for i, s in enumerate(SENSES)}

_DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "sensory_lexicon.json.gz"

_lock = threading.Lock()
_lexicon: SensoryLexicon | None = None


@dataclass(frozen=True, slots=True)
class LexiconEntry:
    sense: str  # one of SENSES
    sense_index: int
    strength: float
    exclusivity: float
    means: tuple[float, ...]  # length 5, aligned with SENSES


class SensoryLexicon:
    """In-memory lemma → LexiconEntry lookup."""

    def __init__(self, entries: dict[str, LexiconEntry], meta: dict[str, Any]) -> None:
        self._entries = entries
        self.meta = meta

    def get(self, lemma: str) -> LexiconEntry | None:
        return self._entries.get(lemma.lower())

    def __contains__(self, lemma: str) -> bool:
        return lemma.lower() in self._entries

    def __len__(self) -> int:
        return len(self._entries)


def _parse_payload(payload: dict[str, Any]) -> SensoryLexicon:
    raw_entries = payload.get("entries") or {}
    parsed: dict[str, LexiconEntry] = {}
    for lemma, data in raw_entries.items():
        sense = data["sense"]
        if sense not in SENSE_INDEX:
            continue
        means_dict = data.get("means") or {}
        means = tuple(float(means_dict.get(s, 0.0)) for s in SENSES)
        parsed[lemma.lower()] = LexiconEntry(
            sense=sense,
            sense_index=SENSE_INDEX[sense],
            strength=float(data["strength"]),
            exclusivity=float(data["exclusivity"]),
            means=means,  # type: ignore[arg-type]
        )
    meta = {
        "source": payload.get("source"),
        "license": payload.get("license"),
        "osf": payload.get("osf"),
        "counts": payload.get("counts"),
        "filters": payload.get("filters"),
    }
    return SensoryLexicon(parsed, meta)


def load_lexicon(path: Path | None = None) -> SensoryLexicon:
    """Load sensory lexicon from the gzipped JSON artifact (no process-wide cache)."""
    target = path or _DEFAULT_PATH
    if not target.is_file():
        raise FileNotFoundError(
            f"Sensory lexicon not found at {target}. "
            "Run scripts/build_sensory_lexicon.py after downloading the "
            "Lancaster Sensorimotor Norms CSV from https://osf.io/7emr6/."
        )

    with gzip.open(target, "rt", encoding="utf-8") as gz:
        payload = json.load(gz)

    return _parse_payload(payload)


def get_lexicon() -> SensoryLexicon:
    """Thread-safe accessor used by the /sensory route."""
    global _lexicon
    if _lexicon is not None:
        return _lexicon
    with _lock:
        if _lexicon is None:
            _lexicon = load_lexicon()
        return _lexicon
