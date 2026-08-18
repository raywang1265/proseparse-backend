#!/usr/bin/env python3
"""Build the compact sensory lexicon artifact from Lancaster Sensorimotor Norms.

Download the aggregated CSV from the OSF Data component
(https://osf.io/7emr6/ → Data → Lancaster_sensorimotor_norms_for_39707_words.csv)
and pass its path as the first argument:

    python scripts/build_sensory_lexicon.py path/to/Lancaster_sensorimotor_norms_for_39707_words.csv

Writes api/data/sensory_lexicon.json.gz (committed). The raw CSV is NOT committed.

License: CC BY 4.0 — Lynott, Connell, Brysbaert, Brand & Carney (2020).
See api/data/LANCASTER_ATTRIBUTION.md.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from pathlib import Path

# Map Lancaster Dominant.perceptual labels → our API sense keys.
DOMINANT_TO_SENSE = {
    "Visual": "sight",
    "Auditory": "sound",
    "Haptic": "touch",
    "Olfactory": "smell",
    "Gustatory": "taste",
    # Interoceptive is a suppressor — never emitted as a highlight sense.
}

SENSE_MEAN_COLS = {
    "sight": "Visual.mean",
    "sound": "Auditory.mean",
    "touch": "Haptic.mean",
    "smell": "Olfactory.mean",
    "taste": "Gustatory.mean",
}

# Per-sense minimum Max_strength.perceptual. Sight is stricter because concrete
# nouns are visually dominant by default and would otherwise flood the output.
MIN_STRENGTH = {
    "sight": 4.0,
    "sound": 3.5,
    "touch": 3.5,
    "smell": 3.5,
    "taste": 3.5,
}

# Dominant sense must beat the next-best perceptual mean by at least this margin.
MIN_MARGIN = 0.5

# Soft floor for exclusivity (0–1). Higher = more modality-specific.
MIN_EXCLUSIVITY = 0.25

# Single alphabetic word only (no multi-word concepts, no punctuation).
WORD_RE = re.compile(r"^[a-z]+$")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "api" / "data" / "sensory_lexicon.json.gz"


def _f(row: dict[str, str], key: str) -> float:
    raw = (row.get(key) or "").strip()
    if not raw:
        return 0.0
    return float(raw)


def build_entries(csv_path: Path) -> dict[str, dict]:
    """Return {lemma: {sense, strength, exclusivity, means}} filtered entries."""
    entries: dict[str, dict] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            word = (row.get("Word") or "").strip().lower()
            if not WORD_RE.match(word):
                continue

            dominant = (row.get("Dominant.perceptual") or "").strip()
            sense = DOMINANT_TO_SENSE.get(dominant)
            if sense is None:
                continue  # Interoceptive or unknown

            strength = _f(row, "Max_strength.perceptual")
            exclusivity = _f(row, "Exclusivity.perceptual")
            if strength < MIN_STRENGTH[sense] or exclusivity < MIN_EXCLUSIVITY:
                continue

            means = {s: _f(row, col) for s, col in SENSE_MEAN_COLS.items()}
            # Also track interoceptive as suppressor for margin checks.
            intero = _f(row, "Interoceptive.mean")
            others = [v for s, v in means.items() if s != sense] + [intero]
            second = max(others) if others else 0.0
            if strength - second < MIN_MARGIN:
                continue

            # Keep the stronger entry if a lemma appears twice (shouldn't, but safe).
            prev = entries.get(word)
            if prev is not None and prev["strength"] >= strength:
                continue

            entries[word] = {
                "sense": sense,
                "strength": round(strength, 4),
                "exclusivity": round(exclusivity, 4),
                "means": {s: round(v, 4) for s, v in means.items()},
            }
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to Lancaster_sensorimotor_norms_for_39707_words.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output path (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    if not args.csv_path.is_file():
        raise SystemExit(f"CSV not found: {args.csv_path}")

    entries = build_entries(args.csv_path)
    by_sense: dict[str, int] = {}
    for e in entries.values():
        by_sense[e["sense"]] = by_sense.get(e["sense"], 0) + 1

    payload = {
        "source": "Lancaster Sensorimotor Norms (Lynott et al., 2020)",
        "license": "CC BY 4.0",
        "osf": "https://osf.io/7emr6/",
        "filters": {
            "minStrength": MIN_STRENGTH,
            "minMargin": MIN_MARGIN,
            "minExclusivity": MIN_EXCLUSIVITY,
        },
        "counts": {"total": len(entries), "bySense": by_sense},
        "entries": entries,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as gz:
        json.dump(payload, gz, separators=(",", ":"), ensure_ascii=True)

    size_kb = args.output.stat().st_size / 1024
    print(f"Wrote {args.output} ({size_kb:.1f} KB)")
    print(f"Entries: {len(entries)} — {by_sense}")


if __name__ == "__main__":
    main()
