"""Sensory prose analysis: five-sense trigger-word spans via lexicon + embeddings."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from spacy.tokens import Doc, Span, Token

from analysis import Span as CharSpan
from analysis import cp_spans_to_utf16, find_quote_regions, is_inside_regions, normalize_spans
from sensory_lexicon import SENSES, LexiconEntry, SensoryLexicon

# Cosine threshold for Tier-2 sentence classification against sense anchors.
EMBEDDING_THRESHOLD = 0.35

# Soft lexicon strength floor when selecting trigger words after a Tier-2 hit.
SOFT_LEXICON_STRENGTH = 2.5

# Verbs need a higher Lancaster strength than adjectives to avoid tagging common
# action/speech verbs that happen to be auditorily/visually rated (asked, filled).
VERB_MIN_STRENGTH = 4.0
VERB_MIN_EXCLUSIVITY = 0.35

# High-frequency verbs that are almost never sensory description cues.
VERB_DENYLIST = frozenset(
    {
        "ask",
        "say",
        "tell",
        "talk",
        "speak",
        "call",
        "fill",
        "spill",
        "walk",
        "go",
        "come",
        "take",
        "make",
        "get",
        "give",
        "put",
        "look",
        "see",
        "watch",
        "know",
        "think",
        "want",
        "try",
        "use",
        "seem",
        "become",
        "leave",
        "keep",
        "begin",
        "start",
        "stop",
        "turn",
        "move",
        "stand",
        "sit",
        "run",
        "hold",
        "bring",
        "carry",
        "open",
        "close",
        "find",
        "follow",
        "reach",
        "pass",
        "return",
        "appear",
        "remain",
        "continue",
        "include",
        "happen",
        "allow",
        "need",
        "help",
        "show",
        "hear",  # meta-perception verb, not a sound description
        "feel",  # too often emotional / generic; touch uses damp/soft etc.
        "taste",  # meta; prefer sweet/bitter adjectives
        "smell",  # meta; prefer pungent/aroma
    }
)

# High-frequency adjectives that are scalar/evaluative, not sensory description.
ADJ_DENYLIST = frozenset(
    {
        "long",
        "short",
        "small",
        "large",
        "big",
        "little",
        "good",
        "bad",
        "great",
        "old",
        "new",
        "young",
        "early",
        "late",
        "high",
        "low",
        "first",
        "last",
        "next",
        "same",
        "other",
        "own",
        "few",
        "many",
        "much",
        "more",
        "most",
        "only",
        "such",
        "whole",
        "real",
        "true",
        "false",
        "possible",
        "likely",
        "certain",
        "general",
        "particular",
        "important",
        "different",
        "similar",
        "common",
        "simple",
        "single",
        "main",
        "full",
        "empty",
        "open",
        "close",
        "right",
        "left",
        "wrong",
        "best",
        "worst",
        "better",
        "worse",
    }
)

# Content POS tags eligible as trigger words.
CONTENT_POS = {"ADJ", "ADV", "NOUN", "PROPN", "VERB"}
MODIFIER_POS = {"ADJ", "ADV"}

# Nouns are visually dominant in Lancaster; only accept noun triggers for
# non-visual senses (and never for sight via Tier-1 alone).
NOUN_ALLOWED_SENSES = frozenset({"sound", "touch", "smell", "taste"})

SENSE_ANCHORS: dict[str, str] = {
    "sight": "visual appearance bright color light shadow sight vision",
    "sound": "auditory sound noise echo pitch tone loudness hearing",
    "touch": "tactile texture rough soft surface temperature touch feel",
    "smell": "olfactory smell scent odor fragrance aroma reek",
    "taste": "gustatory taste flavor sweet bitter sour salty tongue",
}

INCLUDE_DIALOGUE_DEFAULT = True

_anchor_lock = threading.Lock()
_anchor_matrix: np.ndarray | None = None  # shape (5, 384), L2-normalized


class Embedder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class TriggerHit:
    start: int
    end: int
    sense: str
    confidence: float
    tier: int  # 1 = lexicon, 2 = embedding


def anchor_matrix(embedder: Embedder) -> np.ndarray:
    """Return cached (5, dim) L2-normalized sense-anchor vectors."""
    global _anchor_matrix
    if _anchor_matrix is not None:
        return _anchor_matrix
    with _anchor_lock:
        if _anchor_matrix is not None:
            return _anchor_matrix
        texts = [SENSE_ANCHORS[s] for s in SENSES]
        vectors = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        _anchor_matrix = np.asarray(vectors, dtype=np.float32)
        return _anchor_matrix


def reset_anchor_cache() -> None:
    """Test helper: clear cached anchors between FakeEmbedder swaps."""
    global _anchor_matrix
    with _anchor_lock:
        _anchor_matrix = None


def _token_span(token: Token) -> CharSpan:
    return (token.idx, token.idx + len(token.text))


def _extend_with_advmod(token: Token) -> CharSpan:
    """Pull in a preceding advmod child (e.g. 'faintly sweet')."""
    start, end = _token_span(token)
    for child in token.children:
        if child.dep_ == "advmod" and child.i < token.i:
            start = min(start, child.idx)
            end = max(end, child.idx + len(child.text))
    return (start, end)


def _is_content_token(token: Token) -> bool:
    if token.is_space or token.is_punct or token.is_stop:
        return False
    return token.pos_ in CONTENT_POS


def _lemma_key(token: Token) -> str:
    return token.lemma_.lower() if token.lemma_ else token.text.lower()


def has_descriptive_modifier(sent: Span) -> bool:
    return any(tok.pos_ in MODIFIER_POS or tok.dep_ == "acomp" for tok in sent)


def has_lexicon_hit(sent: Span, lexicon: SensoryLexicon) -> bool:
    return any(lexicon.get(_lemma_key(tok)) is not None for tok in sent if _is_content_token(tok))


def is_candidate_sentence(sent: Span, lexicon: SensoryLexicon) -> bool:
    return has_descriptive_modifier(sent) or has_lexicon_hit(sent, lexicon)


def _pos_allows_trigger(token: Token, sense: str, entry: LexiconEntry | None = None) -> bool:
    if token.pos_ in MODIFIER_POS:
        lemma = _lemma_key(token)
        if token.pos_ == "ADJ" and lemma in ADJ_DENYLIST:
            return False
        return True
    if token.pos_ == "VERB":
        lemma = _lemma_key(token)
        if lemma in VERB_DENYLIST:
            return False
        if entry is not None and (
            entry.strength < VERB_MIN_STRENGTH or entry.exclusivity < VERB_MIN_EXCLUSIVITY
        ):
            return False
        return True
    if token.pos_ in {"NOUN", "PROPN"}:
        return sense in NOUN_ALLOWED_SENSES
    return False


def lexicon_triggers(
    sent: Span,
    lexicon: SensoryLexicon,
    *,
    regions: list[CharSpan] | None = None,
    include_dialogue: bool = INCLUDE_DIALOGUE_DEFAULT,
) -> list[TriggerHit]:
    """Tier-1: strong Lancaster hits on content lemmas → trigger spans."""
    hits: list[TriggerHit] = []
    quote_regions = regions or []
    for token in sent:
        if not _is_content_token(token):
            continue
        if not include_dialogue and is_inside_regions(token.idx, quote_regions):
            continue
        entry = lexicon.get(_lemma_key(token))
        if entry is None:
            continue
        if not _pos_allows_trigger(token, entry.sense, entry):
            continue
        start, end = _extend_with_advmod(token)
        hits.append(
            TriggerHit(
                start=start,
                end=end,
                sense=entry.sense,
                confidence=round(entry.strength / 5.0, 4),
                tier=1,
            )
        )
    return hits


def _soft_lexicon_score(entry: LexiconEntry | None, sense: str) -> float:
    if entry is None:
        return 0.0
    idx = SENSES.index(sense)
    return float(entry.means[idx]) if idx < len(entry.means) else 0.0


def _pick_triggers_for_sense(
    sent: Span,
    sense: str,
    lexicon: SensoryLexicon,
    confidence: float,
    *,
    regions: list[CharSpan] | None = None,
    include_dialogue: bool = INCLUDE_DIALOGUE_DEFAULT,
) -> list[TriggerHit]:
    """After Tier-2 classifies a sentence, pick cue words to highlight."""
    quote_regions = regions or []
    scored: list[tuple[float, Token]] = []
    for token in sent:
        if not _is_content_token(token):
            continue
        if not include_dialogue and is_inside_regions(token.idx, quote_regions):
            continue
        entry = lexicon.get(_lemma_key(token))
        if not _pos_allows_trigger(token, sense, entry):
            continue
        soft = _soft_lexicon_score(entry, sense)
        if soft >= SOFT_LEXICON_STRENGTH:
            scored.append((soft, token))
        elif token.pos_ in MODIFIER_POS or token.dep_ == "acomp":
            # Fallback: descriptive modifiers in a sensory sentence.
            scored.append((soft + 0.1, token))

    if not scored:
        return []

    # Prefer the strongest cues; keep top few to avoid highlighting everything.
    scored.sort(key=lambda x: (-x[0], x[1].i))
    keep = scored[:3]
    hits: list[TriggerHit] = []
    for _score, token in keep:
        start, end = _extend_with_advmod(token)
        hits.append(
            TriggerHit(
                start=start,
                end=end,
                sense=sense,
                confidence=round(confidence, 4),
                tier=2,
            )
        )
    return hits


def embedding_triggers(
    sents: list[Span],
    embedder: Embedder,
    lexicon: SensoryLexicon,
    *,
    threshold: float = EMBEDDING_THRESHOLD,
    regions: list[CharSpan] | None = None,
    include_dialogue: bool = INCLUDE_DIALOGUE_DEFAULT,
) -> list[TriggerHit]:
    """Tier-2: batch-encode leftover sentences and classify vs sense anchors."""
    if not sents:
        return []

    texts = [s.text.strip() for s in sents]
    vectors = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    sent_mat = np.asarray(vectors, dtype=np.float32)
    anchors = anchor_matrix(embedder)
    # Both sides L2-normalized → cosine = matmul.
    sim = sent_mat @ anchors.T  # (n, 5)

    hits: list[TriggerHit] = []
    for i, sent in enumerate(sents):
        row = sim[i]
        best_idx = int(np.argmax(row))
        top = float(row[best_idx])
        if top < threshold:
            continue
        sense = SENSES[best_idx]
        hits.extend(
            _pick_triggers_for_sense(
                sent,
                sense,
                lexicon,
                top,
                regions=regions,
                include_dialogue=include_dialogue,
            )
        )
    return hits


def merge_adjacent(hits: list[TriggerHit]) -> list[TriggerHit]:
    """Join overlapping / abutting same-sense spans (whitespace gap allowed)."""
    if not hits:
        return []
    by_sense: dict[str, list[TriggerHit]] = {s: [] for s in SENSES}
    for h in hits:
        by_sense.setdefault(h.sense, []).append(h)

    merged: list[TriggerHit] = []
    for sense, group in by_sense.items():
        group.sort(key=lambda h: (h.start, h.end))
        cur: TriggerHit | None = None
        for h in group:
            if cur is None:
                cur = h
                continue
            # Abut or overlap (allow a single space between).
            if h.start <= cur.end + 1:
                cur = TriggerHit(
                    start=cur.start,
                    end=max(cur.end, h.end),
                    sense=sense,
                    confidence=max(cur.confidence, h.confidence),
                    tier=min(cur.tier, h.tier),
                )
            else:
                merged.append(cur)
                cur = h
        if cur is not None:
            merged.append(cur)
    return merged


def resolve_overlaps(hits: list[TriggerHit]) -> list[TriggerHit]:
    """Ensure each char range belongs to at most one sense.

    Prefer lower tier number (lexicon over embedding), then higher confidence,
    then earlier sense order as a stable tie-break.
    """
    if not hits:
        return []
    sense_rank = {s: i for i, s in enumerate(SENSES)}
    ordered = sorted(
        hits,
        key=lambda h: (h.tier, -h.confidence, sense_rank.get(h.sense, 99), h.start, h.end),
    )
    taken: list[CharSpan] = []
    kept: list[TriggerHit] = []
    for h in ordered:
        span = (h.start, h.end)
        if any(span[0] < t[1] and t[0] < span[1] for t in taken):
            continue
        taken.append(span)
        kept.append(h)
    return kept


def candidate_sentences(doc: Doc, lexicon: SensoryLexicon) -> list[Span]:
    return [sent for sent in doc.sents if is_candidate_sentence(sent, lexicon)]


def analyze_sensory_paragraph(
    doc: Doc,
    lexicon: SensoryLexicon,
    embedder: Embedder | None = None,
    *,
    include_dialogue: bool = INCLUDE_DIALOGUE_DEFAULT,
    embedding_threshold: float = EMBEDDING_THRESHOLD,
    debug: bool = False,
) -> dict[str, Any]:
    """Return per-sense UTF-16 spans + counts for one paragraph Doc."""
    text = doc.text
    regions = find_quote_regions(text) if not include_dialogue else []

    candidates = candidate_sentences(doc, lexicon)
    tier1_hits: list[TriggerHit] = []
    covered_sents: set[int] = set()

    for sent in candidates:
        hits = lexicon_triggers(
            sent,
            lexicon,
            regions=regions if not include_dialogue else None,
            include_dialogue=include_dialogue,
        )
        if hits:
            tier1_hits.extend(hits)
            covered_sents.add(sent.start)

    leftover = [s for s in candidates if s.start not in covered_sents]
    tier2_hits: list[TriggerHit] = []
    if leftover and embedder is not None:
        tier2_hits = embedding_triggers(
            leftover,
            embedder,
            lexicon,
            threshold=embedding_threshold,
            regions=regions if not include_dialogue else None,
            include_dialogue=include_dialogue,
        )

    all_hits = resolve_overlaps(merge_adjacent(tier1_hits + tier2_hits))

    by_sense: dict[str, list[CharSpan]] = {s: [] for s in SENSES}
    details: list[dict[str, Any]] = []
    for h in all_hits:
        by_sense[h.sense].append((h.start, h.end))
        if debug:
            details.append(
                {
                    "span": cp_spans_to_utf16(text, [(h.start, h.end)])[0],
                    "sense": h.sense,
                    "confidence": h.confidence,
                    "tier": h.tier,
                }
            )

    result: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for sense in SENSES:
        spans_cp = normalize_spans(by_sense[sense])
        utf16 = cp_spans_to_utf16(text, spans_cp)
        result[sense] = utf16
        counts[sense] = len(utf16)
    result["counts"] = counts
    if debug:
        result["details"] = details
    return result
