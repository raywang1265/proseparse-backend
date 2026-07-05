"""Prose analysis: passive/active voice spans and dialogue tags."""

from __future__ import annotations

from typing import Iterable

from spacy.tokens import Doc, Token

# Seed speech-verb lexicon (lemmas). Dialogue tags are detected primarily by
# SYNTAX (a verb that governs quoted material; see detect_tags). This small list
# is only a fallback for split-sentence attribution like `"...!" Thomas muttered.`
# where the quote is a separate sentence and carries no syntactic link to the
# verb. Novel speech verbs in comma-attribution (`"...," she rasped`) are caught
# syntactically without needing to appear here.
ATTRIBUTION_VERBS = {
    "said",
    "say",
    "says",
    "asked",
    "ask",
    "replied",
    "reply",
    "replies",
    "answered",
    "answer",
    "whispered",
    "whisper",
    "muttered",
    "mutter",
    "snapped",
    "snap",
    "shouted",
    "shout",
    "called",
    "call",
    "cried",
    "cry",
    "offered",
    "offer",
    "added",
    "add",
    "continued",
    "continue",
    "insisted",
    "insist",
    "laughed",
    "laugh",
    "sighed",
    "sigh",
    "began",
    "begin",
    "demanded",
    "demand",
}

PASSIVE_AUX_LEMMAS = {
    "be",
    "get",
}

# Inherently finite verb tags (carry tense/agreement on their own).
FINITE_TAGS = {"VBD", "VBP", "VBZ"}
# Non-finite tags that are only "active" when carried by a finite auxiliary
# (e.g. "had been running", "will go").
NONFINITE_TAGS = {"VBG", "VB"}
# Auxiliary tags that make a construction finite.
FINITE_AUX_TAGS = {"VBD", "VBP", "VBZ", "MD"}

# How far past a closing quote a split-sentence attribution verb may sit
# (allows for an intervening subject: `." Thomas muttered`).
TAG_WINDOW_CHARS = 40

Span = tuple[int, int]


def utf16_code_unit_len(char: str) -> int:
    return 2 if ord(char) > 0xFFFF else 1


def codepoint_index_to_utf16(text: str, codepoint_index: int) -> int:
    """Map a Python str code-point index to a UTF-16 code-unit offset."""
    if codepoint_index <= 0:
        return 0
    utf16 = 0
    for i, ch in enumerate(text):
        if i == codepoint_index:
            return utf16
        utf16 += utf16_code_unit_len(ch)
    return utf16


def to_utf16_span(text: str, start: int, end: int) -> list[int]:
    """Convert a code-point half-open span to UTF-16 half-open offsets."""
    return [
        codepoint_index_to_utf16(text, start),
        codepoint_index_to_utf16(text, end),
    ]


def spans_overlap(a: Span, b: Span) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def normalize_spans(spans: Iterable[Span]) -> list[Span]:
    """Sort, dedupe, and drop overlaps within one category."""
    unique = sorted(set(spans))
    out: list[Span] = []
    for start, end in unique:
        if start >= end:
            continue
        if out and start < out[-1][1]:
            continue
        out.append((start, end))
    return out


def is_passive_participle(token: Token) -> bool:
    if token.tag_ != "VBN":
        return False
    if any(child.dep_ in {"auxpass", "nsubjpass"} for child in token.children):
        return True
    for child in token.children:
        if child.dep_ == "aux" and child.lemma_ in PASSIVE_AUX_LEMMAS:
            return True
    return False


def passive_construction_span(token: Token) -> Span | None:
    if not is_passive_participle(token):
        return None

    start = token.idx
    end = token.idx + len(token.text)

    for child in token.children:
        if child.dep_ in {"auxpass", "aux", "neg"}:
            start = min(start, child.idx)
            end = max(end, child.idx + len(child.text))
        elif child.dep_ == "advmod" and child.i < token.i:
            start = min(start, child.idx)
            end = max(end, child.idx + len(child.text))

    return (start, end)


def find_quote_regions(text: str) -> list[Span]:
    """Return (open_idx, close_idx) char spans for quoted passages.

    Curly quotes (\u201c/\u201d) are matched as pairs; straight quotes (") toggle.
    An unclosed quote runs to the end of the paragraph. `close_idx` is the index
    of the closing mark (or len(text) if unclosed); a token is "inside" the quote
    when open_idx < token.idx < close_idx.
    """
    regions: list[Span] = []
    open_idx: int | None = None
    for i, ch in enumerate(text):
        if ch == "\u201c":
            if open_idx is None:
                open_idx = i
        elif ch == "\u201d":
            if open_idx is not None:
                regions.append((open_idx, i))
                open_idx = None
        elif ch == '"':
            if open_idx is None:
                open_idx = i
            else:
                regions.append((open_idx, i))
                open_idx = None
    if open_idx is not None:
        regions.append((open_idx, len(text)))
    return regions


def is_inside_regions(idx: int, regions: list[Span]) -> bool:
    return any(start < idx < end for start, end in regions)


def detect_tags(doc: Doc, regions: list[Span]) -> list[Span]:
    """Dialogue-attribution verbs, detected syntactically (approach A).

    A1 (syntactic, lexicon-free): a verb that sits OUTSIDE the quotes but governs
    a (non-punctuation) token INSIDE a quote region is a speech verb. This covers
    same-sentence attribution in both directions -- `she said, "..."` and
    `"...," she said` -- because the quoted clause attaches to the verb as
    ccomp/parataxis. Novel verbs (`rasped`, `growled`) are caught here for free.

    A2 (seed fallback): for split-sentence attribution (`"...!" Thomas muttered.`)
    the quote is its own sentence with no link to the verb, so we fall back to the
    first seed speech verb shortly after a closing quote.
    """
    spans: list[Span] = []
    seen: set[Span] = set()

    def add(token: Token) -> None:
        span = (token.idx, token.idx + len(token.text))
        if span not in seen:
            seen.add(span)
            spans.append(span)

    # A1: verb outside quotes governing a token inside a quote.
    for token in doc:
        if token.pos_ != "VERB" or is_inside_regions(token.idx, regions):
            continue
        for child in token.children:
            if child.dep_ == "punct":
                continue
            if is_inside_regions(child.idx, regions):
                add(token)
                break

    # A2: first seed speech verb just after a closing quote (split sentences).
    for _, close_idx in regions:
        window_end = min(len(doc.text), close_idx + TAG_WINDOW_CHARS)
        for token in doc:
            if token.idx <= close_idx or token.idx > window_end:
                continue
            if token.pos_ != "VERB":
                continue
            lemma = token.lemma_.lower()
            if lemma in ATTRIBUTION_VERBS or token.text.lower() in ATTRIBUTION_VERBS:
                add(token)
                break

    return spans


def detect_passive(doc: Doc, taken: list[Span], regions: list[Span]) -> list[Span]:
    spans: list[Span] = []
    seen: set[Span] = set()

    for token in doc:
        if is_inside_regions(token.idx, regions):
            continue
        span = passive_construction_span(token)
        if span is None or span in seen:
            continue
        if any(spans_overlap(span, t) for t in taken):
            continue
        seen.add(span)
        spans.append(span)

    return spans


def has_finite_aux(token: Token) -> bool:
    """True if a finite auxiliary/modal carries this verb (makes it finite)."""
    return any(
        child.dep_ in {"aux", "auxpass"} and child.tag_ in FINITE_AUX_TAGS
        for child in token.children
    )


def is_finite_active_verb(token: Token) -> bool:
    """Active voice = a FINITE lexical clause, not a bare participle/infinitive.

    VBD/VBP/VBZ are inherently finite. Bare gerunds/participles (`throwing`,
    `not turning`) and infinitives (`to repeat`) are NOT active constructions;
    a VBG/VB only counts when a finite auxiliary carries it (`had been running`,
    `will go`).
    """
    if token.pos_ != "VERB":
        return False
    if is_passive_participle(token):
        return False
    if token.tag_ in FINITE_TAGS:
        return True
    if token.tag_ in NONFINITE_TAGS:
        return has_finite_aux(token)
    return False


def active_verb_span(token: Token) -> Span:
    start = token.idx
    end = token.idx + len(token.text)

    for child in token.children:
        if child.dep_ in {"aux", "neg"}:
            start = min(start, child.idx)
            end = max(end, child.idx + len(child.text))

    return (start, end)


def detect_active(doc: Doc, taken: list[Span], regions: list[Span]) -> list[Span]:
    spans: list[Span] = []
    seen: set[Span] = set()

    for token in doc:
        if is_inside_regions(token.idx, regions):
            continue
        if not is_finite_active_verb(token):
            continue
        span = active_verb_span(token)
        if span in seen:
            continue
        if any(spans_overlap(span, t) for t in taken):
            continue
        seen.add(span)
        spans.append(span)

    return spans


def cp_spans_to_utf16(text: str, spans: list[Span]) -> list[list[int]]:
    return [to_utf16_span(text, s, e) for s, e in spans]


def analyze_paragraph(doc: Doc) -> dict:
    text = doc.text
    regions = find_quote_regions(text)

    tag_spans_cp = normalize_spans(detect_tags(doc, regions))

    passive_spans_cp = normalize_spans(detect_passive(doc, tag_spans_cp, regions))
    taken_for_active = tag_spans_cp + passive_spans_cp
    active_spans_cp = normalize_spans(detect_active(doc, taken_for_active, regions))

    passive_spans = cp_spans_to_utf16(text, passive_spans_cp)
    active_spans = cp_spans_to_utf16(text, active_spans_cp)
    tag_spans = cp_spans_to_utf16(text, tag_spans_cp)

    return {
        "passive": passive_spans,
        "active": active_spans,
        "tags": tag_spans,
        "counts": {
            "passive": len(passive_spans),
            "active": len(active_spans),
        },
    }
