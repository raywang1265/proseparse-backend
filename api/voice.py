"""Character voice extraction: dialogue attribution, stylometry, embeddings."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol

from gender_guesser.detector import Detector
from spacy.tokens import Doc, Token

from analysis import TAG_WINDOW_CHARS, detect_tags, find_quote_regions, to_utf16_span

UNKNOWN = "UNKNOWN"
PRONOUN_GENDER = {
    "he": "m",
    "him": "m",
    "his": "m",
    "himself": "m",
    "she": "f",
    "her": "f",
    "hers": "f",
    "herself": "f",
    "they": "n",
    "them": "n",
    "their": "n",
    "theirs": "n",
    "themselves": "n",
}
# Offline ~40k-name dictionary; created once (each Detector reads the data file).
_GENDER_DETECTOR = Detector(case_sensitive=False)
HONORIFICS = {"mr", "mrs", "ms", "miss", "dr", "sir", "lady", "lord", "prof", "professor"}
# Match clitics inside words (can't) and standalone ('ll).
CONTRACTION_RE = re.compile(
    r"(?:n't|'ll|'re|'ve|'d|'m|(?<=\w)'s)\b",
    re.IGNORECASE,
)
PUNCT_MARKS = (".", ",", "!", "?", ";", ":", "-", "'", '"', "…", "—", "–")
ADJACENT_NAME_WINDOW = 60
# Pronoun antecedents: look back this many *narrative* sentences, zipping past
# quote regions so a long speech does not exhaust the window.
PRONOUN_LOOKBACK_SENTENCES = 4
EMBEDDING_DIM = 384


class Embedder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


@dataclass
class PersonMention:
    start: int
    end: int
    text: str
    canonical: str
    gender: str | None  # "m" | "f" | "n" | None


@dataclass
class QuoteAssignment:
    open_idx: int
    close_idx: int
    dialogue: str
    speaker: str | None = None
    has_tag: bool = False


@dataclass
class CharacterCluster:
    name: str
    dialogues: list[str] = field(default_factory=list)


def _strip_honorific(name: str) -> str:
    parts = name.replace(".", " ").split()
    while parts and parts[0].lower().rstrip(".") in HONORIFICS:
        parts = parts[1:]
    return " ".join(parts).strip() or name.strip()


def canonicalize_name(name: str) -> str:
    cleaned = _strip_honorific(name)
    # Prefer the last token as the short key for merging "Mr. Thomas" / "Thomas".
    tokens = cleaned.split()
    if not tokens:
        return name.strip()
    if len(tokens) == 1:
        return tokens[0].title()
    # Keep multi-token form title-cased; short-form lookups use last token.
    return cleaned.title()


def short_name_key(canonical: str) -> str:
    tokens = canonical.split()
    return tokens[-1].lower() if tokens else canonical.lower()


def infer_gender_from_name(name: str) -> str | None:
    """Map a character name to m/f for pronoun resolution; None if ambiguous/unknown."""
    key = short_name_key(canonicalize_name(name))
    # Title-case for the detector's dictionary keys (case_sensitive=False still
    # benefits from a normalized first-letter form for uncommon spellings).
    guess = _GENDER_DETECTOR.get_gender(key.title())
    if guess in {"male", "mostly_male"}:
        return "m"
    if guess in {"female", "mostly_female"}:
        return "f"
    return None


def _spans_overlap_chars(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def extract_persons(doc: Doc, quote_regions: list[tuple[int, int]] | None = None) -> list[PersonMention]:
    """PERSON NER mentions, plus outside-quote PROPN fallback when NER misses names."""
    regions = quote_regions or []
    mentions: list[PersonMention] = []
    covered: list[tuple[int, int]] = []

    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        canon = canonicalize_name(ent.text)
        mentions.append(
            PersonMention(
                start=ent.start_char,
                end=ent.end_char,
                text=ent.text,
                canonical=canon,
                gender=infer_gender_from_name(canon),
            )
        )
        covered.append((ent.start_char, ent.end_char))

    # Fallback: capitalized PROPN tokens outside quotes (en_core_web_sm often misses
    # uncommon given names like "Mara").
    for token in doc:
        if token.pos_ != "PROPN" or not token.text[:1].isupper():
            continue
        if any(o < token.idx < c for o, c in regions):
            continue
        if any(_spans_overlap_chars(token.idx, token.idx + len(token.text), s, e) for s, e in covered):
            continue
        if token.text.lower().rstrip(".") in HONORIFICS:
            continue
        canon = canonicalize_name(token.text)
        mentions.append(
            PersonMention(
                start=token.idx,
                end=token.idx + len(token.text),
                text=token.text,
                canonical=canon,
                gender=infer_gender_from_name(canon),
            )
        )
        covered.append((token.idx, token.idx + len(token.text)))
    return mentions


def merge_name_aliases(mentions: list[PersonMention]) -> dict[str, str]:
    """Map short-name keys to a single cluster display name (title-cased last token)."""
    by_short: dict[str, str] = {}
    for m in mentions:
        key = short_name_key(m.canonical)
        by_short[key] = key.title()
    return by_short


def resolve_person_token(
    token: Token,
    mentions: list[PersonMention],
    alias_map: dict[str, str],
) -> str | None:
    """If token is (part of) a PERSON mention, return its cluster name."""
    for m in mentions:
        if m.start <= token.idx < m.end:
            return alias_map.get(short_name_key(m.canonical), m.canonical)
    if token.pos_ == "PROPN":
        key = token.text.lower()
        if key in alias_map:
            return alias_map[key]
        return token.text.title()
    return None


def _token_for_tag_span(doc: Doc, span: tuple[int, int]) -> Token | None:
    s, e = span
    for token in doc:
        if s <= token.idx < e:
            return token
    return None


def bind_tags_to_quotes(
    doc: Doc,
    quotes: list[tuple[int, int]],
    tag_spans: list[tuple[int, int]],
) -> dict[int, Token]:
    """Map each quote index to at most one tag verb.

    Each tag binds to exactly one quote — the nearest eligible neighbor — so a
    speech verb cannot "leak" onto later untagged dialogue lines.
    """
    if not quotes or not tag_spans:
        return {}

    # Candidate (quote_idx, tag_idx, dist, side) where side 0=after-close, 1=before-open.
    candidates: list[tuple[int, int, int, int]] = []
    for ti, (ts, te) in enumerate(tag_spans):
        for qi, (open_idx, close_idx) in enumerate(quotes):
            prev_close = quotes[qi - 1][1] if qi > 0 else -10**9
            next_open = quotes[qi + 1][0] if qi + 1 < len(quotes) else 10**9

            # Post-quote attribution: tag sits after this close and before next open.
            if close_idx <= ts < next_open:
                dist = ts - close_idx
                if dist <= TAG_WINDOW_CHARS:
                    candidates.append((qi, ti, dist, 0))

            # Pre-quote attribution: tag sits after previous close and before this open.
            if prev_close < te <= open_idx:
                dist = open_idx - te
                if dist <= TAG_WINDOW_CHARS:
                    candidates.append((qi, ti, dist, 1))

    # Prefer post-quote, then smaller distance; each tag and each quote used once.
    candidates.sort(key=lambda c: (c[3], c[2], c[0], c[1]))
    used_quotes: set[int] = set()
    used_tags: set[int] = set()
    bound: dict[int, Token] = {}
    for qi, ti, _dist, _side in candidates:
        if qi in used_quotes or ti in used_tags:
            continue
        token = _token_for_tag_span(doc, tag_spans[ti])
        if token is None:
            continue
        bound[qi] = token
        used_quotes.add(qi)
        used_tags.add(ti)

    # Syntactic governance fallback for unbound quotes.
    for qi, (open_idx, close_idx) in enumerate(quotes):
        if qi in bound:
            continue
        for token in doc:
            if token.pos_ != "VERB":
                continue
            if open_idx < token.idx < close_idx:
                continue
            for child in token.children:
                if child.dep_ == "punct":
                    continue
                if open_idx < child.idx < close_idx:
                    bound[qi] = token
                    break
            if qi in bound:
                break
    return bound


def find_tag_verb_for_quote(
    doc: Doc,
    quote: tuple[int, int],
    tag_spans: list[tuple[int, int]],
    *,
    quotes: list[tuple[int, int]] | None = None,
    quote_index: int | None = None,
    bound: dict[int, Token] | None = None,
) -> Token | None:
    """Compatibility wrapper: prefer precomputed 1:1 tag bindings when available."""
    if bound is not None and quote_index is not None:
        return bound.get(quote_index)
    all_quotes = quotes or [quote]
    idx = quote_index if quote_index is not None else 0
    return bind_tags_to_quotes(doc, all_quotes, tag_spans).get(idx)


def subject_of_verb(
    verb: Token,
    *,
    quote: tuple[int, int] | None = None,
) -> Token | None:
    """Return the speech-verb subject, preferring tokens *outside* the quote.

    Inverted dialogue like ``\"Hello!\" said Isabel`` often gets the in-quote
    content marked as ``nsubj`` of ``said`` as well as the real speaker. Taking
    the first ``nsubj`` then promotes capitalized dialogue (e.g. a fantasy noun)
    into a fake character name.
    """
    open_idx, close_idx = quote if quote is not None else (None, None)

    def outside_quote(tok: Token) -> bool:
        if open_idx is None or close_idx is None:
            return True
        return not (open_idx < tok.idx < close_idx)

    def pick(candidates: list[Token]) -> Token | None:
        if not candidates:
            return None
        # Prefer the subject closest to the verb (usually the post-quote name).
        return min(candidates, key=lambda t: abs(t.idx - verb.idx))

    outside = [
        c
        for c in verb.children
        if c.dep_ in {"nsubj", "nsubjpass"} and outside_quote(c)
    ]
    chosen = pick(outside)
    if chosen is not None:
        return chosen

    # No outside subject — do NOT fall back to in-quote nsubj (that is dialogue).
    head = verb.head
    if head is not verb and head.pos_ in {"VERB", "AUX"}:
        outside_head = [
            c
            for c in head.children
            if c.dep_ in {"nsubj", "nsubjpass"} and outside_quote(c)
        ]
        return pick(outside_head)
    return None


def _span_fully_inside_regions(
    start: int, end: int, regions: list[tuple[int, int]]
) -> bool:
    """True if [start, end) lies entirely inside some quote region."""
    return any(o <= start and end <= c for o, c in regions)


def _char_inside_regions(idx: int, regions: list[tuple[int, int]]) -> bool:
    return any(o < idx < c for o, c in regions)


def narrative_lookback_window_start(
    doc: Doc,
    pronoun: Token,
    regions: list[tuple[int, int]],
    *,
    max_sentences: int = PRONOUN_LOOKBACK_SENTENCES,
) -> int:
    """Start char of the pronoun lookback window (2–4 narrative sentences).

    Walks spaCy sentences backward from the pronoun. Sentences that lie fully
    inside a quote region are skipped (do not consume the budget), so long
    character speeches cannot push earlier narrative names out of range.
    """
    sents = list(doc.sents)
    if not sents:
        return 0

    pronoun_sent_i = 0
    for i, sent in enumerate(sents):
        if sent.start_char <= pronoun.idx < sent.end_char:
            pronoun_sent_i = i
            break

    counted = 0
    window_start = sents[pronoun_sent_i].start_char
    for i in range(pronoun_sent_i, -1, -1):
        sent = sents[i]
        if _span_fully_inside_regions(sent.start_char, sent.end_char, regions):
            continue
        counted += 1
        window_start = sent.start_char
        if counted >= max_sentences:
            break
    return window_start


def resolve_pronoun_speaker(
    pronoun: Token,
    mentions: list[PersonMention],
    alias_map: dict[str, str],
    known_genders: dict[str, str],
    doc: Doc,
    regions: list[tuple[int, int]],
) -> str | None:
    gender = PRONOUN_GENDER.get(pronoun.text.lower())
    if gender is None:
        return None

    window_start = narrative_lookback_window_start(doc, pronoun, regions)
    candidates: list[PersonMention] = []
    for m in mentions:
        if m.end > pronoun.idx or m.start < window_start:
            continue
        # Zip past names inside quoted speech (vocatives / quoted mentions).
        if _char_inside_regions(m.start, regions):
            continue
        cluster = alias_map.get(short_name_key(m.canonical), m.canonical)
        g = known_genders.get(cluster) or m.gender
        if gender == "n":
            candidates.append(m)
        elif g == gender:
            candidates.append(m)
        elif g is None and gender in {"m", "f"}:
            # Unknown gender — weaker candidate; only use if no better match.
            candidates.append(m)
    if not candidates:
        return None
    # Prefer gender-agreeing, then most recent.
    agreeing = [
        m
        for m in candidates
        if (known_genders.get(alias_map.get(short_name_key(m.canonical), m.canonical)) or m.gender)
        == gender
        or gender == "n"
    ]
    pool = agreeing if agreeing else candidates
    best = max(pool, key=lambda m: m.end)
    return alias_map.get(short_name_key(best.canonical), best.canonical)


def nearest_person_near_quote(
    quote: tuple[int, int],
    mentions: list[PersonMention],
    alias_map: dict[str, str],
    *,
    exclude_inside_quote: bool = True,
) -> str | None:
    open_idx, close_idx = quote
    best: PersonMention | None = None
    best_dist = 10**9
    for m in mentions:
        if exclude_inside_quote and open_idx < m.start < close_idx:
            # Vocative / addressee inside the quote — listener, not speaker.
            continue
        # Prefer mentions just after the close (split-sentence) or just before open.
        if m.start >= close_idx:
            dist = m.start - close_idx
        elif m.end <= open_idx:
            dist = open_idx - m.end
        else:
            continue
        if dist <= ADJACENT_NAME_WINDOW and dist < best_dist:
            best = m
            best_dist = dist
    if best is None:
        return None
    return alias_map.get(short_name_key(best.canonical), best.canonical)


def quote_has_nearby_tag(
    quote: tuple[int, int],
    tag_spans: list[tuple[int, int]],
) -> bool:
    open_idx, close_idx = quote
    for s, e in tag_spans:
        if close_idx <= s <= close_idx + TAG_WINDOW_CHARS:
            return True
        if open_idx - TAG_WINDOW_CHARS <= e <= open_idx:
            return True
    return False


def narrative_introduces_person(
    doc: Doc,
    between_start: int,
    between_end: int,
    mentions: list[PersonMention],
) -> bool:
    """True if narrative (non-quote) text between two quotes re-introduces a PERSON subject."""
    if between_end <= between_start:
        return False
    for m in mentions:
        if between_start <= m.start < between_end:
            return True
    # Also check for a finite clause with a PROPN subject in the gap.
    for token in doc:
        if token.idx < between_start or token.idx >= between_end:
            continue
        if token.dep_ in {"nsubj", "nsubjpass"} and token.pos_ == "PROPN":
            return True
    return False


def attribute_quotes(doc: Doc) -> list[QuoteAssignment]:
    text = doc.text
    regions = find_quote_regions(text)
    tag_spans_cp = detect_tags(doc, regions)
    mentions = extract_persons(doc, regions)
    alias_map = merge_name_aliases(mentions)
    bound = bind_tags_to_quotes(doc, regions, tag_spans_cp)

    known_genders: dict[str, str] = {}
    for m in mentions:
        cluster = alias_map.get(short_name_key(m.canonical), m.canonical)
        if m.gender and cluster not in known_genders:
            known_genders[cluster] = m.gender

    assignments: list[QuoteAssignment] = []
    for qi, (open_idx, close_idx) in enumerate(regions):
        dialogue = text[open_idx + 1 : close_idx].strip()
        if not dialogue:
            continue
        verb = bound.get(qi)
        qa = QuoteAssignment(
            open_idx=open_idx,
            close_idx=close_idx,
            dialogue=dialogue,
            has_tag=verb is not None,
        )

        # Layer 1–2: explicit tag verb subject (named or pronoun).
        if verb is not None:
            subj = subject_of_verb(verb, quote=(open_idx, close_idx))
            # Belt-and-suspenders: never treat in-quote tokens as the speaker.
            if subj is not None and not (open_idx < subj.idx < close_idx):
                person = resolve_person_token(subj, mentions, alias_map)
                if person is not None:
                    qa.speaker = person
                else:
                    resolved = resolve_pronoun_speaker(
                        subj, mentions, alias_map, known_genders, doc, regions
                    )
                    if resolved is not None:
                        qa.speaker = resolved
                        g = PRONOUN_GENDER.get(subj.text.lower())
                        if g in {"m", "f"}:
                            known_genders[resolved] = g

        # Layer 3: adjacent / split-sentence name — only when a tag is bound.
        if qa.speaker is None and qa.has_tag:
            nearby = nearest_person_near_quote(
                (open_idx, close_idx), mentions, alias_map
            )
            if nearby is not None:
                qa.speaker = nearby

        assignments.append(qa)

    # Layer 4: rapid-fire tagless alternation.
    _apply_alternation(doc, assignments, mentions)
    for qa in assignments:
        if qa.speaker is None:
            qa.speaker = UNKNOWN
    return assignments


def _apply_alternation(
    doc: Doc,
    assignments: list[QuoteAssignment],
    mentions: list[PersonMention],
) -> None:
    i = 0
    while i < len(assignments):
        if assignments[i].speaker is not None or assignments[i].has_tag:
            i += 1
            continue
        # Start of a tagless unattributed run.
        run_start = i
        while (
            i < len(assignments)
            and assignments[i].speaker is None
            and not assignments[i].has_tag
        ):
            # Break run if narrative between this quote and the previous
            # re-introduces a PERSON (for i > run_start).
            if i > run_start:
                prev = assignments[i - 1]
                curr = assignments[i]
                if narrative_introduces_person(
                    doc, prev.close_idx, curr.open_idx, mentions
                ):
                    break
            i += 1
        run_end = i
        run = assignments[run_start:run_end]

        # Seed A/B from the last two distinct attributed speakers before the run.
        seeds: list[str] = []
        for j in range(run_start - 1, -1, -1):
            sp = assignments[j].speaker
            if sp is None or sp == UNKNOWN:
                continue
            if sp not in seeds:
                seeds.append(sp)
            if len(seeds) == 2:
                break
        if len(seeds) < 2:
            # Leave as None → UNKNOWN later.
            continue
        speaker_a, speaker_b = seeds[0], seeds[1]
        # Alternate B, A, B, A... (next speaker is the other one).
        for k, qa in enumerate(run):
            qa.speaker = speaker_b if k % 2 == 0 else speaker_a


def compute_stylometry(dialogue_texts: list[str], nlp_doc_factory: Any) -> dict[str, Any]:
    """Return raw integer tallies + unique lemma list for frontend aggregation."""
    joined = " ".join(dialogue_texts)
    doc: Doc = nlp_doc_factory(joined) if joined.strip() else nlp_doc_factory("")

    punct_counts: dict[str, int] = {p: 0 for p in PUNCT_MARKS}
    for ch in joined:
        if ch in punct_counts:
            punct_counts[ch] += 1

    contraction_count = len(CONTRACTION_RE.findall(joined))

    pos_counts: dict[str, int] = defaultdict(int)
    unique_lemmas: set[str] = set()
    token_count = 0
    word_count = 0
    for token in doc:
        if token.is_space:
            continue
        token_count += 1
        pos_counts[token.pos_] += 1
        if token.is_alpha:
            word_count += 1
            unique_lemmas.add(token.lemma_.lower())

    sentence_count = sum(1 for _ in doc.sents) if joined.strip() else 0

    return {
        "sentenceCount": sentence_count,
        "tokenCount": token_count,
        "wordCount": word_count,
        "charCount": len(joined),
        "contractionCount": contraction_count,
        "punctuation": punct_counts,
        "posCounts": dict(pos_counts),
        "uniqueLemmas": sorted(unique_lemmas),
    }


def cluster_dialogue(assignments: list[QuoteAssignment]) -> dict[str, list[str]]:
    clusters: dict[str, list[str]] = defaultdict(list)
    for qa in assignments:
        speaker = qa.speaker or UNKNOWN
        clusters[speaker].append(qa.dialogue)
    return dict(clusters)


def build_paragraph_slices(
    paragraphs: list[tuple[int, str]],
) -> tuple[str, list[tuple[int, int, int, str]]]:
    """Join paragraphs with ``\\n\\n`` and return (joined_text, slices).

    Each slice is ``(block, start, end, text)`` in joined-text code-point offsets
    (half-open ``[start, end)`` covering that paragraph's text only).
    """
    parts: list[str] = []
    slices: list[tuple[int, int, int, str]] = []
    pos = 0
    for i, (block, text) in enumerate(paragraphs):
        start = pos
        end = pos + len(text)
        slices.append((block, start, end, text))
        parts.append(text)
        pos = end
        if i < len(paragraphs) - 1:
            pos += 2  # "\n\n"
    return "\n\n".join(parts), slices


def _quote_abs_end(close_idx: int, doc_len: int) -> int:
    """Half-open end covering the closing quote mark when present."""
    return close_idx if close_idx >= doc_len else close_idx + 1


def map_assignments_to_paragraph_spans(
    assignments: list[QuoteAssignment],
    slices: list[tuple[int, int, int, str]],
    doc_len: int,
) -> dict[str, list[dict[str, Any]]]:
    """Map attributed quotes to per-speaker ``{block, span}`` UTF-16 spans.

    ``span`` is a half-open UTF-16 ``[start, end]`` relative to that paragraph's
    text (same convention as ``/analyze``), covering the quote including marks.
    Quotes that cross a paragraph boundary are clipped to the paragraph that
    contains the opening mark.
    """
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not slices:
        return {}

    for qa in assignments:
        speaker = qa.speaker or UNKNOWN
        abs_start = qa.open_idx
        abs_end = _quote_abs_end(qa.close_idx, doc_len)

        owner: tuple[int, int, int, str] | None = None
        for block, start, end, text in slices:
            if start <= abs_start < end:
                owner = (block, start, end, text)
                break
        if owner is None:
            continue

        block, para_start, para_end, para_text = owner
        local_start = max(0, abs_start - para_start)
        local_end = min(para_end, abs_end) - para_start
        if local_start >= local_end:
            continue
        by_speaker[speaker].append(
            {
                "block": block,
                "span": to_utf16_span(para_text, local_start, local_end),
            }
        )
    return dict(by_speaker)


def embed_character_texts(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    prepared = [t if t.strip() else " " for t in texts]
    vectors = embedder.encode(prepared, normalize_embeddings=True)
    return [[float(x) for x in row] for row in vectors]


def analyze_voice_chapter(
    doc: Doc,
    embedder: Embedder,
    *,
    stylometry_nlp: Any,
    paragraphs: list[tuple[int, str]] | None = None,
) -> list[dict[str, Any]]:
    """Attribute quotes, then emit per-character vectors, tallies, and dialogue spans.

    ``paragraphs`` is ``[(block, text), ...]`` in the same order used to build
    ``doc``. When omitted, ``spans`` lists are empty.
    """
    assignments = attribute_quotes(doc)
    clusters = cluster_dialogue(assignments)

    span_map: dict[str, list[dict[str, Any]]] = {}
    if paragraphs is not None:
        _, slices = build_paragraph_slices(paragraphs)
        span_map = map_assignments_to_paragraph_spans(
            assignments, slices, len(doc.text)
        )

    names = sorted(clusters.keys(), key=lambda n: (n == UNKNOWN, n.lower()))
    dialogues = [" ".join(clusters[n]) for n in names]
    vectors = embed_character_texts(embedder, dialogues)

    results: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        sty = compute_stylometry(clusters[name], stylometry_nlp)
        unique_lemmas = sty.pop("uniqueLemmas")
        results.append(
            {
                "name": name,
                "vector": vectors[i] if i < len(vectors) else [0.0] * EMBEDDING_DIM,
                "stylometry": sty,
                "uniqueLemmas": unique_lemmas,
                "spans": span_map.get(name, []),
            }
        )
    return results
