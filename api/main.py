"""FastAPI batch analysis service for ProseParse."""

from __future__ import annotations

import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from typing import Annotated, Any

import spacy
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from analysis import analyze_paragraph
from sensory import analyze_sensory_paragraph
from sensory_lexicon import SENSES, get_lexicon
from voice import EMBEDDING_DIM, analyze_voice_chapter

logger = logging.getLogger("proseparse.analysis")

# Guard against pathological input (e.g. a whole chapter sent as one "paragraph").
# spaCy's parser memory scales with doc length; reject oversized paragraphs with a
# clean 413 rather than letting spaCy raise or the container OOM.
MAX_CHARS_PER_PARAGRAPH = 100_000
MAX_PARAGRAPHS_PER_VOICE = 200
MINILM_MODEL = os.getenv("MINILM_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

nlp: spacy.Language | None = None
nlp_lock = threading.Lock()

# Voice pipeline models — lazy-loaded on first /voice request so /analyze keeps
# its lighter RAM footprint until voice is actually used.
nlp_ner: spacy.Language | None = None
nlp_ner_lock = threading.Lock()
embedder: Any | None = None
embedder_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global nlp
    # Auth is mandatory: refuse to boot without a key so a misconfigured deploy
    # can never serve traffic unauthenticated.
    if not os.getenv("ANALYSIS_API_KEY"):
        raise RuntimeError(
            "ANALYSIS_API_KEY is required. Set it (e.g. via Cloud Run "
            "--set-secrets ANALYSIS_API_KEY=...) before starting the service."
        )
    logger.info("Loading spaCy model en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    yield
    nlp = None


app = FastAPI(title="ProseParse Analysis", lifespan=lifespan)


class ParagraphIn(BaseModel):
    block: int = Field(ge=0)
    text: str = Field(min_length=1)


class AnalyzeRequest(BaseModel):
    sessionId: str | None = None
    batchIndex: int = Field(ge=0)
    paragraphs: list[ParagraphIn] = Field(min_length=1, max_length=20)

    @field_validator("paragraphs")
    @classmethod
    def paragraphs_non_empty(cls, paragraphs: list[ParagraphIn]) -> list[ParagraphIn]:
        for p in paragraphs:
            if not p.text.strip():
                raise ValueError("paragraph text must not be blank")
        return paragraphs


class VoiceCounts(BaseModel):
    passive: int = Field(ge=0)
    active: int = Field(ge=0)


class ParagraphResult(BaseModel):
    block: int
    passive: list[list[int]]
    active: list[list[int]]
    tags: list[list[int]]
    counts: VoiceCounts | None = None


class AnalyzeResponse(BaseModel):
    batchIndex: int
    results: list[ParagraphResult]


class VoiceRequest(BaseModel):
    sessionId: str | None = None
    batchIndex: int = Field(ge=0)
    paragraphs: list[ParagraphIn] = Field(
        min_length=1, max_length=MAX_PARAGRAPHS_PER_VOICE
    )

    @field_validator("paragraphs")
    @classmethod
    def paragraphs_non_empty(cls, paragraphs: list[ParagraphIn]) -> list[ParagraphIn]:
        for p in paragraphs:
            if not p.text.strip():
                raise ValueError("paragraph text must not be blank")
        return paragraphs


class StylometryTallies(BaseModel):
    sentenceCount: int = Field(ge=0)
    tokenCount: int = Field(ge=0)
    wordCount: int = Field(ge=0)
    charCount: int = Field(ge=0)
    contractionCount: int = Field(ge=0)
    punctuation: dict[str, int]
    posCounts: dict[str, int]


class DialogueSpan(BaseModel):
    block: int = Field(ge=0)
    span: list[int]  # UTF-16 half-open [start, end] within that paragraph


class CharacterVoice(BaseModel):
    name: str
    vector: list[float]
    stylometry: StylometryTallies
    uniqueLemmas: list[str]
    spans: list[DialogueSpan] = Field(default_factory=list)


class VoiceResponse(BaseModel):
    batchIndex: int
    characters: list[CharacterVoice]


class SensoryRequest(BaseModel):
    sessionId: str | None = None
    batchIndex: int = Field(ge=0)
    paragraphs: list[ParagraphIn] = Field(min_length=1, max_length=20)
    includeDialogue: bool = True
    debug: bool = False

    @field_validator("paragraphs")
    @classmethod
    def paragraphs_non_empty(cls, paragraphs: list[ParagraphIn]) -> list[ParagraphIn]:
        for p in paragraphs:
            if not p.text.strip():
                raise ValueError("paragraph text must not be blank")
        return paragraphs


class SensoryCounts(BaseModel):
    sight: int = Field(ge=0)
    sound: int = Field(ge=0)
    touch: int = Field(ge=0)
    smell: int = Field(ge=0)
    taste: int = Field(ge=0)


class SensoryDetail(BaseModel):
    span: list[int]
    sense: str
    confidence: float
    tier: int


class SensoryParagraphResult(BaseModel):
    block: int
    sight: list[list[int]]
    sound: list[list[int]]
    touch: list[list[int]]
    smell: list[list[int]]
    taste: list[list[int]]
    counts: SensoryCounts
    details: list[SensoryDetail] | None = None


class SensoryResponse(BaseModel):
    batchIndex: int
    results: list[SensoryParagraphResult]


def verify_api_key(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    expected = os.getenv("ANALYSIS_API_KEY")
    if not expected:
        # Should be unreachable (lifespan fails fast), but guard defensively so
        # we never fall open to unauthenticated access.
        raise HTTPException(status_code=500, detail="Server authentication is not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(token, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


def ensure_embedder() -> Any:
    """Lazy-load MiniLM once; shared by /voice and /sensory."""
    global embedder
    if embedder is None:
        with embedder_lock:
            if embedder is None:
                logger.info("Loading SentenceTransformer %s", MINILM_MODEL)
                from sentence_transformers import SentenceTransformer

                embedder = SentenceTransformer(MINILM_MODEL)
    assert embedder is not None
    return embedder


def ensure_voice_models() -> tuple[spacy.Language, Any]:
    """Lazy-load NER spaCy + MiniLM on first /voice call."""
    global nlp_ner

    if nlp_ner is None:
        with nlp_ner_lock:
            if nlp_ner is None:
                logger.info("Loading spaCy model en_core_web_sm (with NER) for /voice")
                nlp_ner = spacy.load("en_core_web_sm")

    model = ensure_embedder()
    assert nlp_ner is not None
    return nlp_ner, model


def concatenate_paragraphs(paragraphs: list[ParagraphIn]) -> str:
    """Join batch paragraphs in order with double newlines for attribution span."""
    return "\n\n".join(p.text for p in paragraphs)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_batch(
    body: AnalyzeRequest,
    _: Annotated[None, Depends(verify_api_key)] = None,
) -> AnalyzeResponse:
    if nlp is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    for p in body.paragraphs:
        if len(p.text) > MAX_CHARS_PER_PARAGRAPH:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Paragraph block {p.block} exceeds "
                    f"{MAX_CHARS_PER_PARAGRAPH} characters"
                ),
            )

    if body.sessionId:
        logger.info(
            "analyze batch=%s session=%s paragraphs=%s",
            body.batchIndex,
            body.sessionId,
            len(body.paragraphs),
        )

    texts = [p.text for p in body.paragraphs]
    blocks = [p.block for p in body.paragraphs]

    with nlp_lock:
        docs = list(nlp.pipe(texts))

    results: list[ParagraphResult] = []
    for block, doc in zip(blocks, docs):
        analyzed = analyze_paragraph(doc)
        results.append(
            ParagraphResult(
                block=block,
                passive=analyzed["passive"],
                active=analyzed["active"],
                tags=analyzed["tags"],
                counts=VoiceCounts(**analyzed["counts"]),
            )
        )

    return AnalyzeResponse(batchIndex=body.batchIndex, results=results)


@app.post("/voice", response_model=VoiceResponse)
def voice_batch(
    body: VoiceRequest,
    _: Annotated[None, Depends(verify_api_key)] = None,
) -> VoiceResponse:
    for p in body.paragraphs:
        if len(p.text) > MAX_CHARS_PER_PARAGRAPH:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Paragraph block {p.block} exceeds "
                    f"{MAX_CHARS_PER_PARAGRAPH} characters"
                ),
            )

    if body.sessionId:
        logger.info(
            "voice batch=%s session=%s paragraphs=%s",
            body.batchIndex,
            body.sessionId,
            len(body.paragraphs),
        )

    ner, model = ensure_voice_models()
    para_tuples = [(p.block, p.text) for p in body.paragraphs]
    batch_text = concatenate_paragraphs(body.paragraphs)

    with nlp_ner_lock:
        doc = ner(batch_text)

    with embedder_lock:
        characters = analyze_voice_chapter(
            doc,
            model,
            stylometry_nlp=ner,
            paragraphs=para_tuples,
        )

    for ch in characters:
        if len(ch["vector"]) != EMBEDDING_DIM:
            logger.warning(
                "unexpected embedding dim %s for %s (expected %s)",
                len(ch["vector"]),
                ch["name"],
                EMBEDDING_DIM,
            )

    return VoiceResponse(
        batchIndex=body.batchIndex,
        characters=[CharacterVoice(**ch) for ch in characters],
    )


@app.post("/sensory", response_model=SensoryResponse)
def sensory_batch(
    body: SensoryRequest,
    _: Annotated[None, Depends(verify_api_key)] = None,
) -> SensoryResponse:
    if nlp is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    for p in body.paragraphs:
        if len(p.text) > MAX_CHARS_PER_PARAGRAPH:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Paragraph block {p.block} exceeds "
                    f"{MAX_CHARS_PER_PARAGRAPH} characters"
                ),
            )

    if body.sessionId:
        logger.info(
            "sensory batch=%s session=%s paragraphs=%s",
            body.batchIndex,
            body.sessionId,
            len(body.paragraphs),
        )

    lexicon = get_lexicon()
    model = ensure_embedder()
    texts = [p.text for p in body.paragraphs]
    blocks = [p.block for p in body.paragraphs]

    with nlp_lock:
        docs = list(nlp.pipe(texts))

    results: list[SensoryParagraphResult] = []
    with embedder_lock:
        for block, doc in zip(blocks, docs):
            analyzed = analyze_sensory_paragraph(
                doc,
                lexicon,
                model,
                include_dialogue=body.includeDialogue,
                debug=body.debug,
            )
            details = None
            if body.debug and analyzed.get("details") is not None:
                details = [SensoryDetail(**d) for d in analyzed["details"]]
            results.append(
                SensoryParagraphResult(
                    block=block,
                    sight=analyzed["sight"],
                    sound=analyzed["sound"],
                    touch=analyzed["touch"],
                    smell=analyzed["smell"],
                    taste=analyzed["taste"],
                    counts=SensoryCounts(**analyzed["counts"]),
                    details=details,
                )
            )

    # Defensive: ensure every sense key is present (model schema already does).
    assert all(hasattr(r, s) for r in results for s in SENSES)

    return SensoryResponse(batchIndex=body.batchIndex, results=results)
