"""FastAPI batch analysis service for ProseParse."""

from __future__ import annotations

import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from typing import Annotated

import spacy
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from analysis import analyze_paragraph

logger = logging.getLogger("proseparse.analysis")

# Guard against pathological input (e.g. a whole chapter sent as one "paragraph").
# spaCy's parser memory scales with doc length; reject oversized paragraphs with a
# clean 413 rather than letting spaCy raise or the container OOM.
MAX_CHARS_PER_PARAGRAPH = 100_000

nlp: spacy.Language | None = None
nlp_lock = threading.Lock()


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
