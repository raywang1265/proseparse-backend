# ProseParse Backend

Backend API for ProseParse — a FastAPI service that analyzes prose for passive/active voice, dialogue tags, and per-character voice fingerprints. Deployed to Google Cloud Run via `cloudbuild.yaml`.

## What it does

- **`POST /analyze`** — Batch analysis of paragraphs: passive/active voice spans, dialogue tags, and voice counts (spaCy).
- **`POST /voice`** — Character voice extraction: dialogue attribution, stylometry, unique lemmas, embedding vectors, and dialogue spans (spaCy NER + MiniLM).
- **`GET /health`** — Liveness check (no auth).

All analysis endpoints require a Bearer token (`ANALYSIS_API_KEY`).

## Local setup

**Requirements:** Python 3.12+

```bash
# From the repo root
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r api/requirements.txt
```

Set the API key (required — the server will not start without it):

```bash
# Windows (PowerShell)
$env:ANALYSIS_API_KEY = "your-local-dev-key"

# macOS / Linux
export ANALYSIS_API_KEY="your-local-dev-key"
```

Run the server from the `api/` directory:

```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

The service listens on `http://localhost:8080`. `/analyze` loads spaCy at startup; `/voice` lazy-loads NER and the sentence-transformer model on first use (the first request may take longer while MiniLM downloads).

### Example request

```bash
curl -X POST http://localhost:8080/analyze \
  -H "Authorization: Bearer your-local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "batchIndex": 0,
    "paragraphs": [
      {"block": 0, "text": "The wind was carried inland like a rumor."}
    ]
  }'
```

### Tests

```bash
cd api
pytest
```

### Docker (optional)

```bash
docker build -f api/Dockerfile -t proseparse-analysis api
docker run --rm -p 8080:8080 -e ANALYSIS_API_KEY=your-local-dev-key proseparse-analysis
```

## Project layout

```
api/
  main.py        # FastAPI app and routes
  analysis.py    # Passive/active voice and dialogue tags
  voice.py       # Character voice extraction
  requirements.txt
  Dockerfile
scripts/
  probe_live.py  # Smoke-test a deployed instance
cloudbuild.yaml  # Cloud Run deploy pipeline
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANALYSIS_API_KEY` | Yes | Bearer token for `/analyze` and `/voice` |
| `MINILM_MODEL` | No | Sentence-transformer model (default: `sentence-transformers/all-MiniLM-L6-v2`) |
| `PORT` | No | HTTP port (default: `8080`) |
