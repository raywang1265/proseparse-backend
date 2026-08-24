# ProseParse Backend

Backend API for ProseParse — a FastAPI service that analyzes prose for passive/active voice, dialogue tags, per-character voice fingerprints, five-sense sensory description spans, and direct vs indirect exposition. Deployed to Google Cloud Run via `cloudbuild.yaml`.

## What it does

- **`POST /analyze`** — Batch analysis of paragraphs: passive/active voice spans, dialogue tags, and voice counts (spaCy).
- **`POST /voice`** — Character voice extraction: dialogue attribution, stylometry, unique lemmas, embedding vectors, and dialogue spans (spaCy NER + MiniLM).
- **`POST /sensory`** — Five-sense trigger-word spans (`sight` / `sound` / `touch` / `smell` / `taste`) via Lancaster lexicon + MiniLM sentence embeddings. Same batching and UTF-16 span contract as `/analyze`.
- **`POST /exposition`** — Paragraph-level direct vs indirect exposition scores from a fine-tuned DeBERTa-v3 classifier (`gu1npen/proseparse-exposition-finetune`). Same batching contract as `/analyze`. The Hub repo's `main` branch must contain the checkpoint (`config.json`, `model.safetensors`, `tokenizer.json`); until then set `EXPOSITION_REVISION` to a branch such as `experiment1-epoch3-deberta`.
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

The service listens on `http://localhost:8080`. `/analyze` loads spaCy at startup; `/voice` and `/sensory` lazy-load MiniLM on first use (shared embedder); `/exposition` lazy-loads the DeBERTa classifier. The first request for a lazy model may take longer while weights download locally — both models are baked into the Docker image.

### Example requests

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

```bash
curl -X POST http://localhost:8080/sensory \
  -H "Authorization: Bearer your-local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "batchIndex": 0,
    "paragraphs": [
      {"block": 0, "text": "The pungent aroma of garlic filled the kitchen."}
    ]
  }'
```

`/sensory` response shape (UTF-16 half-open spans relative to each paragraph):

```json
{
  "batchIndex": 0,
  "results": [
    {
      "block": 0,
      "sight": [],
      "sound": [],
      "touch": [],
      "smell": [[4, 17]],
      "taste": [[21, 27]],
      "counts": { "sight": 0, "sound": 0, "touch": 0, "smell": 1, "taste": 1 }
    }
  ]
}
```

Optional request flags: `includeDialogue` (default `true`), `debug` (adds per-span `details` with confidence/tier).

```bash
curl -X POST http://localhost:8080/exposition \
  -H "Authorization: Bearer your-local-dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "batchIndex": 0,
    "paragraphs": [
      {"block": 0, "text": "Thomas was a kind man who had always been afraid of the dark."}
    ]
  }'
```

`/exposition` response shape (`pDirect` is softmax P(direct); `directShare` is `round(100 * pDirect)`):

```json
{
  "batchIndex": 0,
  "results": [
    {
      "block": 0,
      "label": "direct",
      "pDirect": 0.8123,
      "directShare": 81,
      "truncated": false
    }
  ]
}
```

`truncated` is `true` when the paragraph exceeded the model's 384-token window.

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

### Rebuilding the sensory lexicon

The committed artifact `api/data/sensory_lexicon.json.gz` is a filtered derivative of the
[Lancaster Sensorimotor Norms](https://osf.io/7emr6/) (CC BY 4.0). To rebuild after
changing filters in `scripts/build_sensory_lexicon.py`:

```bash
# Download Lancaster_sensorimotor_norms_for_39707_words.csv from
# https://osf.io/download/48wsc/
python scripts/build_sensory_lexicon.py path/to/Lancaster_sensorimotor_norms_for_39707_words.csv
```

See [api/data/LANCASTER_ATTRIBUTION.md](api/data/LANCASTER_ATTRIBUTION.md) for citation and license details.

## Project layout

```
api/
  main.py              # FastAPI app and routes
  analysis.py          # Passive/active voice and dialogue tags
  voice.py             # Character voice extraction
  sensory.py           # Five-sense trigger spans
  sensory_lexicon.py   # Lancaster lexicon loader
  exposition.py        # Direct vs indirect exposition classifier
  data/                # sensory_lexicon.json.gz + attribution
  requirements.txt
  Dockerfile
scripts/
  probe_live.py            # Smoke-test a deployed instance
  build_sensory_lexicon.py # Rebuild lexicon from Lancaster CSV
cloudbuild.yaml            # Cloud Run deploy pipeline
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANALYSIS_API_KEY` | Yes | Bearer token for `/analyze`, `/voice`, `/sensory`, and `/exposition` |
| `MINILM_MODEL` | No | Sentence-transformer model (default: `sentence-transformers/all-MiniLM-L6-v2`) |
| `EXPOSITION_MODEL` | No | Hub repo id for the exposition classifier (default: `gu1npen/proseparse-exposition-finetune`) |
| `EXPOSITION_REVISION` | No | Hub revision / branch (default: `main`) |
| `TORCH_NUM_THREADS` | No | Intra-op thread count for the classifier (default: `2`) |
| `PORT` | No | HTTP port (default: `8080`) |
