FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Hugging Face / sentence-transformers cache (pre-populated at build time).
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers \
    MINILM_MODEL=sentence-transformers/all-MiniLM-L6-v2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake MiniLM into the image so Cloud Run cold starts stay offline / predictable.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY analysis.py voice.py main.py ./

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
