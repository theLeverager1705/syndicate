FROM python:3.12-slim

# onnxruntime needs libgomp; opencv (a rapidocr dep) needs libgl and libglib.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces runs containers as uid 1000, and the app writes its local
# rule mirror and schema marker into the working directory. Without this the
# fallback store cannot write and every redaction fails the moment the hosted
# database is unavailable -- exactly when the fallback is needed.
RUN useradd -m -u 1000 appuser && mkdir -p /app/memory && chown -R appuser:appuser /app
USER appuser
ENV HOME=/home/appuser

# Warm the OCR model at build time so the first user request is not the one
# that pays for a 10MB model download.
RUN python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()" || true

# 7860 is what Hugging Face Spaces expects. Render and most other hosts set
# PORT themselves, which overrides this, so the same image works on both.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn web.app:app --host 0.0.0.0 --port ${PORT}"]
