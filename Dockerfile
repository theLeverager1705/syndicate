FROM python:3.12-slim

# onnxruntime needs libgomp; opencv (a rapidocr dep) needs libgl and libglib.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Warm the OCR model at build time so the first user request is not the one
# that pays for a 10MB model download.
RUN python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()" || true

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn web.app:app --host 0.0.0.0 --port ${PORT}"]
