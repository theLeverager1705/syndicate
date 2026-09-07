"""Text extraction with pixel boxes.

RapidOCR (ONNX) is used rather than Tesseract: it installs from pip with no
system binary, which matters when the whole thing has to work on a laptop
that was set up an hour ago.

Results are cached by file content hash. The verifier re-reads redacted images
constantly, and OCR is by far the slowest step in the pipeline.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "runs" / ".ocr_cache"

_engine = None


@dataclass
class Token:
    text: str
    bbox: tuple[int, int, int, int]   # x0, y0, x1, y1
    confidence: float                 # 0..1


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine


def _hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _quad_to_box(quad) -> tuple[int, int, int, int]:
    xs = [float(p[0]) for p in quad]
    ys = [float(p[1]) for p in quad]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def extract(image_path: str, use_cache: bool = True) -> list[Token]:
    """Return every text token found, with pixel-space bounding boxes."""
    key = _hash(image_path)
    cache_file = CACHE_DIR / (key + ".json")

    if use_cache and cache_file.exists():
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
        return [Token(t["text"], tuple(t["bbox"]), t["confidence"]) for t in raw]

    result, _elapse = _get_engine()(image_path)
    tokens: list[Token] = []
    for row in (result or []):
        quad, text, score = row[0], row[1], row[2]
        tokens.append(Token(str(text), _quad_to_box(quad), float(score)))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps([asdict(t) for t in tokens], indent=1), encoding="utf-8"
    )
    return tokens


if __name__ == "__main__":
    import sys

    for t in extract(sys.argv[1]):
        print(f"{t.confidence:.2f}  {str(t.bbox):<28} {t.text}")


def extract_array(array, use_cache: bool = False) -> list[Token]:
    """OCR a numpy RGB array. Used by the web app so an upload never hits disk."""
    result, _elapse = _get_engine()(array)
    tokens: list[Token] = []
    for row in (result or []):
        quad, text, score = row[0], row[1], row[2]
        tokens.append(Token(str(text), _quad_to_box(quad), float(score)))
    return tokens
