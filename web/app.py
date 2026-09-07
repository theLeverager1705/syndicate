"""Verity on the web.

Privacy posture of the hosted version, stated plainly because it matters:

  * Uploaded images are held in memory only, never written to disk, and dropped
    after a few minutes or as soon as the redaction is returned.
  * Only learned RULES reach Evorozen Neural DB -- field name, decision,
    confidence, layout anchors. Never the image, never the OCR text, never the
    identifier itself.
  * Each browser gets its own anonymous id, so one person's learned policy
    never governs another's post.
"""
from __future__ import annotations

import io
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from agent import classify, ocr, redact, verify
from agent.memory import PolicyMemory
from agent.reflect import Decision, reflect_on_run
from agent.store import default_store

app = FastAPI(title="Verity")

STATIC = __file__.replace("app.py", "static")
PENDING_TTL = 600          # seconds an uploaded image may sit in memory
MAX_UPLOAD = 8 * 1024 * 1024

# Longest edge fed to OCR. Real users upload phone photos: a 12MP image is
# ~36MB decoded, which is most of a small container's memory before OCR has
# allocated anything. Downscaling costs no accuracy at document scale -- text
# stays well above the detector's minimum size -- and cuts inference time
# several-fold. Boxes are scaled back to original coordinates before redaction,
# so the output is full resolution.
MAX_EDGE = int(os.environ.get('VERITY_MAX_EDGE', '1100'))


@dataclass
class Pending:
    """An upload awaiting the user's decisions. Memory only, never disk."""
    image: bytes
    tokens: list
    proposals: list
    context: dict[str, str]
    created: float


_pending: dict[str, Pending] = {}

# Finished redactions, held briefly so the browser can fetch them as binary.
# Base64-in-JSON inflates a payload by a third and forces the whole image
# through the JSON parser; a 9MP result is ~20MB that way, which is rough on a
# phone. Serving bytes from a short-lived handle keeps the response small.
_results: dict[str, tuple[bytes, float]] = {}


def _sweep() -> None:
    cutoff = time.time() - PENDING_TTL
    for key in [k for k, v in _pending.items() if v.created < cutoff]:
        _pending.pop(key, None)
    for key in [k for k, (_, t) in _results.items() if t < cutoff]:
        _results.pop(key, None)


def _memory_for(user_id: str) -> PolicyMemory:
    """Policy scoped to one anonymous user."""
    return PolicyMemory(store=default_store(), namespace=user_id)


def _context(platform: str, doc_type: str) -> dict[str, str]:
    return {
        "doc_type": doc_type or "document",
        "platform": platform or "linkedin",
        "issuer": "",
    }


@app.post("/api/screen")
async def screen(
    file: UploadFile,
    user_id: str = Form(...),
    platform: str = Form("linkedin"),
    doc_type: str = Form("document"),
) -> JSONResponse:
    """Read an image and propose what to hide. Nothing is modified yet."""
    _sweep()
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "image too large (8MB max)")

    try:
        Image.open(io.BytesIO(raw)).verify()
    except Exception:
        raise HTTPException(400, "not a readable image")

    # RapidOCR wants a path or array; decode from bytes in memory.
    import numpy as np

    img = Image.open(io.BytesIO(raw)).convert("RGB")

    scale = 1.0
    longest = max(img.size)
    if longest > MAX_EDGE:
        scale = MAX_EDGE / longest
        small = img.resize((max(1, round(img.width * scale)),
                            max(1, round(img.height * scale))), Image.LANCZOS)
    else:
        small = img

    tokens = ocr.extract_array(np.array(small))

    if scale != 1.0:
        # Map boxes back onto the full-resolution image the user will download.
        inv = 1.0 / scale
        for t in tokens:
            x0, y0, x1, y1 = t.bbox
            t.bbox = (int(x0 * inv), int(y0 * inv), int(x1 * inv), int(y1 * inv))

    memory = _memory_for(user_id)
    context = _context(platform, doc_type)
    rules = memory.retrieve(context)
    earned = memory.earned(context)
    proposals, usage = classify.propose(tokens, context, rules, earned)

    token_id = uuid.uuid4().hex
    _pending[token_id] = Pending(raw, tokens, proposals, context, time.time())

    items = []
    for i, p in enumerate(proposals):
        idx = p.token_indices[0] if p.token_indices else None
        items.append({
            "i": i,
            "field": p.field_name,
            "value": tokens[idx].text if idx is not None else "",
            "decision": p.decision,
            "source": p.source,
            "needs_you": p.needs_user,
            "why": p.rationale,
            "box": list(tokens[idx].bbox) if idx is not None else None,
        })

    return JSONResponse({
        "token": token_id,
        "width": img.width,
        "height": img.height,
        "proposals": items,
        "from_memory": sum(1 for p in proposals if p.source == "memory"),
        "asked": sum(1 for p in proposals if p.needs_user),
        "model_calls": usage.model_calls,
        "tokens_out": usage.tokens_out,
    })


@app.post("/api/apply")
async def apply(payload: dict[str, Any]) -> JSONResponse:
    """Apply the user's decisions, verify the result, and learn from it."""
    _sweep()
    token_id = payload.get("token", "")
    user_id = payload.get("user_id", "")
    choices: dict[str, str] = payload.get("decisions", {})

    pending = _pending.get(token_id)
    if pending is None:
        raise HTTPException(410, "upload expired, please try again")

    to_redact: list[int] = []
    decisions: list[Decision] = []
    for i, p in enumerate(pending.proposals):
        final = choices.get(str(i), p.decision)
        if final == "redact":
            to_redact.extend(p.token_indices)
        idx = p.token_indices[0] if p.token_indices else None
        decisions.append(Decision(
            p.field_name, p.decision, final, p.source, p.rule_id,
            asked_user=p.needs_user,
            anchor=classify._label_for(pending.tokens, idx) if idx is not None else None,
        ))

    boxes = [pending.tokens[i].bbox for i in to_redact]
    secrets = [pending.tokens[i].text for i in to_redact]

    src = io.BytesIO(pending.image)
    strength, out_bytes, leaked = 1, pending.image, []
    while strength <= 4:
        src.seek(0)
        out_bytes = redact.apply_bytes(src.read(), boxes, strength=strength)
        result = verify.check_bytes(out_bytes, secrets)
        leaked = result.still_readable
        if result.ok:
            break
        strength += 1

    memory = _memory_for(user_id)
    report = reflect_on_run(memory, f"web_{int(time.time())}", pending.context, decisions)

    recorder = getattr(memory.store, "record_run", None)
    if recorder is not None:
        try:
            from datetime import datetime, timezone
            recorder({
                "run_id": f"web_{int(time.time())}",
                "document": "web-upload",
                "context": pending.context,
                "asked": sum(1 for d in decisions if d.asked_user),
                "from_memory": sum(1 for d in decisions if d.source == "memory"),
                "model_calls": 0,
                "tokens_out": 0,
                "latency_ms": 0,
                "verify_attempts": strength,
                "final_strength": strength,
                "learned": report.summary(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            pass

    _pending.pop(token_id, None)     # the image leaves memory immediately

    result_id = uuid.uuid4().hex
    _results[result_id] = (out_bytes, time.time())

    return JSONResponse({
        "image_url": f"/api/result/{result_id}",
        "bytes": len(out_bytes),
        "strength": strength,
        "verified": not leaked,
        "still_readable": leaked,
        "learned": report.summary(),
        "rules": len(memory.all_rules()),
        "degraded": bool(getattr(memory.store, "degraded", False)),
    })


@app.get("/api/result/{result_id}")
async def result(result_id: str) -> Response:
    """Serve one finished redaction as binary, once."""
    item = _results.get(result_id)
    if item is None:
        raise HTTPException(410, "result expired")
    return Response(
        content=item[0],
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="safe.png"',
                 "Cache-Control": "no-store"},
    )


@app.get("/api/policy")
async def policy(user_id: str) -> JSONResponse:
    """What this user's agent has learned so far."""
    memory = _memory_for(user_id)
    rules = sorted(memory.all_rules(), key=lambda r: (-r.confidence, r.field_name))
    return JSONResponse({"rules": [{
        "field": r.field_name,
        "decision": r.decision,
        "confidence": round(r.confidence, 2),
        "auto": r.auto,
        "seen": len(r.support),
        "context": r.context,
    } for r in rules]})


@app.on_event("startup")
async def _warm() -> None:
    """Load the OCR models before serving. Cheaper than doing it inside a
    request that is also holding an image and its intermediate buffers."""
    try:
        ocr._get_engine()
    except Exception:
        pass


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
