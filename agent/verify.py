"""Checking our own work, without asking a model whether it did well.

After redacting, we re-run OCR on the OUTPUT image and assert the target
strings are no longer recoverable. This is the difference between "we blurred
a region" and "the information is gone" -- and it is deliberately not an LLM
judgement, because a model asked to grade its own output is not evidence.

Partial survival counts as failure. A licence plate leaking as MH12AB12
instead of MH12AB1234 is still a licence plate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import ocr

# A surviving fragment this long (or this fraction of the original) is treated
# as a leak. Four characters is about where an identifier fragment stops being
# guessable noise and starts being a lookup key.
MIN_FRAGMENT = 4
LEAK_FRACTION = 0.6


@dataclass
class VerifyResult:
    ok: bool
    still_readable: list[str] = field(default_factory=list)
    attempts: int = 1


def normalise(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum()).casefold()


def _longest_common_run(a: str, b: str) -> int:
    """Length of the longest contiguous substring shared by a and b."""
    if not a or not b:
        return 0
    best = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def leaked(target: str, observed_tokens: list[str]) -> bool:
    """Is `target` still recoverable from what OCR read back?"""
    t = normalise(target)
    if not t:
        return False

    threshold = max(MIN_FRAGMENT, int(len(t) * LEAK_FRACTION))

    for raw in observed_tokens:
        o = normalise(raw)
        if not o:
            continue
        if t in o:
            return True
        if _longest_common_run(t, o) >= threshold:
            return True
    return False


def check(out_path: str, must_be_absent: list[str], attempts: int = 1) -> VerifyResult:
    """Re-read the redacted image and report which targets survived."""
    tokens = [t.text for t in ocr.extract(out_path, use_cache=False)]
    survivors = [s for s in must_be_absent if leaked(s, tokens)]
    return VerifyResult(ok=not survivors, still_readable=survivors, attempts=attempts)


def check_bytes(image_bytes: bytes, must_be_absent: list[str]) -> VerifyResult:
    """Deterministic verification against in-memory PNG bytes."""
    import io

    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as im:
        array = np.array(im.convert("RGB"))
    tokens = [t.text for t in ocr.extract_array(array)]
    survivors = [s for s in must_be_absent if leaked(s, tokens)]
    return VerifyResult(ok=not survivors, still_readable=survivors)
