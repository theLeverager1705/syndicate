"""Removing information from an image, with escalation.

Blur is a spectrum, not a guarantee. Low-strength blur on large high-contrast
digits is frequently still readable by OCR -- and if OCR can read it, so can a
motivated human. So strength escalates, and past a threshold we stop blurring
and start deleting.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

Box = tuple[int, int, int, int]

# Measured, not guessed. Sweeping blur radius against OCR on our own fixtures:
# document text became unreadable at radius 4, but a large high-contrast
# licence plate survived radius 4 and needed 6. There is no single correct
# strength -- which is exactly why the pipeline verifies and escalates rather
# than blurring once and trusting it.
#
# Level 1 starts deliberately gentle: over-destroying an image the user wanted
# to share is its own kind of failure.
BASE_RADIUS = 3.0

# Beyond this strength, blur is abandoned for an opaque fill. A blurred region
# retains low-frequency structure that can sometimes be inverted; a filled one
# cannot.
OPAQUE_AT = 4


def _params(strength: int) -> tuple[float, int]:
    """(blur radius, padding) for an escalation level."""
    radius = BASE_RADIUS * (1.8 ** (strength - 1))
    padding = 4 + 6 * (strength - 1)
    return radius, padding


def _expand(box: Box, padding: int, size: tuple[int, int]) -> Box:
    x0, y0, x1, y1 = box
    w, h = size
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(w, x1 + padding),
        min(h, y1 + padding),
    )


def apply(
    image_path: str,
    boxes: list[Box],
    out_path: str,
    strength: int = 1,
) -> str:
    """Redact `boxes` from `image_path`, writing to `out_path`.

    The input file is never modified. Returns out_path.
    """
    radius, padding = _params(strength)
    with Image.open(image_path) as src:
        img = src.convert("RGB")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if not boxes:
        img.save(out_path)
        return out_path

    draw = ImageDraw.Draw(img)
    for box in boxes:
        region_box = _expand(box, padding, img.size)
        if region_box[2] <= region_box[0] or region_box[3] <= region_box[1]:
            continue

        if strength >= OPAQUE_AT:
            draw.rectangle(region_box, fill=(17, 17, 17))
        else:
            region = img.crop(region_box).filter(ImageFilter.GaussianBlur(radius))
            img.paste(region, region_box[:2])

    img.save(out_path)
    return out_path
