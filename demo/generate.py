"""Synthetic demo documents. No real personal data goes in the video.

Layouts, palettes and fonts vary between documents on purpose: if every input
were the same template, the learning result would be an artefact of the
template rather than evidence the agent generalises.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "images"

FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/verdana.ttf",
]


def font(size: int, idx: int = 0) -> ImageFont.FreeTypeFont:
    order = FONT_CANDIDATES[idx % len(FONT_CANDIDATES):] + FONT_CANDIDATES
    for path in order:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def scorecard(doc_id, issuer, rows, style):
    w, h = 900, 640
    bg = [(255, 255, 255), (250, 250, 245), (248, 250, 252)][style % 3]
    accent = [(10, 60, 120), (120, 20, 40), (20, 90, 70)][style % 3]

    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)

    d.rectangle([0, 0, w, 78], fill=accent)
    d.text((32, 22), issuer + " - SCORECARD", font=font(30, style), fill=(255, 255, 255))

    y = 120
    label_f, value_f = font(20, style), font(26, style)
    for label, value in rows:
        d.text((60, y), label, font=label_f, fill=(90, 90, 90))
        d.text((400, y - 4), value, font=value_f, fill=(15, 15, 15))
        y += 54

    # A QR-like block. OCR will not read it, but it is a real identifier carrier
    # and the agent has to reason about it without being able to decode it.
    qx, qy = w - 175, h - 195
    d.rectangle([qx, qy, qx + 125, qy + 125], outline=(0, 0, 0), width=2)
    rnd = random.Random(doc_id)
    for i in range(10):
        for j in range(10):
            if rnd.random() > 0.5:
                d.rectangle(
                    [qx + 8 + i * 11, qy + 8 + j * 11, qx + 17 + i * 11, qy + 17 + j * 11],
                    fill=(0, 0, 0),
                )

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (doc_id + ".png")
    img.save(path)
    return path


def vehicle(doc_id, plate, style):
    w, h = 900, 600
    img = Image.new("RGB", (w, h), (152, 190, 220))
    d = ImageDraw.Draw(img)

    d.rectangle([0, 380, w, h], fill=(110, 115, 120))
    body = [(180, 40, 50), (30, 60, 140), (235, 235, 235)][style % 3]
    d.rounded_rectangle([150, 210, 760, 430], radius=40, fill=body)
    d.rounded_rectangle([250, 235, 640, 330], radius=22, fill=(60, 80, 100))
    d.ellipse([210, 390, 320, 500], fill=(25, 25, 25))
    d.ellipse([600, 390, 710, 500], fill=(25, 25, 25))

    px, py = 375, 442
    d.rectangle([px, py, px + 250, py + 64], fill=(255, 255, 255), outline=(0, 0, 0), width=3)
    d.text((px + 14, py + 13), plate, font=font(38, style), fill=(0, 0, 0))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (doc_id + ".png")
    img.save(path)
    return path


SCORECARDS = [
    ("gate_2024", "GATE 2024", [
        ("Candidate Name", "A. Sharma"), ("Registration No", "GA24107733"),
        ("Roll Number", "CS24B0417"), ("Date of Birth", "14-08-2002"),
        ("Score", "742"), ("All India Rank", "184"),
    ], ["GA24107733", "CS24B0417", "14-08-2002"]),
    ("jee_main", "JEE MAIN", [
        ("Candidate Name", "A. Sharma"), ("Application No", "JM250884219"),
        ("Roll Number", "JM25R7781"), ("Date of Birth", "14-08-2002"),
        ("Percentile", "99.21"), ("All India Rank", "1042"),
    ], ["JM250884219", "JM25R7781", "14-08-2002"]),
    ("upsc_pre", "UPSC PRELIMS", [
        ("Candidate Name", "A. Sharma"), ("Registration No", "UP2400913"),
        ("Roll Number", "UP24C0221"), ("Score", "118.6"),
        ("All India Rank", "612"),
    ], ["UP2400913", "UP24C0221"]),
    ("ssc_cgl", "SSC CGL", [
        ("Candidate Name", "A. Sharma"), ("Registration No", "SS24771204"),
        ("Roll Number", "SS24R3390"), ("Date of Birth", "14-08-2002"),
        ("Score", "331.5"), ("All India Rank", "2208"),
    ], ["SS24771204", "SS24R3390", "14-08-2002"]),
    ("gate_2025", "GATE 2025", [
        ("Candidate Name", "A. Sharma"), ("Registration No", "GA25330918"),
        ("Roll Number", "CS25B1120"), ("Date of Birth", "14-08-2002"),
        ("Score", "801"), ("All India Rank", "97"),
    ], ["GA25330918", "CS25B1120", "14-08-2002"]),
    ("cat_score", "CAT", [
        ("Candidate Name", "A. Sharma"), ("Registration No", "CT24559041"),
        ("Roll Number", "CT24R8802"), ("Percentile", "98.44"),
        ("All India Rank", "1533"),
    ], ["CT24559041", "CT24R8802"]),
]

VEHICLES = [
    ("car_front", "MH12AB1234"),
    ("bike_new", "KA05CJ8890"),
    ("car_side", "TN09XY4417"),
]


def build() -> dict:
    manifest: dict[str, dict] = {}

    for i, (doc_id, issuer, rows, secrets) in enumerate(SCORECARDS):
        path = scorecard(doc_id, issuer, rows, i)
        manifest[doc_id] = {
            "path": str(path),
            "must_redact": secrets,
            "context": {
                "doc_type": "exam_scorecard",
                "platform": "linkedin",
                "issuer": issuer.split()[0],
            },
        }

    for i, (doc_id, plate) in enumerate(VEHICLES):
        path = vehicle(doc_id, plate, i)
        manifest[doc_id] = {
            "path": str(path),
            "must_redact": [plate],
            "context": {
                "doc_type": "vehicle_photo",
                "platform": "instagram",
                "issuer": "",
            },
        }

    out = Path(__file__).resolve().parent / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    m = build()
    print("generated " + str(len(m)) + " images in " + str(OUT))
    for k, v in m.items():
        print("  " + k.ljust(12) + " must redact: " + ", ".join(v["must_redact"]))
