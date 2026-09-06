"""Documents and a simulated user for the evaluation harness.

The point of these fixtures is that a *generic* PII policy gets this
particular user wrong. If the naive baseline already matched the user, the
learning curve would be measuring nothing.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Doc:
    doc_id: str
    context: dict[str, str]
    fields: list[str]


# 10 documents. Contexts repeat enough for rules to earn confidence, and vary
# enough that context-blind memory would be caught out.
DOCS: list[Doc] = [
    # Interleaved deliberately: each context recurs enough for rules to earn
    # confidence, and the Instagram offer letter arrives AFTER the LinkedIn
    # ones so we can show the agent declining to transfer a rule across a
    # context it has not verified.
    Doc("gate_2024", {"doc_type": "exam_scorecard", "platform": "linkedin", "issuer": "GATE"},
        ["candidate_name", "registration_number", "roll_number", "all_india_rank", "score", "qr_code", "dob"]),
    Doc("jee_main", {"doc_type": "exam_scorecard", "platform": "linkedin", "issuer": "JEE"},
        ["candidate_name", "application_number", "roll_number", "all_india_rank", "score", "dob"]),
    Doc("car_front", {"doc_type": "vehicle_photo", "platform": "instagram", "issuer": ""},
        ["license_plate", "face", "location_landmark"]),
    Doc("upsc_pre", {"doc_type": "exam_scorecard", "platform": "linkedin", "issuer": "UPSC"},
        ["candidate_name", "registration_number", "roll_number", "all_india_rank", "score", "qr_code"]),
    Doc("offer_acme", {"doc_type": "offer_letter", "platform": "linkedin", "issuer": "ACME"},
        ["candidate_name", "employee_id", "job_title", "salary", "start_date", "employer_name"]),
    Doc("bike_new", {"doc_type": "vehicle_photo", "platform": "instagram", "issuer": ""},
        ["license_plate", "face", "location_landmark"]),
    Doc("ssc_cgl", {"doc_type": "exam_scorecard", "platform": "linkedin", "issuer": "SSC"},
        ["candidate_name", "registration_number", "roll_number", "all_india_rank", "score", "dob"]),
    Doc("offer_initech", {"doc_type": "offer_letter", "platform": "linkedin", "issuer": "INITECH"},
        ["candidate_name", "employee_id", "job_title", "salary", "start_date", "employer_name"]),
    Doc("car_side", {"doc_type": "vehicle_photo", "platform": "instagram", "issuer": ""},
        ["license_plate", "face", "location_landmark"]),
    Doc("offer_globex", {"doc_type": "offer_letter", "platform": "instagram", "issuer": "GLOBEX"},
        ["candidate_name", "employee_id", "job_title", "salary", "employer_name"]),
    Doc("gate_2025", {"doc_type": "exam_scorecard", "platform": "linkedin", "issuer": "GATE"},
        ["candidate_name", "registration_number", "roll_number", "all_india_rank", "score", "qr_code", "dob"]),
    Doc("cat_score", {"doc_type": "exam_scorecard", "platform": "linkedin", "issuer": "CAT"},
        ["candidate_name", "registration_number", "roll_number", "all_india_rank", "score"]),
]


# The user's actual posture. The agent never sees this -- it must be inferred
# from corrections. Note the context-dependent entries: this user is happy to
# name their employer on LinkedIn and not on Instagram, and is proud of their
# rank but not their raw score.
USER_POLICY: dict[tuple[str, str], str] = {
    # (field, platform) -> decision.  platform "" = applies anywhere.
    ("candidate_name", ""): "keep",
    ("registration_number", ""): "redact",
    ("application_number", ""): "redact",
    ("roll_number", ""): "redact",
    ("all_india_rank", ""): "keep",
    ("score", ""): "keep",
    ("qr_code", ""): "redact",
    ("dob", ""): "redact",
    ("employee_id", ""): "redact",
    ("job_title", ""): "keep",
    ("salary", ""): "redact",
    ("start_date", ""): "keep",
    ("employer_name", "linkedin"): "keep",
    ("employer_name", "instagram"): "redact",
    ("license_plate", ""): "redact",
    ("face", ""): "keep",
    ("location_landmark", "instagram"): "redact",
}


def user_decision(field: str, context: dict[str, str]) -> str:
    """What the simulated user actually wants. Ground truth."""
    platform = context.get("platform", "")
    if (field, platform) in USER_POLICY:
        return USER_POLICY[(field, platform)]
    return USER_POLICY.get((field, ""), "keep")


# A generic PII detector's prior: redact anything that smells like an
# identifier. This is what a non-learning product ships, and it is wrong about
# this user on score, all_india_rank, candidate_name, face, and employer_name.
NAIVE_REDACT = {
    "candidate_name", "registration_number", "application_number", "roll_number",
    "qr_code", "dob", "employee_id", "salary", "license_plate", "face",
    "employer_name", "location_landmark", "score",
}


def naive_proposal(field: str, context: dict[str, str]) -> str:
    """Stand-in for the model's fresh judgement, context-blind by design."""
    return "redact" if field in NAIVE_REDACT else "keep"
