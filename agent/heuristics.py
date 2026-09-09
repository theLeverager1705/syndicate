"""Naming fields without a model.

The model is the best namer we have, but it is a remote dependency on someone
else's free tier, and when it goes away every field degrades to "unresolved_3",
which is technically honest and looks broken.

Most of the documents this tool sees are laid out label-then-value, and the
labels repeat. That is enough to name the common fields deterministically, for
free, in microseconds. The model then only handles what these rules genuinely
cannot -- and once memory has learned a field, neither is consulted.

These rules never decide alone. Every proposal they produce is marked
needs_user, because a pattern match is a guess about what a value *is*, not
evidence about what this person wants done with it.
"""
from __future__ import annotations

import re

# Label text (normalised to lowercase alphanumerics) -> canonical field name.
# Ordered longest-first at match time so "registrationno" wins over "no".
LABEL_FIELDS: dict[str, str] = {
    "candidatename": "candidate_name",
    "studentname": "candidate_name",
    "name": "candidate_name",
    "registrationno": "registration_number",
    "registrationnumber": "registration_number",
    "regno": "registration_number",
    "applicationno": "application_number",
    "applicationnumber": "application_number",
    "rollnumber": "roll_number",
    "rollno": "roll_number",
    "dateofbirth": "date_of_birth",
    "dob": "date_of_birth",
    "allindiarank": "all_india_rank",
    "air": "all_india_rank",
    "rank": "all_india_rank",
    "score": "score",
    "marks": "score",
    "percentile": "score",
    "totalmarks": "score",
    "employeeid": "employee_id",
    "empid": "employee_id",
    "employeecode": "employee_id",
    "salary": "salary",
    "ctc": "salary",
    "annualsalary": "salary",
    "jobtitle": "job_title",
    "designation": "job_title",
    "position": "job_title",
    "startdate": "start_date",
    "joiningdate": "start_date",
    "dateofjoining": "start_date",
    "employer": "employer_name",
    "company": "employer_name",
    "organisation": "employer_name",
    "organization": "employer_name",
}

# Value shapes, tried only when no label identified the field.
VALUE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Indian vehicle registration: MH12AB1234, KA-05-CJ-8890
    ("license_plate", re.compile(r"^[A-Z]{2}[\s-]?\d{1,2}[\s-]?[A-Z]{1,3}[\s-]?\d{4}$", re.I)),
    # Dates: 14-08-2002, 14/08/2002, 2002-08-14
    ("date_of_birth", re.compile(r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$|^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$")),
    # Mixed alphanumeric identifier of meaningful length, e.g. GA24107733
    ("identifier", re.compile(r"^(?=.*[A-Z])(?=.*\d)[A-Z0-9]{6,20}$", re.I)),
]

# Default handling per field when a person has expressed no preference yet.
# Deliberately cautious: identifiers are proposed for hiding, achievements are
# proposed for keeping. Both are still put to the user.
DEFAULT_DECISION: dict[str, str] = {
    "candidate_name": "keep",
    "score": "keep",
    "all_india_rank": "keep",
    "job_title": "keep",
    "start_date": "keep",
    "employer_name": "keep",
    "registration_number": "redact",
    "application_number": "redact",
    "roll_number": "redact",
    "date_of_birth": "redact",
    "employee_id": "redact",
    "salary": "redact",
    "license_plate": "redact",
    "identifier": "redact",
}


def _norm(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum()).casefold()


def field_from_label(label: str | None) -> str | None:
    """Canonical field name for a layout label, if we recognise it."""
    if not label:
        return None
    norm = _norm(label)
    if not norm:
        return None
    if norm in LABEL_FIELDS:
        return LABEL_FIELDS[norm]
    # Longest key first so "registrationno" beats a bare "no".
    for key in sorted(LABEL_FIELDS, key=len, reverse=True):
        if key in norm:
            return LABEL_FIELDS[key]
    return None


def field_from_value(value: str) -> str | None:
    """Field name inferred from the shape of the value alone."""
    stripped = value.strip()
    if not stripped:
        return None
    for name, pattern in VALUE_PATTERNS:
        if pattern.match(stripped):
            return name
    return None


def identify(value: str, label: str | None) -> tuple[str, str] | None:
    """(field_name, suggested_decision), or None when nothing is recognised.

    The label is trusted over the value's shape: a number under "All India
    Rank" is a rank, even though it looks like any other number.
    """
    name = field_from_label(label) or field_from_value(value)
    if name is None:
        return None
    return name, DEFAULT_DECISION.get(name, "redact")
