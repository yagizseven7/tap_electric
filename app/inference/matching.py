"""
Turning text (from a QR code or from the model) into a charger.

Charger IDs follow the EVSE ID format used across Europe (eMI3 / ISO 15118):

    NL*TNM*E12345*1
    ^^ ^^^ ^^^^^^^^
    |  |   "E" + the charge point, often with a connector number
    |  operator ID (3 letters/digits)
    country code

The asterisks are optional (NLTNME123451 is the same ID), so we compare
IDs with the asterisks removed.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from urllib.parse import unquote

from app.storage.chargers import Charger

EVSE_PATTERN = re.compile(r"([A-Z]{2})\*?([A-Z0-9]{3})\*?(E[A-Z0-9*]{1,30})")


def normalize(text: str) -> str:
    """Upper-case, URL-decode (%2A -> *), keep only letters, digits and '*'."""
    return re.sub(r"[^A-Z0-9*]", "", unquote(text).upper())


def comparison_key(evse_id: str) -> str:
    return normalize(evse_id).replace("*", "")


def extract_evse_id(text: str) -> str | None:
    """Find an EVSE ID inside any text: a bare ID, a URL from a QR code
    ('https://example.com/start/NL*TNM*E12345*1'), or OCR output with extra
    words around it. Returns it in the standard form with asterisks.

    We first look at each piece between separators (/ : . space ...) on its
    own; otherwise 'https://example.com/...' would be glued into one string
    and 'HTTPS...' would look like the start of an ID. Only if no single
    piece matches do we glue everything together, for OCR output where
    the model put spaces inside the ID ('NL TNM E12345 1').
    """
    upper = unquote(text).upper()
    pieces = [p for p in re.split(r"[^A-Z0-9*]+", upper) if p]

    candidates = (
        [EVSE_PATTERN.fullmatch(p) for p in reversed(pieces)]   # a piece that IS an ID (IDs sit at the end of URLs)
        + [EVSE_PATTERN.search(p) for p in reversed(pieces)]    # a piece that CONTAINS an ID
        + [EVSE_PATTERN.search(normalize(text))]                # last resort: everything glued together
    )
    match = next((m for m in candidates if m is not None), None)
    if match is None:
        return None
    country, operator, rest = match.groups()
    return f"{country}*{operator}*{rest.strip('*')}"


def similarity(a: str, b: str) -> float:
    """0.0 (nothing in common) to 1.0 (identical), ignoring asterisks.

    'NLTNME12345' vs 'NLTNME12346' -> about 0.91: one wrong character
    still gives a high score, which is exactly what makes a slightly
    misread ID useful.
    """
    return SequenceMatcher(None, comparison_key(a), comparison_key(b)).ratio()


@dataclass(frozen=True)
class MatchCandidate:
    charger: Charger
    score: float  # similarity between what we read and this charger's ID


def rank_chargers(read_text: str, chargers: list[Charger]) -> list[MatchCandidate]:
    """Score every nearby charger against the text we read, best first."""
    query = extract_evse_id(read_text) or normalize(read_text)
    if not query:
        return []
    ranked = [MatchCandidate(c, similarity(query, c.evse_id)) for c in chargers]
    return sorted(ranked, key=lambda m: m.score, reverse=True)
