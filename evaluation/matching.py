"""Field comparison and matching rules for evaluation."""

import re

from rapidfuzz import fuzz

from invoice_extractor.schema import LineItem

_LEGAL_SUFFIXES = (
    "pvt ltd",
    "private limited",
    "pvt",
    "private",
    "ltd",
    "limited",
    "llp",
    "inc",
    "co",
    "company",
)


def _clean_vendor(name: str) -> str:
    """Lowercase, strip punctuation, and remove trailing corporate legal suffixes."""
    cleaned = re.sub(r"[^\w\s]", " ", name.lower())
    tokens = cleaned.split()
    joined = " ".join(tokens)
    for suffix in _LEGAL_SUFFIXES:
        if joined.endswith(suffix):
            joined = joined[: -len(suffix)].strip()
            break
    return joined


def match_vendor(predicted: str | None, expected: str | None) -> bool:
    """Match vendor names allowing minor variation and legal entity suffix differences."""
    if predicted is None or expected is None:
        return predicted is expected
    c_pred = _clean_vendor(predicted)
    c_exp = _clean_vendor(expected)
    if c_pred == c_exp:
        return True
    return fuzz.ratio(c_pred, c_exp) >= 85


def match_invoice_number(predicted: str | None, expected: str | None) -> bool:
    """Match invoice numbers ignoring case, whitespace, and leading/trailing spaces."""
    if predicted is None or expected is None:
        return predicted is expected
    clean_p = predicted.strip().lower().replace(" ", "")
    clean_e = expected.strip().lower().replace(" ", "")
    return clean_p == clean_e


def _parse_float(val: float | str | None) -> float | None:
    """Convert numerical values or currency formatted strings to float."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    cleaned = re.sub(r"[^\d.-]", "", str(val))
    try:
        return float(cleaned)
    except ValueError:
        return None


def match_number(
    predicted: float | str | None,
    expected: float | str | None,
    tolerance: float = 0.01,
) -> bool:
    """Compare numeric fields within an absolute numerical tolerance."""
    p_num = _parse_float(predicted)
    e_num = _parse_float(expected)
    if p_num is None or e_num is None:
        return p_num is e_num
    return abs(p_num - e_num) <= tolerance


def match_line_items(
    predicted: list[LineItem] | list[dict] | None,
    expected: list[LineItem] | list[dict] | None,
) -> float:
    """Greedy 1-to-1 matching of line items on amount and quantity, returning F1 score."""
    p_items = _normalize_line_items(predicted)
    e_items = _normalize_line_items(expected)

    if not p_items and not e_items:
        return 1.0
    if not p_items or not e_items:
        return 0.0

    matched_e = set()
    true_positives = 0

    for p in p_items:
        for idx, e in enumerate(e_items):
            if idx in matched_e:
                continue
            amt_match = abs(p["amount"] - e["amount"]) <= 0.01
            qty_match = _match_quantity(p.get("quantity"), e.get("quantity"))
            if amt_match and qty_match:
                matched_e.add(idx)
                true_positives += 1
                break

    precision = true_positives / len(p_items) if p_items else 0.0
    recall = true_positives / len(e_items) if e_items else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def _match_quantity(q1: float | None, q2: float | None) -> bool:
    """Match quantities treating None and None as equal."""
    if q1 is None or q2 is None:
        return q1 is q2
    return abs(float(q1) - float(q2)) <= 0.001


def _normalize_line_items(items: list[LineItem] | list[dict] | None) -> list[dict]:
    """Convert line item objects or dicts into standard dictionary representations."""
    if not items:
        return []
    res = []
    for it in items:
        if isinstance(it, dict):
            res.append({"amount": float(it.get("amount", 0.0)), "quantity": it.get("quantity")})
        else:
            res.append({"amount": float(it.amount), "quantity": it.quantity})
    return res


def is_non_invoice_correct(data: dict) -> bool:
    """Assert document is identified as non-invoice and all extraction fields are null."""
    if data.get("is_invoice") is not False:
        return False
    fields = [
        "vendor",
        "invoice_number",
        "invoice_date",
        "currency",
        "line_items",
        "tax_total",
        "total",
    ]
    return all(data.get(f) is None for f in fields)
