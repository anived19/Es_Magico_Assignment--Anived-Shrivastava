"""Pure validation functions evaluating invoices against business rules."""

import re
from datetime import date, timedelta

from rapidfuzz import fuzz

from invoice_extractor.config import OCR_FUZZY_RATIO, TOTALS_TOLERANCE
from invoice_extractor.schema import DocumentType, Invoice, Issue, IssueCode

AMBIGUOUS_DOCUMENT_TYPES: set[DocumentType] = {
    DocumentType.CREDIT_NOTE,
    DocumentType.RECEIPT,
    DocumentType.QUOTATION,
}


def _check_non_invoice_fields(invoice: Invoice) -> Issue | None:
    """Flag non-invoice documents that have any invoice-specific fields populated."""
    if invoice.is_invoice is not False:
        return None
    fields = [
        invoice.vendor,
        invoice.invoice_number,
        invoice.invoice_date,
        invoice.currency,
        invoice.line_items,
        invoice.tax_total,
        invoice.total,
    ]
    if any(f is not None for f in fields):
        return Issue(
            IssueCode.NON_INVOICE_HAS_FIELDS,
            "Document marked non-invoice but contains populated invoice fields",
        )
    return None


def _check_missing_fields(invoice: Invoice) -> list[Issue]:
    """Ensure required metadata and line item entries are present on invoices."""
    if invoice.document_type is None:
        return [Issue(IssueCode.MISSING_FIELD, "document_type is missing or null")]
    if invoice.is_invoice is not True:
        return []

    issues: list[Issue] = []
    required = {
        "vendor": invoice.vendor,
        "invoice_number": invoice.invoice_number,
        "invoice_date": invoice.invoice_date,
        "currency": invoice.currency,
        "total": invoice.total,
    }
    for field_name, val in required.items():
        if val is None:
            issues.append(Issue(IssueCode.MISSING_FIELD, f"Required field '{field_name}' is null"))

    if not invoice.line_items:
        issues.append(Issue(IssueCode.MISSING_FIELD, "line_items is empty or null"))
    return issues


def _check_ambiguous_type(invoice: Invoice) -> Issue | None:
    """Send borderline financial document types to manual human review."""
    if invoice.document_type in AMBIGUOUS_DOCUMENT_TYPES:
        return Issue(
            IssueCode.AMBIGUOUS_TYPE,
            f"Borderline document type '{invoice.document_type.value}' requires review",
        )
    return None


def _check_date(invoice: Invoice, today: date | None = None) -> Issue | None:
    """Validate that the invoice date is a realistic ISO date within allowed range."""
    if invoice.is_invoice is not True or not invoice.invoice_date:
        return None
    try:
        parsed = date.fromisoformat(invoice.invoice_date)
    except ValueError:
        return Issue(IssueCode.INVALID_DATE, f"Date '{invoice.invoice_date}' is not ISO YYYY-MM-DD")

    ref_date = today or date.today()
    max_date = ref_date + timedelta(days=1)
    if not (date(2000, 1, 1) <= parsed <= max_date):
        return Issue(
            IssueCode.INVALID_DATE,
            f"Date '{invoice.invoice_date}' outside plausible range (2000-01-01 to {max_date})",
        )
    return None


def _check_currency(invoice: Invoice) -> Issue | None:
    """Validate currency is a 3-letter uppercase alphabetic ISO code."""
    if invoice.is_invoice is not True or not invoice.currency:
        return None
    if (
        len(invoice.currency) != 3
        or not invoice.currency.isalpha()
        or not invoice.currency.isupper()
    ):
        return Issue(
            IssueCode.INVALID_CURRENCY,
            f"Currency '{invoice.currency}' is not a 3-letter uppercase code",
        )
    return None


def _check_totals(invoice: Invoice) -> Issue | None:
    """Validate that line item amounts plus tax equal total within tolerance."""
    if invoice.is_invoice is not True or invoice.total is None or invoice.line_items is None:
        return None
    items_sum = sum(item.amount for item in invoice.line_items)
    tax = invoice.tax_total or 0.0
    expected = items_sum + tax
    diff = abs(expected - invoice.total)
    if diff > TOTALS_TOLERANCE:
        return Issue(
            IssueCode.TOTALS_MISMATCH,
            f"Sum of items ({items_sum:.2f}) + tax ({tax:.2f}) != total ({invoice.total:.2f}), "
            f"difference={diff:.2f} > tolerance={TOTALS_TOLERANCE}",
        )
    return None


def _matches_amount(token: str, expected: float) -> bool:
    """Check if a numeric token matches expected amount in Anglo/Indian or European format."""
    try:
        val1 = float(token.replace(",", ""))
        if abs(val1 - expected) <= 0.01:
            return True
    except ValueError:
        pass

    try:
        val2 = float(token.replace(".", "").replace(",", "."))
        if abs(val2 - expected) <= 0.01:
            return True
    except ValueError:
        pass
    return False


def _check_grounding(
    invoice: Invoice,
    source_text: str | None,
    is_ocr: bool = False,
) -> list[Issue]:
    """Ground extracted invoice number and total directly against digital text layer or OCR."""
    if not source_text or invoice.is_invoice is not True:
        return []
    issues: list[Issue] = []

    if invoice.invoice_number:
        clean_inv = re.sub(r"[^\w]", "", invoice.invoice_number.lower())
        clean_src = re.sub(r"[^\w]", "", source_text.lower())
        if is_ocr:
            grounded = fuzz.partial_ratio(clean_inv, clean_src) >= OCR_FUZZY_RATIO
            if not grounded:
                issues.append(
                    Issue(
                        IssueCode.OCR_MISMATCH,
                        f"Invoice number '{invoice.invoice_number}' not grounded in OCR text",
                    )
                )
        else:
            grounded = clean_inv in clean_src
            if not grounded:
                issues.append(
                    Issue(
                        IssueCode.NOT_IN_SOURCE_TEXT,
                        f"Invoice number '{invoice.invoice_number}' not grounded in source text",
                    )
                )

    if invoice.total is not None:
        raw_tokens = re.findall(r"(?:\d[\d.,]*\d|\d+)", source_text)
        if not any(_matches_amount(tok, invoice.total) for tok in raw_tokens):
            code = IssueCode.OCR_MISMATCH if is_ocr else IssueCode.NOT_IN_SOURCE_TEXT
            desc = "OCR text" if is_ocr else "source text"
            issues.append(
                Issue(
                    code,
                    f"Invoice total {invoice.total:.2f} not grounded in {desc}",
                )
            )
    return issues


def _check_suspect_non_invoice(
    invoice: Invoice,
    source_text: str | None,
) -> list[Issue]:
    """Flag non-invoices if digital source text contains unambiguous invoice headers."""
    if not source_text or invoice.is_invoice is not False:
        return []
    has_inv_num = bool(re.search(r"(?i)\binvoice\s*(?:no\b|number\b|#)", source_text))
    has_tax_heading = bool(
        re.search(r"^\s*tax\s+invoice\s*$", source_text, re.MULTILINE | re.IGNORECASE)
    )
    if has_inv_num or has_tax_heading:
        return [
            Issue(
                IssueCode.SUSPECT_NON_INVOICE,
                "Document classified as non-invoice but contains clear invoice headers",
            )
        ]
    return []


def run_checks(
    invoice: Invoice,
    source_text: str | None = None,
    is_ocr: bool = False,
    today: date | None = None,
) -> list[Issue]:
    """Execute all pure deterministic validation checks against an extracted invoice."""
    single_checks = [
        _check_non_invoice_fields(invoice),
        _check_ambiguous_type(invoice),
        _check_date(invoice, today=today),
        _check_currency(invoice),
        _check_totals(invoice),
    ]
    issues = [iss for iss in single_checks if iss is not None]
    issues.extend(_check_missing_fields(invoice))
    issues.extend(_check_grounding(invoice, source_text, is_ocr=is_ocr))
    issues.extend(_check_suspect_non_invoice(invoice, source_text))
    return issues
