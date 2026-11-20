"""Unit tests for deterministic validation checks."""

from datetime import date, timedelta

from invoice_extractor import checks
from invoice_extractor.schema import DocumentType, Invoice, IssueCode, LineItem


def test_totals_pass_inside_tolerance_and_fail_outside() -> None:
    """Validate totals check succeeds within tolerance and flags outside."""
    inv = Invoice(
        document_type=DocumentType.INVOICE,
        vendor="Acme",
        invoice_number="INV-1",
        invoice_date="2026-01-01",
        currency="USD",
        line_items=[
            LineItem(description="Item 1", amount=100.0),
            LineItem(description="Item 2", amount=50.0),
        ],
        tax_total=10.0,
        total=160.5,  # diff = 0.5 <= tolerance (1.0)
    )
    issues = checks.run_checks(inv)
    assert not any(i.code == IssueCode.TOTALS_MISMATCH for i in issues)

    # Outside tolerance (diff = 5.0)
    inv.total = 165.0
    issues = checks.run_checks(inv)
    mismatch = [i for i in issues if i.code == IssueCode.TOTALS_MISMATCH]
    assert len(mismatch) == 1
    assert "Sum of items (150.00) + tax (10.00) != total (165.00)" in mismatch[0].detail


def test_non_invoice_with_fields_is_flagged() -> None:
    """Assert non-invoices with populated invoice fields are flagged."""
    inv = Invoice(
        document_type=DocumentType.BANK_STATEMENT,
        total=500.0,
    )
    issues = checks.run_checks(inv)
    assert any(i.code == IssueCode.NON_INVOICE_HAS_FIELDS for i in issues)

    # Clean non-invoice has no issues
    clean_inv = Invoice(document_type=DocumentType.BANK_STATEMENT)
    clean_issues = checks.run_checks(clean_inv)
    assert len(clean_issues) == 0


def test_borderline_document_types_are_flagged_ambiguous() -> None:
    """Verify borderline types trigger AMBIGUOUS_TYPE review."""
    for b_type in [DocumentType.CREDIT_NOTE, DocumentType.RECEIPT, DocumentType.QUOTATION]:
        inv = Invoice(document_type=b_type)
        issues = checks.run_checks(inv)
        assert any(i.code == IssueCode.AMBIGUOUS_TYPE for i in issues)


def test_invalid_date_and_currency() -> None:
    """Validate invalid date formats, out of bounds dates, and malformed currency."""
    # Malformed date
    inv = Invoice(
        document_type=DocumentType.INVOICE,
        vendor="A",
        invoice_number="1",
        invoice_date="2026/05/10",
        currency="USD",
        total=10.0,
        line_items=[LineItem(description="x", amount=10.0)],
    )
    issues = checks.run_checks(inv)
    assert any(i.code == IssueCode.INVALID_DATE for i in issues)

    # Future date beyond tomorrow
    future = (date.today() + timedelta(days=10)).isoformat()
    inv.invoice_date = future
    issues = checks.run_checks(inv)
    assert any(i.code == IssueCode.INVALID_DATE for i in issues)

    # Invalid currency
    inv.invoice_date = "2026-01-01"
    inv.currency = "us"
    issues = checks.run_checks(inv)
    assert any(i.code == IssueCode.INVALID_CURRENCY for i in issues)

    inv.currency = "USDT"
    issues = checks.run_checks(inv)
    assert any(i.code == IssueCode.INVALID_CURRENCY for i in issues)


def test_grounding_passes_and_fails() -> None:
    """Check text grounding of invoice number and total amount against source."""
    source_text = "Tax Invoice SGS/1847 Dated 2026-08-19 Total Amount: 23,139.80 INR"
    inv = Invoice(
        document_type=DocumentType.INVOICE,
        vendor="SGS",
        invoice_number="SGS/1847",
        invoice_date="2026-08-19",
        currency="INR",
        total=23139.8,
        line_items=[LineItem(description="Item", amount=23139.8)],
    )
    # Passes grounding
    issues = checks.run_checks(inv, source_text=source_text)
    assert not any(i.code == IssueCode.NOT_IN_SOURCE_TEXT for i in issues)

    # Hallucinated invoice number
    inv.invoice_number = "SGS/9999"
    issues = checks.run_checks(inv, source_text=source_text)
    assert any(i.code == IssueCode.NOT_IN_SOURCE_TEXT and "SGS/9999" in i.detail for i in issues)

    # Hallucinated total
    inv.invoice_number = "SGS/1847"
    inv.total = 99999.0
    inv.line_items = [LineItem(description="Item", amount=99999.0)]
    issues = checks.run_checks(inv, source_text=source_text)
    assert any(i.code == IssueCode.NOT_IN_SOURCE_TEXT and "99999.00" in i.detail for i in issues)
