"""Unit tests for pipeline extraction attempts, retry logic, and selection rules."""

from unittest.mock import patch

from invoice_extractor.ingest import Document
from invoice_extractor.pipeline import process
from invoice_extractor.schema import (
    DocumentType,
    ExtractionFailed,
    UnavailableReason,
    Usage,
)

FAKE_PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"


def _fake_doc() -> Document:
    return Document(
        sha256="fake_sha",
        route="text_layer",
        text="Invoice No: INV-100\nTotal: 20.00\nDate: 2026-01-01\nUSD",
    )


def test_retry_successful_on_fixable_issue() -> None:
    """Test that a fixable issue triggers attempt 2 and selects the cleaner attempt."""
    call1_json = (
        '{"document_type": "INVOICE", "vendor": "Acme", "invoice_number": "INV-100", '
        '"invoice_date": "2026-01-01", "currency": "USD", "total": 20.0, '
        '"tax_total": 0.0, "line_items": [{"description": "Item", "amount": 10.0}]}'
    )
    call2_json = (
        '{"document_type": "INVOICE", "vendor": "Acme", "invoice_number": "INV-100", '
        '"invoice_date": "2026-01-01", "currency": "USD", "total": 20.0, '
        '"tax_total": 0.0, "line_items": [{"description": "Item", "amount": 20.0}]}'
    )

    with (
        patch("invoice_extractor.ingest.load", return_value=_fake_doc()),
        patch(
            "invoice_extractor.llm.extract",
            side_effect=[
                (call1_json, Usage(input_tokens=100, output_tokens=50, calls=1)),
                (call2_json, Usage(input_tokens=150, output_tokens=50, calls=1)),
            ],
        ),
    ):
        result = process(FAKE_PDF_BYTES)
        assert result.meta.attempts == 2
        assert result.needs_review is False
        assert result.total == 20.0


def test_retry_schema_failure_preserves_first_attempt() -> None:
    """Test that if attempt 2 fails schema validation, attempt 1 is kept despite more issues."""
    call1_json = (
        '{"document_type": "INVOICE", "vendor": "Acme", "invoice_number": "INV-100", '
        '"invoice_date": "2026-01-01", "currency": "USD", "total": 50.0, '
        '"tax_total": 0.0, "line_items": [{"description": "Item", "amount": 10.0}]}'
    )
    call2_broken_json = '{"document_type": "INVOICE", broken json...'

    with (
        patch("invoice_extractor.ingest.load", return_value=_fake_doc()),
        patch(
            "invoice_extractor.llm.extract",
            side_effect=[
                (call1_json, Usage(input_tokens=100, output_tokens=50, calls=1)),
                (call2_broken_json, Usage(input_tokens=150, output_tokens=10, calls=1)),
            ],
        ),
    ):
        result = process(FAKE_PDF_BYTES)
        assert result.meta.attempts == 2
        assert result.vendor == "Acme"
        assert result.total == 50.0
        assert result.needs_review is True


def test_retry_schema_failure_on_attempt1_succeeds_on_attempt2() -> None:
    """Test that a schema-broken attempt 1 is replaced if attempt 2 parses successfully."""
    call1_broken = "{bad json"
    call2_valid = (
        '{"document_type": "INVOICE", "vendor": "Acme", "invoice_number": "INV-100", '
        '"invoice_date": "2026-01-01", "currency": "USD", "total": 20.0, '
        '"tax_total": 0.0, "line_items": [{"description": "Item", "amount": 20.0}]}'
    )

    with (
        patch("invoice_extractor.ingest.load", return_value=_fake_doc()),
        patch(
            "invoice_extractor.llm.extract",
            side_effect=[
                (call1_broken, Usage(input_tokens=100, output_tokens=10, calls=1)),
                (call2_valid, Usage(input_tokens=150, output_tokens=50, calls=1)),
            ],
        ),
    ):
        result = process(FAKE_PDF_BYTES)
        assert result.meta.attempts == 2
        assert result.vendor == "Acme"
        assert result.needs_review is False


def test_non_retryable_issues_do_not_trigger_attempt2() -> None:
    """Test that non-retryable issues (e.g. AMBIGUOUS_TYPE) do not cause a second LLM call."""
    receipt_json = (
        '{"document_type": "RECEIPT", "vendor": null, "invoice_number": null, '
        '"invoice_date": null, "currency": null, "line_items": null, '
        '"tax_total": null, "total": null}'
    )

    with (
        patch("invoice_extractor.ingest.load", return_value=_fake_doc()),
        patch(
            "invoice_extractor.llm.extract",
            return_value=(receipt_json, Usage(input_tokens=100, output_tokens=50, calls=1)),
        ),
    ):
        result = process(FAKE_PDF_BYTES)
        assert result.meta.attempts == 1
        assert result.document_type == DocumentType.RECEIPT
        assert result.needs_review is True
        assert any("AMBIGUOUS_TYPE" in r for r in result.review_reasons)


def test_api_failure_returns_unavailable() -> None:
    """Test that ExtractionFailed returns an unavailable Result without raising."""
    with (
        patch("invoice_extractor.ingest.load", return_value=_fake_doc()),
        patch(
            "invoice_extractor.llm.extract",
            side_effect=ExtractionFailed("Quota exhausted"),
        ),
    ):
        result = process(FAKE_PDF_BYTES)
        assert result.unavailable_reason == UnavailableReason.API_FAILURE
        assert result.needs_review is True
        assert result.is_invoice is None
