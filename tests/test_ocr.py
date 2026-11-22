"""Unit tests for EasyOCR grounding and fuzzy scan validation."""

from unittest.mock import MagicMock, patch

from invoice_extractor import checks, ocr
from invoice_extractor.schema import DocumentType, Invoice, IssueCode


def test_ocr_read_text_filters_confidence_and_sorts() -> None:
    """Verify read_text drops low-confidence words and sorts lines top-to-bottom."""
    mock_reader = MagicMock()
    # Format: (bbox, text, confidence)
    # Item 1: top=100, left=50
    # Item 2: top=50, left=20 (should sort before Item 1)
    # Item 3: low confidence (should be dropped)
    mock_reader.readtext.return_value = [
        ([(50, 100), (150, 100), (150, 120), (50, 120)], "Bottom Line", 0.85),
        ([(20, 50), (120, 50), (120, 70), (20, 70)], "Top Line", 0.90),
        ([(10, 10), (100, 10), (100, 30), (10, 30)], "Blurry Noise", 0.20),
    ]

    with patch.object(ocr, "_get_readers", return_value=(mock_reader, MagicMock())):
        text = ocr.read_text([b"fake_image_bytes"])

    lines = [line for line in text.splitlines() if not line.startswith("---")]
    assert "Blurry Noise" not in lines
    assert lines == ["Top Line", "Bottom Line"]


def test_grounding_fuzzy_matching_for_ocr() -> None:
    """Verify fuzzy grounding allows minor OCR character discrepancies."""
    inv = Invoice(
        document_type=DocumentType.INVOICE,
        vendor="Acme Corp",
        invoice_number="INV-2026-9901",
        invoice_date="2026-05-01",
        currency="INR",
        total=500.0,
    )
    # OCR transcribed slightly imperfectly with a substituted digit / space
    ocr_source = "Tax Invoice INV-2026-990I Total 500.00"

    # With is_ocr=True, partial_ratio >= 90 matches
    issues_ocr = checks.run_checks(inv, source_text=ocr_source, is_ocr=True)
    disallowed = {IssueCode.NOT_IN_SOURCE_TEXT, IssueCode.OCR_MISMATCH}
    assert not any(i.code in disallowed for i in issues_ocr)

    # With is_ocr=False (strict), exact substring fails
    issues_strict = checks.run_checks(inv, source_text=ocr_source, is_ocr=False)
    assert any(
        i.code == IssueCode.NOT_IN_SOURCE_TEXT and "INV-2026-9901" in i.detail
        for i in issues_strict
    )
