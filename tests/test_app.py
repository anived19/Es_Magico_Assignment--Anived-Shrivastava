"""Integration tests for the FastAPI web service endpoints."""

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import app
from invoice_extractor.schema import DocumentType, Meta, Result

client = TestClient(app)


def test_index_route() -> None:
    """Test that GET / successfully serves the static index.html."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Invoice Extraction Pipeline" in response.text


def test_extract_unsupported_bytes() -> None:
    """Test that POST /extract with unreadable binary payload returns an unavailable Result."""
    garbage = b"Not a real PDF or image stream."
    response = client.post(
        "/extract",
        files={"file": ("test.bin", garbage, "application/octet-stream")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_invoice"] is None
    assert data["needs_review"] is True
    assert data["unavailable_reason"] == "UNSUPPORTED_TYPE"


def test_extract_valid_mock_invoice() -> None:
    """Test that POST /extract correctly serializes pipeline Result for a valid invoice."""
    fake_meta = Meta(
        route="text_layer",
        model="gemini-test",
        attempts=1,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.0001,
    )
    fake_result = Result(
        document_type=DocumentType.INVOICE,
        vendor="Acme Supplies Pvt Ltd",
        invoice_number="INV-2041",
        invoice_date="2026-03-14",
        currency="INR",
        line_items=[],
        tax_total=10.0,
        total=110.0,
        needs_review=False,
        review_reasons=[],
        meta=fake_meta,
    )

    with patch("app.process", return_value=fake_result):
        response = client.post(
            "/extract?no_cache=true&ocr_check=false",
            files={"file": ("dummy.pdf", b"%PDF-1.4...", "application/pdf")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_invoice"] is True
        assert data["vendor"] == "Acme Supplies Pvt Ltd"
        assert data["needs_review"] is False
        assert data["meta"]["route"] == "text_layer"


def test_extract_cached_l01_file() -> None:
    """Test that POST /extract works end-to-end on real L01.pdf bytes."""
    doc_path = Path("data/documents/L01.pdf")
    if not doc_path.is_file():
        return

    content = doc_path.read_bytes()
    fake_json = (
        '{"document_type": "INVOICE", "vendor": "Northwind Logistics Ltd", '
        '"invoice_number": "NWL-88213", "invoice_date": "2026-04-08", '
        '"currency": "GBP", "total": 683.40, "tax_total": 113.90, '
        '"line_items": [{"description": "Freight", "quantity": 3.0, "amount": 435.0}, '
        '{"description": "Fuel", "quantity": 1.0, "amount": 62.5}, '
        '{"description": "Warehousing", "quantity": 12.0, "amount": 57.0}, '
        '{"description": "Admin", "quantity": 3.0, "amount": 15.0}]}'
    )
    from invoice_extractor.schema import Usage

    with patch(
        "invoice_extractor.llm.extract",
        return_value=(fake_json, Usage(input_tokens=942, output_tokens=696, calls=1)),
    ):
        response = client.post(
            "/extract",
            files={"file": ("L01.pdf", content, "application/pdf")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_invoice"] is True
        assert "Northwind" in data["vendor"]
