"""Unit tests for document ingestion, type sniffing, and unreadable handling."""

import io

import numpy as np
import pymupdf
import pytest
from PIL import Image

from invoice_extractor import ingest
from invoice_extractor.schema import UnavailableReason, Unreadable


def _make_digital_pdf(text: str) -> bytes:
    """Generate a clean single-page digital PDF in memory."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    data = doc.tobytes()
    doc.close()
    return data


def _make_image_bytes(color: tuple[int, int, int], noise: bool = False) -> bytes:
    """Generate a PNG image in memory with optional pixel noise."""
    arr = np.full((100, 100, 3), color, dtype=np.uint8)
    if noise:
        noise_arr = np.random.randint(0, 50, (100, 100, 3), dtype=np.uint8)
        arr = np.clip(arr + noise_arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_digital_pdf_routes_to_text_layer() -> None:
    """Verify that a standard digital PDF routes to text_layer."""
    pdf_bytes = _make_digital_pdf("Invoice INV-1002 from Acme Supplies Pvt Ltd. Total amount 500.")
    doc = ingest.load(pdf_bytes)
    assert doc.route == "text_layer"
    assert doc.text is not None
    assert "Invoice INV-1002" in doc.text


def test_png_image_routes_to_vision() -> None:
    """Verify that an image file routes to vision."""
    img_bytes = _make_image_bytes((120, 120, 120), noise=True)
    doc = ingest.load(img_bytes)
    assert doc.route == "vision"
    assert len(doc.images) == 1


def test_blank_image_raises_unreadable_blank() -> None:
    """Assert uniform blank images raise Unreadable with BLANK reason."""
    blank_bytes = _make_image_bytes((255, 255, 255), noise=False)
    with pytest.raises(Unreadable) as exc:
        ingest.load(blank_bytes)
    assert exc.value.reason == UnavailableReason.BLANK


def test_random_bytes_raises_unsupported_type() -> None:
    """Assert non-PDF, non-image random bytes raise UNSUPPORTED_TYPE."""
    random_bytes = b"This is not a pdf and definitely not an image format."
    with pytest.raises(Unreadable) as exc:
        ingest.load(random_bytes)
    assert exc.value.reason == UnavailableReason.UNSUPPORTED_TYPE


def test_truncated_pdf_raises_corrupt() -> None:
    """Assert truncated PDF bytes raise CORRUPT."""
    valid_pdf = _make_digital_pdf("Tax invoice content for testing truncated stream.")
    half_pdf = valid_pdf[: len(valid_pdf) // 3]
    with pytest.raises(Unreadable) as exc:
        ingest.load(half_pdf)
    assert exc.value.reason == UnavailableReason.CORRUPT


def test_content_sniffing_ignores_filenames() -> None:
    """Prove that file bytes determine extraction route without any filename metadata."""
    pdf_bytes = _make_digital_pdf("Sample invoice text with more than thirty non-space characters.")
    doc = ingest.load(pdf_bytes)
    assert doc.route == "text_layer"
