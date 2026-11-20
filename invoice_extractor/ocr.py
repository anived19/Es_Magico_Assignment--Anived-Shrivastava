"""EasyOCR grounding engine for visual scans and document images."""

import functools
import logging

from invoice_extractor.config import OCR_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


@functools.cache
def _get_readers() -> tuple[object, object]:
    """Lazily initialize and cache EasyOCR reader instances for supported languages."""
    try:
        import easyocr
    except ImportError as err:
        raise ImportError(
            "easyocr is required for OCR grounding. Install it via requirements-ocr.txt."
        ) from err

    logger.info("Initializing EasyOCR readers (gpu=False)...")
    reader_mr = easyocr.Reader(["en", "hi", "mr"], gpu=False)
    reader_kn = easyocr.Reader(["en", "kn"], gpu=False)
    return reader_mr, reader_kn


def _extract_page_lines(img_bytes: bytes, readers: tuple[object, object]) -> list[str]:
    """Run dual readers on a page image and return confident detections sorted by position."""
    raw_detections: list[tuple[float, float, str]] = []
    seen_boxes: set[tuple[int, int]] = set()

    for reader in readers:
        try:
            results = reader.readtext(img_bytes)
        except Exception as exc:
            logger.warning("EasyOCR detection error: %s", exc)
            continue

        for bbox, text, conf in results:
            if conf < OCR_CONFIDENCE_THRESHOLD or not text.strip():
                continue
            top = min(pt[1] for pt in bbox)
            left = min(pt[0] for pt in bbox)
            grid_key = (round(top / 10.0), round(left / 10.0))
            if grid_key in seen_boxes:
                continue
            seen_boxes.add(grid_key)
            raw_detections.append((top, left, text.strip()))

    raw_detections.sort(key=lambda item: (round(item[0] / 15.0), item[1]))
    return [item[2] for item in raw_detections]


def read_text(images: list[bytes]) -> str:
    """Extract and ground text lines from an ordered list of document images."""
    readers = _get_readers()
    all_lines: list[str] = []
    for idx, img_bytes in enumerate(images):
        page_lines = _extract_page_lines(img_bytes, readers)
        all_lines.append(f"--- ocr page {idx + 1} ---")
        all_lines.extend(page_lines)
    return "\n".join(all_lines)
