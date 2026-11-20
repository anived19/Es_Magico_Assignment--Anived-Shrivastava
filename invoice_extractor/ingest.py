"""Document ingestion, unreadable detection, routing, and sharpness assessment."""

import hashlib
import io
from dataclasses import dataclass, field

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageOps, UnidentifiedImageError

from invoice_extractor.config import (
    BLANK_STD_THRESHOLD,
    BLUR_THRESHOLD,
    MAX_PAGES,
    MIN_TEXT_CHARS,
    PDF_MAGIC,
    RENDER_DPI,
    SCAN_IMAGE_COVERAGE,
)
from invoice_extractor.schema import Issue, IssueCode, UnavailableReason, Unreadable


@dataclass
class Document:
    """Preprocessed document package with routing metadata and raw content."""

    sha256: str
    route: str
    text: str | None = None
    images: list[bytes] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)


def _check_blank_image(gray: np.ndarray) -> bool:
    """Determine if a grayscale image matrix is nearly uniform blank content."""
    return float(np.std(gray)) < BLANK_STD_THRESHOLD


def _compute_laplacian_var(gray: np.ndarray) -> float:
    """Compute the variance of the Laplacian as a proxy for image sharpness."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _is_digital_page(page: pymupdf.Page) -> bool:
    """Evaluate whether a PDF page contains substantial text and low scan coverage."""
    text = page.get_text("text").strip()
    non_space_chars = len("".join(text.split()))
    if non_space_chars < MIN_TEXT_CHARS:
        return False
    rect = page.rect
    page_area = rect.width * rect.height
    if page_area <= 0:
        return False
    img_info = page.get_image_info()
    img_area = sum(info["bbox"].width * info["bbox"].height for info in img_info if "bbox" in info)
    return (img_area / page_area) < SCAN_IMAGE_COVERAGE


def _render_page_png(page: pymupdf.Page) -> tuple[bytes, np.ndarray]:
    """Render a PDF page to PNG bytes and return its grayscale array."""
    pix = page.get_pixmap(dpi=RENDER_DPI)
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if pix.n >= 3 else img_array[:, :, 0]
    return pix.tobytes("png"), gray


def _load_pdf(data: bytes, doc_sha: str) -> Document:
    """Extract or render PDF content with unreadable checks and routing decision."""
    try:
        pdf_doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise Unreadable(UnavailableReason.CORRUPT, f"Cannot open PDF: {exc}") from exc

    if pdf_doc.needs_pass:
        pdf_doc.close()
        raise Unreadable(UnavailableReason.ENCRYPTED, "PDF is password protected")

    page_count = min(len(pdf_doc), MAX_PAGES)
    if page_count == 0:
        pdf_doc.close()
        raise Unreadable(UnavailableReason.CORRUPT, "PDF has zero pages")

    all_digital = all(_is_digital_page(pdf_doc[i]) for i in range(page_count))

    if all_digital:
        text_blocks: list[str] = []
        is_all_blank = True
        for i in range(page_count):
            page = pdf_doc[i]
            p_text = page.get_text("text", sort=True)
            text_blocks.append(f"--- page {i + 1} ---\n{p_text}")
            _, gray = _render_page_png(page)
            if not _check_blank_image(gray):
                is_all_blank = False
        pdf_doc.close()
        if is_all_blank:
            raise Unreadable(UnavailableReason.BLANK, "PDF text pages are all visually blank")
        return Document(sha256=doc_sha, route="text_layer", text="\n".join(text_blocks))

    png_images: list[bytes] = []
    warnings: list[Issue] = []
    all_blank = True

    for i in range(page_count):
        png_bytes, gray = _render_page_png(pdf_doc[i])
        png_images.append(png_bytes)
        if not _check_blank_image(gray):
            all_blank = False
        if i == 0:
            var = _compute_laplacian_var(gray)
            if var < BLUR_THRESHOLD:
                warnings.append(
                    Issue(
                        IssueCode.LOW_SHARPNESS,
                        f"Page 1 Laplacian variance {var:.1f} below threshold {BLUR_THRESHOLD}",
                    )
                )

    pdf_doc.close()
    if all_blank:
        raise Unreadable(UnavailableReason.BLANK, "All PDF rendered pages are blank")
    return Document(sha256=doc_sha, route="vision", images=png_images, warnings=warnings)


def _load_image(data: bytes, doc_sha: str) -> Document:
    """Decode and inspect standalone image byte streams."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except UnidentifiedImageError as exc:
        raise Unreadable(UnavailableReason.UNSUPPORTED_TYPE, "Unsupported file format") from exc
    except Exception as exc:
        raise Unreadable(UnavailableReason.CORRUPT, f"Corrupt image file: {exc}") from exc

    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    cv_img = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if cv_img is None or _check_blank_image(cv_img):
        raise Unreadable(UnavailableReason.BLANK, "Image content is blank")

    warnings: list[Issue] = []
    var = _compute_laplacian_var(cv_img)
    if var < BLUR_THRESHOLD:
        warnings.append(
            Issue(
                IssueCode.LOW_SHARPNESS,
                f"Image Laplacian variance {var:.1f} below threshold {BLUR_THRESHOLD}",
            )
        )
    return Document(sha256=doc_sha, route="vision", images=[png_bytes], warnings=warnings)


def load(data: bytes) -> Document:
    """Inspect raw document bytes and return a routed Document or raise Unreadable."""
    doc_sha = hashlib.sha256(data).hexdigest()
    if data[:1024].startswith(PDF_MAGIC):
        return _load_pdf(data, doc_sha)
    return _load_image(data, doc_sha)
