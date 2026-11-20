"""Configuration and environment variables for invoice extraction."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get_required_env(name: str) -> str:
    """Read a mandatory environment variable or raise an informative error."""
    val = os.getenv(name)
    if not val:
        raise ValueError(f"Missing required environment variable: {name}")
    return val


# Google Gemini API key from AI Studio
GEMINI_API_KEY: str = _get_required_env("GEMINI_API_KEY")

# Flash-class multimodal model ID recommended in Gemini docs
GEMINI_MODEL: str = _get_required_env("GEMINI_MODEL")

# Pricing per 1M tokens in USD from Google AI pricing page
PRICE_INPUT_PER_M_USD: float = float(_get_required_env("PRICE_INPUT_PER_M_USD"))
PRICE_OUTPUT_PER_M_USD: float = float(_get_required_env("PRICE_OUTPUT_PER_M_USD"))

# Maximum pages extracted to bound latency and token consumption
MAX_PAGES: int = 6

# DPI for rendering PDF pages to image for the vision route
RENDER_DPI: int = 150

# Minimum non-space characters for a page to count as digital
MIN_TEXT_CHARS: int = 30

# Maximum page area ratio occupied by embedded images before treating as scan
SCAN_IMAGE_COVERAGE: float = 0.5

# Grayscale standard deviation below which a page is treated as blank
BLANK_STD_THRESHOLD: float = 5.0

# Variance of Laplacian below which an image triggers a low sharpness warning
BLUR_THRESHOLD: float = 50.0

# Maximum absolute difference in currency units between item sum + tax and total
TOTALS_TOLERANCE: float = 1.0

# Maximum retry attempts for transient Gemini API failures
MAX_API_ATTEMPTS: int = 3

# Magic byte prefix identifying standard PDF documents
PDF_MAGIC: bytes = b"%PDF-"

# Cache directory location for raw Gemini responses
CACHE_DIR: Path = Path(".cache")

# Minimum word confidence threshold to retain text detection from EasyOCR
OCR_CONFIDENCE_THRESHOLD: float = 0.4

# Minimum RapidFuzz partial_ratio for fuzzy grounding of invoice numbers in scans
OCR_FUZZY_RATIO: float = 90.0
