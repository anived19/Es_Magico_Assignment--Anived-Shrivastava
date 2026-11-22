"""Orchestrates document loading, extraction attempts, validation, and cost calculation."""

import logging

from pydantic import ValidationError

from invoice_extractor import checks, ingest, llm, prompts
from invoice_extractor.config import (
    GEMINI_MODEL,
    PRICE_INPUT_PER_M_USD,
    PRICE_OUTPUT_PER_M_USD,
)
from invoice_extractor.ingest import Document
from invoice_extractor.schema import (
    ExtractionFailed,
    Invoice,
    Issue,
    IssueCode,
    Meta,
    Result,
    UnavailableReason,
    Unreadable,
    Usage,
)

logger = logging.getLogger(__name__)

NON_RETRYABLE: set[IssueCode] = {
    IssueCode.AMBIGUOUS_TYPE,
    IssueCode.OCR_MISMATCH,
    IssueCode.OCR_UNAVAILABLE,
    IssueCode.SUSPECT_NON_INVOICE,
}


def _calculate_cost(usage: Usage) -> float:
    """Compute total USD cost from input and output token consumption."""
    inp_cost = (usage.input_tokens / 1_000_000.0) * PRICE_INPUT_PER_M_USD
    out_cost = (usage.output_tokens / 1_000_000.0) * PRICE_OUTPUT_PER_M_USD
    return round(inp_cost + out_cost, 6)


def _resolve_source_text(doc: Document, ocr_check: bool) -> tuple[str | None, bool, list[Issue]]:
    """Determine grounding source text from native PDF layer or optional EasyOCR."""
    if doc.route == "text_layer":
        return doc.text, False, []
    if doc.route == "vision" and ocr_check:
        try:
            from invoice_extractor import ocr

            text = ocr.read_text(doc.images)
            return text, True, []
        except ImportError as err:
            logger.error("EasyOCR requested but not installed: %s", err)
            raise RuntimeError(
                "easyocr is not installed. Install requirements-ocr.txt to use --ocr-check."
            ) from err
        except Exception as err:
            logger.warning("OCR processing failed: %s", err)
            return None, True, [Issue(IssueCode.OCR_UNAVAILABLE, f"OCR processing failed: {err}")]
    return None, False, []


def _attempt_extraction(
    doc: Document,
    source_text: str | None,
    is_ocr: bool,
    feedback: str | None = None,
) -> tuple[Invoice | None, list[Issue], Usage]:
    """Perform one extraction attempt and validate returned structured data."""
    raw_json, usage = llm.extract(doc, feedback)
    try:
        invoice = Invoice.model_validate_json(raw_json)
    except ValidationError as err:
        return None, [Issue(IssueCode.SCHEMA_INVALID, f"JSON validation error: {err}")], usage

    issues = checks.run_checks(invoice, source_text, is_ocr=is_ocr)
    return invoice, issues, usage


def _select_best_attempt(
    attempt1: tuple[Invoice | None, list[Issue]],
    attempt2: tuple[Invoice | None, list[Issue]],
) -> tuple[Invoice | None, list[Issue]]:
    """Select the superior attempt prioritizing successful schema parsing and issue count."""
    return min([attempt1, attempt2], key=lambda a: (a[0] is None, len(a[1])))


def process(data: bytes, ocr_check: bool = False) -> Result:
    """Process raw document bytes end-to-end and produce a validated Result."""
    try:
        doc = ingest.load(data)
    except Unreadable as err:
        return Result.unavailable(err.reason, err.detail, model=GEMINI_MODEL)

    source_text, is_ocr, ocr_issues = _resolve_source_text(doc, ocr_check)

    try:
        inv1, issues1, usage1 = _attempt_extraction(doc, source_text, is_ocr)
    except ExtractionFailed as err:
        return Result.unavailable(
            UnavailableReason.API_FAILURE, err.detail, model=GEMINI_MODEL, route=doc.route
        )

    best_inv, best_issues = inv1, issues1
    total_usage = usage1
    attempts = 1

    retryable = [iss for iss in issues1 if iss.code not in NON_RETRYABLE]
    if retryable:
        feedback = prompts.feedback_message(retryable)
        try:
            inv2, issues2, usage2 = _attempt_extraction(doc, source_text, is_ocr, feedback)
            total_usage = usage1 + usage2
            attempts = 2
            best_inv, best_issues = _select_best_attempt((inv1, issues1), (inv2, issues2))
        except ExtractionFailed as err:
            logger.warning("Retry attempt failed, falling back to attempt 1: %s", err)

    final_issues = best_issues + doc.warnings + ocr_issues
    needs_review = bool(final_issues)
    review_reasons = [f"{iss.code.value}: {iss.detail}" for iss in final_issues]

    meta = Meta(
        route=doc.route,
        model=GEMINI_MODEL,
        attempts=attempts,
        input_tokens=total_usage.input_tokens,
        output_tokens=total_usage.output_tokens,
        cost_usd=_calculate_cost(total_usage),
    )

    if best_inv is not None:
        return Result(
            **best_inv.model_dump(),
            needs_review=needs_review,
            review_reasons=review_reasons,
            meta=meta,
        )

    return Result(
        document_type=None,
        needs_review=True,
        review_reasons=review_reasons,
        meta=meta,
    )
