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


def _calculate_cost(usage: Usage) -> float:
    """Compute total USD cost from input and output token consumption."""
    inp_cost = (usage.input_tokens / 1_000_000.0) * PRICE_INPUT_PER_M_USD
    out_cost = (usage.output_tokens / 1_000_000.0) * PRICE_OUTPUT_PER_M_USD
    return round(inp_cost + out_cost, 6)


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


def process(data: bytes, ocr_check: bool = False) -> Result:
    """Process raw document bytes end-to-end and produce a validated Result."""
    try:
        doc = ingest.load(data)
    except Unreadable as err:
        return Result.unavailable(err.reason, err.detail, model=GEMINI_MODEL)

    source_text: str | None = None
    is_ocr = False
    if doc.route == "text_layer":
        source_text = doc.text
    elif doc.route == "vision" and ocr_check:
        try:
            from invoice_extractor import ocr

            source_text = ocr.read_text(doc.images)
            is_ocr = True
        except Exception as err:
            logger.warning("OCR cross-check failed to extract text: %s", err)

    try:
        inv1, issues1, usage1 = _attempt_extraction(doc, source_text, is_ocr)
    except ExtractionFailed as err:
        return Result.unavailable(
            UnavailableReason.API_FAILURE,
            err.detail,
            model=GEMINI_MODEL,
            route=doc.route,
        )

    best_inv = inv1
    best_issues = issues1
    total_usage = usage1
    attempts = 1

    if issues1:
        feedback = prompts.feedback_message(issues1)
        try:
            inv2, issues2, usage2 = _attempt_extraction(doc, source_text, is_ocr, feedback)
            total_usage = usage1 + usage2
            attempts = 2
            # Keep the attempt with fewer issues; first attempt wins ties
            if len(issues2) < len(issues1):
                best_inv = inv2
                best_issues = issues2
        except ExtractionFailed as err:
            logger.warning("Retry attempt failed, falling back to attempt 1: %s", err)

    final_issues = best_issues + doc.warnings
    needs_review = bool(final_issues)
    review_reasons = [f"{iss.code.value}: {iss.detail}" for iss in final_issues]
    cost_usd = _calculate_cost(total_usage)

    meta = Meta(
        route=doc.route,
        model=GEMINI_MODEL,
        attempts=attempts,
        input_tokens=total_usage.input_tokens,
        output_tokens=total_usage.output_tokens,
        cost_usd=cost_usd,
    )

    if best_inv is not None:
        data_dict = best_inv.model_dump()
        return Result(
            **data_dict,
            needs_review=needs_review,
            review_reasons=review_reasons,
            meta=meta,
        )

    return Result(
        document_type=None,
        vendor=None,
        invoice_number=None,
        invoice_date=None,
        currency=None,
        line_items=None,
        tax_total=None,
        total=None,
        needs_review=True,
        review_reasons=review_reasons,
        meta=meta,
    )
