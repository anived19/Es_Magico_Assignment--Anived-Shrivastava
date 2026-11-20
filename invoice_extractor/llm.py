"""Gemini API calling, transient retry policy, token accounting, and response caching."""

import functools
import hashlib
import logging

from google import genai
from google.genai import errors, types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from invoice_extractor import cache
from invoice_extractor.config import GEMINI_API_KEY, GEMINI_MODEL, MAX_API_ATTEMPTS
from invoice_extractor.ingest import Document
from invoice_extractor.prompts import PROMPT_VERSION, SYSTEM
from invoice_extractor.schema import ExtractionFailed, Invoice, Usage

logger = logging.getLogger(__name__)


@functools.cache
def _get_client() -> genai.Client:
    """Return a cached, authenticated Gemini API client instance."""
    return genai.Client(api_key=GEMINI_API_KEY)


def _build_contents(doc: Document, feedback: str | None = None) -> list[str | types.Part]:
    """Assemble multimodal or text contents and optional feedback for the API."""
    contents: list[str | types.Part] = []
    if doc.route == "text_layer":
        contents.append(doc.text or "")
    else:
        for img_bytes in doc.images:
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
    if feedback:
        contents.append(feedback)
    return contents


def _compute_cache_key(doc_sha: str, model: str, version: str, feedback: str | None) -> str:
    """Generate deterministic sha256 cache key from extraction parameters."""
    seed = f"{doc_sha}:{model}:{version}:{feedback or ''}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _parse_usage(usage_meta: types.GenerateContentResponseUsageMetadata | None) -> Usage:
    """Extract prompt and candidate token counts from Gemini response metadata."""
    if not usage_meta:
        return Usage(input_tokens=0, output_tokens=0, calls=1)
    inp = (
        getattr(usage_meta, "prompt_token_count", 0)
        or getattr(usage_meta, "total_input_tokens", 0)
        or 0
    )
    cands = (
        getattr(usage_meta, "candidates_token_count", 0)
        or getattr(usage_meta, "total_output_tokens", 0)
        or 0
    )
    thoughts = (
        getattr(usage_meta, "thoughts_token_count", 0)
        or getattr(usage_meta, "total_thought_tokens", 0)
        or 0
    )
    return Usage(input_tokens=int(inp), output_tokens=int(cands + thoughts), calls=1)


@retry(
    stop=stop_after_attempt(MAX_API_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(
        (errors.APIError, errors.ServerError, TimeoutError, ConnectionError)
    ),
    reraise=True,
)
def _call_gemini_api(contents: list[str | types.Part]) -> tuple[str, Usage]:
    """Execute raw Gemini generation request with structured output configuration."""
    client = _get_client()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=Invoice,
    )
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=config,
    )
    text = resp.text or "{}"
    usage = _parse_usage(resp.usage_metadata)
    return text, usage


def extract(doc: Document, feedback: str | None = None) -> tuple[str, Usage]:
    """Call Gemini for structured extraction, managing retries, token usage, and disk caching."""
    cache_key = _compute_cache_key(doc.sha256, GEMINI_MODEL, PROMPT_VERSION, feedback)
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        usage_dict = cached_data.get("usage", {})
        usage = Usage(
            input_tokens=usage_dict.get("input_tokens", 0),
            output_tokens=usage_dict.get("output_tokens", 0),
            calls=usage_dict.get("calls", 1),
        )
        return cached_data.get("raw_json", "{}"), usage

    contents = _build_contents(doc, feedback)
    try:
        raw_json, usage = _call_gemini_api(contents)
    except Exception as exc:
        logger.error("Gemini extraction failed after retries: %s", exc)
        raise ExtractionFailed(f"Gemini API invocation failed: {exc}") from exc

    cache.put(
        cache_key,
        {
            "raw_json": raw_json,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "calls": usage.calls,
            },
        },
    )
    return raw_json, usage
