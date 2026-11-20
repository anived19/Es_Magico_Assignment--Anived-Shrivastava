"""Data schemas, enums, usage metrics, and domain exceptions."""

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field, computed_field


class DocumentType(StrEnum):
    """Categorization of financial and non-financial documents."""

    INVOICE = "INVOICE"
    CREDIT_NOTE = "CREDIT_NOTE"
    RECEIPT = "RECEIPT"
    QUOTATION = "QUOTATION"
    PURCHASE_ORDER = "PURCHASE_ORDER"
    BANK_STATEMENT = "BANK_STATEMENT"
    PERSONAL_OR_OTHER = "PERSONAL_OR_OTHER"


class UnavailableReason(StrEnum):
    """Failure reason when a document cannot be processed."""

    CORRUPT = "CORRUPT"
    ENCRYPTED = "ENCRYPTED"
    BLANK = "BLANK"
    UNSUPPORTED_TYPE = "UNSUPPORTED_TYPE"
    API_FAILURE = "API_FAILURE"


class IssueCode(StrEnum):
    """Deterministic validation issue codes."""

    SCHEMA_INVALID = "SCHEMA_INVALID"
    NON_INVOICE_HAS_FIELDS = "NON_INVOICE_HAS_FIELDS"
    MISSING_FIELD = "MISSING_FIELD"
    AMBIGUOUS_TYPE = "AMBIGUOUS_TYPE"
    INVALID_DATE = "INVALID_DATE"
    INVALID_CURRENCY = "INVALID_CURRENCY"
    TOTALS_MISMATCH = "TOTALS_MISMATCH"
    NOT_IN_SOURCE_TEXT = "NOT_IN_SOURCE_TEXT"
    LOW_SHARPNESS = "LOW_SHARPNESS"


class LineItem(BaseModel):
    """Single itemized charge line within an invoice."""

    description: str
    quantity: float | None = None
    amount: float


class Invoice(BaseModel):
    """Structured extraction target returned by Gemini."""

    document_type: DocumentType | None = None
    vendor: str | None = None
    invoice_number: str | None = None
    invoice_date: str | None = None
    currency: str | None = None
    line_items: list[LineItem] | None = None
    tax_total: float | None = None
    total: float | None = None

    @computed_field
    def is_invoice(self) -> bool | None:
        """Derive invoice boolean flag strictly from document_type."""
        if self.document_type is None:
            return None
        return self.document_type == DocumentType.INVOICE


@dataclass
class Issue:
    """A deterministic validation finding requiring attention."""

    code: IssueCode
    detail: str


@dataclass
class Usage:
    """Cumulative token consumption and API invocation count."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        """Sum token usages across multiple calls."""
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            calls=self.calls + other.calls,
        )


class Meta(BaseModel):
    """Processing metadata and execution metrics."""

    route: str
    model: str
    attempts: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


class Result(Invoice):
    """Final extraction result enriched with review flags and metadata."""

    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    unavailable_reason: UnavailableReason | None = None
    meta: Meta = Field(
        default_factory=lambda: Meta(
            route="unknown",
            model="",
            attempts=0,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
        )
    )

    @classmethod
    def unavailable(
        cls,
        reason: UnavailableReason,
        detail: str,
        model: str = "",
        route: str = "unknown",
    ) -> "Result":
        """Construct an unavailable result when input cannot be processed."""
        return cls(
            document_type=None,
            vendor=None,
            invoice_number=None,
            invoice_date=None,
            currency=None,
            line_items=None,
            tax_total=None,
            total=None,
            needs_review=True,
            review_reasons=[f"{reason.value}: {detail}"],
            unavailable_reason=reason,
            meta=Meta(
                route=route,
                model=model,
                attempts=0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
            ),
        )


class Unreadable(Exception):
    """Raised when an input document cannot be read or decoded."""

    def __init__(self, reason: UnavailableReason, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class ExtractionFailed(Exception):
    """Raised when LLM extraction fails after retry exhaustion."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail
