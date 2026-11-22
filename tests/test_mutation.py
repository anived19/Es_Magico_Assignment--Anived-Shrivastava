"""Mutation tests evaluating the review flag's precision and recall on synthetic errors."""

import copy

from invoice_extractor import checks
from invoice_extractor.schema import DocumentType, Invoice, IssueCode, LineItem

BASE_INVOICE = Invoice(
    document_type=DocumentType.INVOICE,
    vendor="Northwind Logistics Ltd",
    invoice_number="NWL-88213",
    invoice_date="2026-04-08",
    currency="GBP",
    tax_total=113.90,
    total=683.40,
    line_items=[
        LineItem(description="Freight", quantity=3.0, amount=435.0),
        LineItem(description="Fuel", quantity=1.0, amount=62.5),
        LineItem(description="Warehousing", quantity=12.0, amount=57.0),
        LineItem(description="Admin", quantity=3.0, amount=15.0),
    ],
)

BASE_SOURCE = (
    "Northwind Logistics Ltd\n"
    "TAX INVOICE\n"
    "Invoice No: NWL-88213\n"
    "Date: 2026-04-08\n"
    "GBP\n"
    "Items: 435.00, 62.50, 57.00, 15.00\n"
    "Tax: 113.90\n"
    "Total: 683.40\n"
)


def _mutate_total_mismatch(inv: Invoice) -> tuple[Invoice, IssueCode]:
    mut = copy.deepcopy(inv)
    mut.total = 750.00
    return mut, IssueCode.TOTALS_MISMATCH


def _mutate_invalid_date(inv: Invoice) -> tuple[Invoice, IssueCode]:
    mut = copy.deepcopy(inv)
    mut.invoice_date = "2038-01-01"
    return mut, IssueCode.INVALID_DATE


def _mutate_invalid_currency(inv: Invoice) -> tuple[Invoice, IssueCode]:
    mut = copy.deepcopy(inv)
    mut.currency = "GBPX"
    return mut, IssueCode.INVALID_CURRENCY


def _mutate_hallucinated_invoice_num(inv: Invoice) -> tuple[Invoice, IssueCode]:
    mut = copy.deepcopy(inv)
    mut.invoice_number = "FAKE-99999"
    return mut, IssueCode.NOT_IN_SOURCE_TEXT


def _mutate_missing_vendor(inv: Invoice) -> tuple[Invoice, IssueCode]:
    mut = copy.deepcopy(inv)
    mut.vendor = None
    return mut, IssueCode.MISSING_FIELD


def _mutate_empty_line_items(inv: Invoice) -> tuple[Invoice, IssueCode]:
    mut = copy.deepcopy(inv)
    mut.line_items = []
    return mut, IssueCode.MISSING_FIELD


def _mutate_ambiguous_type(inv: Invoice) -> tuple[Invoice, IssueCode]:
    mut = copy.deepcopy(inv)
    mut.document_type = DocumentType.RECEIPT
    mut.vendor = None
    mut.invoice_number = None
    mut.total = None
    return mut, IssueCode.AMBIGUOUS_TYPE


def _mutate_non_invoice_with_fields(inv: Invoice) -> tuple[Invoice, IssueCode]:
    mut = copy.deepcopy(inv)
    mut.document_type = DocumentType.BANK_STATEMENT
    return mut, IssueCode.NON_INVOICE_HAS_FIELDS


def _mutate_misclassified_invoice(inv: Invoice) -> tuple[Invoice, IssueCode]:
    mut = Invoice(document_type=DocumentType.BANK_STATEMENT)
    return mut, IssueCode.SUSPECT_NON_INVOICE


MUTATORS = [
    _mutate_total_mismatch,
    _mutate_invalid_date,
    _mutate_invalid_currency,
    _mutate_hallucinated_invoice_num,
    _mutate_missing_vendor,
    _mutate_empty_line_items,
    _mutate_ambiguous_type,
    _mutate_non_invoice_with_fields,
    _mutate_misclassified_invoice,
]


def test_mutation_suite_flags_every_injected_error() -> None:
    """Run full mutation suite verifying every synthetic defect is caught with 100% recall."""
    # 1. Base invoice is clean
    base_issues = checks.run_checks(BASE_INVOICE, source_text=BASE_SOURCE)
    assert len(base_issues) == 0

    tp = 0
    fn = 0
    for mutator in MUTATORS:
        mutated_inv, expected_code = mutator(BASE_INVOICE)
        issues = checks.run_checks(mutated_inv, source_text=BASE_SOURCE)
        issue_codes = {iss.code for iss in issues}

        if expected_code in issue_codes:
            tp += 1
        else:
            fn += 1

    total_mutations = len(MUTATORS)
    recall = tp / (tp + fn) * 100
    assert fn == 0
    assert tp == total_mutations
    assert recall == 100.0
