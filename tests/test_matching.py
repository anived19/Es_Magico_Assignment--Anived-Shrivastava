"""Unit tests for field matching, F1 calculation, and suffix normalization."""

from evaluation import matching
from evaluation.report import build_evaluation_report, evaluate_document
from invoice_extractor.schema import LineItem


def test_number_matching() -> None:
    """Assert number matching parses currency formatting and matches within tolerance."""
    assert matching.match_number("1,200.00", 1200)
    assert matching.match_number("₹ 1,200.00", "1200.00")
    assert matching.match_number(None, None)
    assert not matching.match_number("1,200.00", 1205.00)


def test_vendor_suffix_handling() -> None:
    """Verify vendor matching strips legal suffixes and uses fuzzy threshold."""
    assert matching.match_vendor("Acme Supplies Pvt Ltd", "Acme Supplies")
    assert matching.match_vendor("Kaveri Office Mart Pvt Ltd", "Kaveri Office Mart Limited")
    assert matching.match_vendor("Meridian Print & Pack LLP", "Meridian Print & Pack")
    assert not matching.match_vendor("Acme Supplies", "Different Vendor Corp")


def test_line_item_f1_matching() -> None:
    """Calculate F1 score across split, merged, and identical line items."""
    expected = [
        LineItem(description="Item A", quantity=2, amount=100.0),
        LineItem(description="Item B", quantity=1, amount=50.0),
    ]
    # Exact match
    pred_exact = [
        LineItem(description="Item A (other desc)", quantity=2, amount=100.0),
        LineItem(description="Item B", quantity=1, amount=50.0),
    ]
    assert matching.match_line_items(pred_exact, expected) == 1.0

    # One missing item (recall = 0.5, precision = 1.0 -> F1 = 0.666...)
    pred_missing = [LineItem(description="Item A", quantity=2, amount=100.0)]
    f1_missing = matching.match_line_items(pred_missing, expected)
    assert abs(f1_missing - 0.6666) < 0.01

    # Split row (two 50s instead of one 100)
    pred_split = [
        LineItem(description="Item A1", quantity=1, amount=50.0),
        LineItem(description="Item A2", quantity=1, amount=50.0),
        LineItem(description="Item B", quantity=1, amount=50.0),
    ]
    f1_split = matching.match_line_items(pred_split, expected)
    # only Item B matches -> tp=1, p=1/3, r=1/2 -> F1 = 2*(1/3*1/2)/(1/3+1/2) = 0.4
    assert abs(f1_split - 0.4) < 0.01


def test_null_handling() -> None:
    """Verify that null comparisons behave consistently."""
    assert matching.match_invoice_number(None, None)
    assert matching.match_vendor(None, None)
    assert not matching.match_invoice_number("INV-1", None)
    assert matching.match_line_items(None, None) == 1.0
    assert matching.match_line_items([], None) == 1.0


def test_non_invoice_exclusion_from_field_denominators() -> None:
    """Assert truth non-invoices do not artificially inflate field accuracy."""
    rec_inv = {
        "stem": "L01",
        "predicted": {
            "is_invoice": True,
            "vendor": "Acme",
            "invoice_number": "1",
            "invoice_date": "2026-01-01",
            "currency": "USD",
            "line_items": [{"description": "x", "quantity": 1, "amount": 10.0}],
            "tax_total": 0.0,
            "total": 10.0,
            "document_type": "INVOICE",
            "needs_review": False,
        },
        "expected": {
            "is_invoice": True,
            "vendor": "Acme",
            "invoice_number": "1",
            "invoice_date": "2026-01-01",
            "currency": "USD",
            "line_items": [{"description": "x", "quantity": 1, "amount": 10.0}],
            "tax_total": 0.0,
            "total": 10.0,
        },
    }
    rec_inv["eval"] = evaluate_document(rec_inv["predicted"], rec_inv["expected"])

    rec_non_inv = {
        "stem": "L08",
        "predicted": {
            "is_invoice": False,
            "vendor": None,
            "invoice_number": None,
            "invoice_date": None,
            "currency": None,
            "line_items": None,
            "tax_total": None,
            "total": None,
            "document_type": "BANK_STATEMENT",
            "needs_review": False,
        },
        "expected": {
            "is_invoice": False,
            "vendor": None,
            "invoice_number": None,
            "invoice_date": None,
            "currency": None,
            "line_items": None,
            "tax_total": None,
            "total": None,
        },
    }
    rec_non_inv["eval"] = evaluate_document(rec_non_inv["predicted"], rec_non_inv["expected"])

    report_text = build_evaluation_report([rec_inv, rec_non_inv])
    # Denominator for vendor should be 1, not 2
    assert "1/1" in report_text
    assert "2/2" in report_text  # is_invoice has denominator 2
