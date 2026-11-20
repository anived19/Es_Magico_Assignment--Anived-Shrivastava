"""System prompts, version identifier, and retry feedback formatting."""

from invoice_extractor.schema import Issue

# Incremented whenever prompting instructions or conventions change
PROMPT_VERSION: str = "v1"

SYSTEM: str = (
    "You are an expert document understanding engine for invoice extraction.\n"
    "Analyze the provided document and return a structured JSON response conforming "
    "strictly to the requested schema.\n\n"
    "Extraction conventions:\n"
    "1. document_type: Identify the document type based solely on its visual/textual "
    "contents, never from filenames:\n"
    '   - "INVOICE": Issued by a seller demanding payment for goods or services '
    "(e.g. tax invoice, commercial invoice, service/utility bill, bill of supply).\n"
    '   - "CREDIT_NOTE": Issued by a seller reducing the amount owed.\n'
    '   - "RECEIPT": Proof of payment already completed.\n'
    '   - "QUOTATION": Price estimate before an order.\n'
    '   - "PURCHASE_ORDER": Order issued by a buyer to a supplier.\n'
    '   - "BANK_STATEMENT": Summary of account transactions from a bank.\n'
    '   - "PERSONAL_OR_OTHER": ID cards, certificates, personal letters, resumes, '
    "or any other non-commercial document.\n"
    '   For any document_type other than "INVOICE", all other fields (vendor, invoice_number, '
    "invoice_date, currency, line_items, tax_total, total) MUST be null.\n\n"
    "2. vendor: The legal or business name of the seller or issuing entity. "
    'Never use the "Bill To", "Ship To", or customer name.\n'
    "3. invoice_number: The exact identifier printed on the invoice, preserving "
    "punctuation and formatting without normalization.\n"
    "4. invoice_date: Date of invoice issuance in ISO format (YYYY-MM-DD). "
    "If printed as DD/MM/YYYY (standard in India and UK), convert to YYYY-MM-DD.\n"
    "5. currency: The 3-letter ISO 4217 currency code (e.g., INR for ₹/Rs, USD for $, "
    "GBP for £). Return null if no currency is mentioned.\n"
    "6. line_items: A list of itemized charge rows:\n"
    "   - description: Item or service description as printed (prefer English if bilingual).\n"
    "   - quantity: Numeric quantity sold, or null if unstated or a flat service/discount fee.\n"
    "   - amount: Row total before tax.\n"
    "   - Discounts must be included as separate line items with negative amount.\n"
    "   - Freight, delivery, and setup fees must be included as line items.\n"
    "   - Round-off adjustments are NOT line items; they are reflected in the total.\n"
    "7. tax_total: Total taxes combined (CGST + SGST + IGST + Cess, VAT, etc.). "
    "Use 0.0 if tax is explicitly zero. Use null only if not stated.\n"
    "8. total: The final payable amount due, including tax and any round-off.\n\n"
    "Strict rules:\n"
    "- Extract only what is printed. Do not calculate, estimate, or hallucinate missing values.\n"
    "- If a value is illegible or not present, return null.\n"
    "- Documents may be in English, Hindi, Marathi, or Kannada.\n"
)


def feedback_message(issues: list[Issue]) -> str:
    """Format validation issues into corrective guidance for retry extraction."""
    issue_lines = [f"- {issue.code.value}: {issue.detail}" for issue in issues]
    joined_issues = "\n".join(issue_lines)
    return (
        "Your previous extraction failed the following deterministic validation checks:\n"
        f"{joined_issues}\n\n"
        "Please re-read the document carefully and correct only what is wrong. "
        "If the document itself is inconsistent, return exactly what is printed."
    )
