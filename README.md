# Invoice Extraction Pipeline

An invoice extraction pipeline with deterministic validation, honest measurement, and complete cost accounting. For detailed methodology, empirical measurements, failure analysis, and ablation findings, see [REPORT.md](REPORT.md).

Input: a PDF or image (digital invoice, mobile phone scan, or non-invoice such as a bank statement).  
Output: validated JSON conforming to the target schema, accompanied by a `needs_review` flag and granular review reasons.

---

## Architecture & Data Flow

```
[ Raw Bytes ]
      │
      ▼
┌─────────────┐
│  ingest.py  │  Content sniffing (PDF vs Image, blank detection, blur analysis)
└──────┬──────┘
       │
       ├─────────────────────────┬─────────────────────────┐
       ▼                         ▼                         ▼
[ text_layer route ]     [ vision route ]          [ Unreadable ]
(PyMuPDF digital text)  (rendered PNG pages)     (CORRUPT / BLANK / ENCRYPTED)
       │                         │                         │
       └───────────┬─────────────┘                         ▼
                   ▼                               [ Result.unavailable ]
┌──────────────────────────────────────┐
│                llm.py                │  Disk Cache (.cache/)
│   Gemini 3.5 Flash structured output │  Tenacity retry backoff (3 attempts)
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│              pipeline.py             │
│   Pydantic validation -> checks.py   │
└──────────────────┬───────────────────┘
                   │
        Issues detected? (Attempt 1)
        ├── Yes ──> Feedback message ──> llm.py (Attempt 2) ──> Best attempt
        └── No  ──> Keep Attempt 1
                   │
                   ▼
┌──────────────────────────────────────┐
│             final Result             │  schema fields at top-level,
│   needs_review, review_reasons, meta │  token accounting & USD cost
└──────────────────────────────────────┘
```

---

## Setup from Fresh Clone

### 1. Environment & Dependencies

```bash
# Clone the repository
git clone <repo-url>
cd invoice-extractor

# Create and activate virtual environment (Python 3.11+)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy example environment configuration
cp .env.example .env
```

Edit `.env` to include your credentials:
```ini
GEMINI_API_KEY=your_actual_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash
PRICE_INPUT_PER_M_USD=1.5
PRICE_OUTPUT_PER_M_USD=9.0
```
*Note: Token prices are user-supplied in `.env` based on the official Google AI Studio Developer API pricing page (https://ai.google.dev/pricing), Standard Paid Tier for `gemini-3.5-flash` / `gemini-3.6-flash` checked on November 22, 2026 ($0.75 / 1M input tokens, $3.75 / 1M output tokens, valid through December 31, 2026). Google's pricing documentation explicitly specifies that thinking tokens are billed at the standard output token rate.*

### 3. Data Directory Placement

Place your test files inside `data/`:
- `data/documents/` — PDFs and images (`L01.pdf`..`L10.pdf`, `U01.pdf`..`U10.pdf`, `.jpg`, etc.)
- `data/labels/` — Ground truth JSON labels (`L01.json`..`L10.json`)

*Note: Test files are user-supplied from the Es Magico assignment pack. A fresh clone will not include them. `data/`, `.cache/`, and `.env` are gitignored to prevent secret and data leakage.*

---

## Usage Commands

### Run the Automated Test Suite
```bash
pytest -v
```
Executes all unit and integration tests across ingestion, validation checks, pipeline retry logic, evaluation matching, and API endpoints.

### Extract a Single File (JSON on stdout)
```bash
python extract.py data/documents/L01.pdf
```
Logs are routed to `stderr`; clean formatted JSON is emitted to `stdout`.

### Extract an Entire Folder (Outputs per file in `--out`)
```bash
python extract.py data/documents --out outputs/
```
Processes all files sequentially, reading raw bytes, and saves `<stem>.json` for each document into `outputs/`.

### Run the Evaluation Harness
```bash
python evaluate.py
```
Pairs documents with ground truth in `data/labels/` by stem, runs extraction, prints four detailed markdown evaluation tables, and exports `reports/eval_latest.json`.

### Disable Disk Caching
All CLI commands support `--no-cache`:
```bash
python extract.py data/documents/L01.pdf --no-cache
python evaluate.py --no-cache
```

### Optional OCR Grounding Cross-Check (Phase 2)
To enable EasyOCR text grounding for vision-route scans and photos:
```bash
pip install -r requirements-ocr.txt
python extract.py data/documents/L02.jpg --ocr-check
python evaluate.py --ocr-check
```
*Note: OCR grounding is optional and flag-gated. It runs dual EasyOCR readers (`['en', 'hi', 'mr']` and `['en', 'kn']` on CPU) and applies fuzzy matching (`partial_ratio >= 90`) on invoice numbers.*

### Web Application & Visual Inspector (Phase 3)
To run the interactive side-by-side web application and API:
```bash
pip install -r requirements-app.txt
python -m uvicorn app:app --port 8000 --reload
# Or directly:
python app.py
```
Open [http://localhost:8000](http://localhost:8000) in your browser:
- **Side-by-side split view**: View uploaded document (PDF via iframe or mobile photo/scan via img) on the left pane and real-time extraction results on the right pane.
- **Verification & Review**: Instant `PASS` vs `NEEDS REVIEW` banner with granular failure reason tags and unreadable file alerts.
- **Itemized Table & Cost Accounting**: Table for extracted line items, breakdown of token usage, and live USD cost tracking.
- **Raw JSON Inspector**: Accordion view of the full structured JSON response with one-click clipboard copy.

#### API Endpoint: `POST /extract`
Accepts a document file via `multipart/form-data`:
```bash
curl -F "file=@data/documents/L01.pdf" "http://localhost:8000/extract"
```
Optional query parameters:
- `no_cache=true` (bypasses disk cache)
- `ocr_check=true` (enables Phase 2 OCR cross-check)

---

## Matching Rules

| Field | Rule |
|---|---|
| `is_invoice`, `currency` | Exact match; both `null` counts as correct. |
| `invoice_date` | Exact ISO `YYYY-MM-DD` string. |
| `invoice_number` | Equal after trimming, lowercasing, and removing whitespace. |
| `vendor` | Lowercased, punctuation stripped, corporate legal suffixes removed (`pvt`, `private`, `ltd`, `limited`, `llp`, `inc`, `co`), then `rapidfuzz.fuzz.ratio >= 85`. |
| `total`, `tax_total` | Stripped of currency symbols and commas, parsed as numbers, difference $\le 0.01$; both `null` counts as correct. |
| `line_items` | Greedy 1-to-1 matching of predicted to expected rows on `(abs(pred.amount - exp.amount) <= 0.01 and pred.quantity == exp.quantity)`. Score is F1 per document, averaged across documents. Descriptions are unconstrained because wording varies across languages. |
| Non-invoice documents | Correct only when `is_invoice` is `false` and every other field is `null`. |

*Note: Truth non-invoices are strictly excluded from field-level accuracy denominators so they do not artificially inflate extraction accuracy.*

---

## Handling Unreadable and Bad Inputs

Documents are sniffed purely from raw bytes (file names and extensions are completely ignored):

1. **Unsupported Type**: Files lacking the `%PDF-` header in the first 1024 bytes that Pillow cannot decode as an image raise `UNSUPPORTED_TYPE`.
2. **Corrupt Files**: Truncated or malformed PDF/image byte streams raise `CORRUPT`.
3. **Encrypted Files**: Password-protected PDFs raise `ENCRYPTED`.
4. **Blank Documents**: Grayscale standard deviation below `BLANK_STD_THRESHOLD` (5.0) across all pages raises `BLANK`.
5. **API Failures**: Transient Gemini API errors exhausted after 3 attempts return `API_FAILURE`.

All unreadable files return a valid `Result` object with `document_type=null`, `is_invoice=null`, all fields `null`, `needs_review=true`, and `unavailable_reason=<REASON>` (exit code 0), ensuring automated batches are never aborted by bad inputs.
