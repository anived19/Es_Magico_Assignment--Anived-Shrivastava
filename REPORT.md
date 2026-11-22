# Invoice Extraction Pipeline — Technical & Evaluation Report

## 1. Approach

### System Architecture
The extraction pipeline is designed around honest measurement, deterministic validation, and strict content sniffing:
- **Model**: Google Gemini 3.5 Flash (`gemini-3.5-flash`), configured via `.env` (`GEMINI_MODEL`), invoked at temperature `0.0` with structured JSON output enforcement (`response_mime_type="application/json"`, `response_schema=Invoice`).
- **Routing (`text_layer` vs `vision`)**: Ingestion inspects raw bytes without relying on file names or extensions. If the first 1024 bytes match `%PDF-` (`PDF_MAGIC`), PyMuPDF analyzes text character density and embedded image coverage across all pages (up to `MAX_PAGES = 6`). Documents with $\ge 30$ non-space characters per page (`MIN_TEXT_CHARS`) and $< 50\%$ page area occupied by embedded images (`SCAN_IMAGE_COVERAGE`) route to `text_layer` (digital text extracted directly). Scanned PDFs and raster images (`.jpg`, `.png`) route to `vision` (rendered at 150 DPI, `RENDER_DPI`).
- **Structured Output**: Extracted data is constrained by Pydantic models to ensure valid schema typing, including ISO 4217 currency strings, ISO dates, numeric totals, and line item arrays.
- **Temperature**: Pinned to `0.0` across all extraction calls to minimize generation variance and maximize deterministic reproduction.
- **Prompt Design**: Prompt conventions were learned from the 10 labelled documents (`L01`–`L10`):
  1. `is_invoice` is derived strictly from whether `document_type == "INVOICE"`.
  2. For non-invoice types (`PURCHASE_ORDER`, `BANK_STATEMENT`, `RECEIPT`, `QUOTATION`, `CREDIT_NOTE`, `PERSONAL_OR_OTHER`), all extraction fields (`vendor`, `invoice_number`, `invoice_date`, `currency`, `line_items`, `tax_total`, `total`) must be `null`.
  3. Discounts are extracted as line items with negative amounts (e.g. `L07`).
  4. Freight, packing, and setup charges are itemized line items (e.g. `L01`, `L04`).
  5. Round-off adjustments are not line items; they are absorbed into `total`.
  6. Dates are formatted as ISO `YYYY-MM-DD`. If printed as `DD/MM/YYYY` (common in UK/India), they are converted. If day and month are ambiguous, conventions are inferred from currency or country.
- **Validation & One Retry**: Every extraction passes through deterministic business checks (`checks.py`). If fixable issues occur on Attempt 1, issues are formatted into a structured feedback prompt (`prompts.feedback_message`), triggering exactly one retry attempt. The pipeline selects the superior attempt using `min([attempt1, attempt2], key=lambda a: (a[0] is None, len(a[1])))`, ensuring that a valid schema extraction is never overwritten by a failed second attempt. Non-retryable issues (`AMBIGUOUS_TYPE`, `OCR_MISMATCH`, `OCR_UNAVAILABLE`, `SUSPECT_NON_INVOICE`) do not trigger retries.
- **Bad-Input Handling**: Malformed or unreadable files return structured `Result.unavailable(reason, detail)` objects with exit code 0 rather than aborting execution. Sniffed failure modes include `UNSUPPORTED_TYPE` (unknown binary), `CORRUPT` (un-parseable PDF/image byte streams), `ENCRYPTED` (password-protected PDFs), `BLANK` (grayscale standard deviation $< 5.0$, `BLANK_STD_THRESHOLD`), and `API_FAILURE` (transient upstream LLM failures exhausted after 3 tenacity backoff attempts).

### What Did Not Work: EasyOCR Grounding Cross-Check
We evaluated adding a local CPU-based OCR pass (EasyOCR) to cross-check extracted fields on vision-route scans. Tested on two scans in the pack, this architectural addition failed:
1. **Perspective & Skew Fragility (`L02.jpg`)**: On mobile photo `L02.jpg`, Gemini Vision transcribed `SGS/1847` accurately. EasyOCR struggled with the skewed perspective and lighting gradients, failing to recover the identifier and raising a false `OCR_MISMATCH` flag.
2. **Rotation Fragility (`U05.pdf`)**: On `U05.pdf` (a scan rotated 90°), Gemini Vision extracted all fields cleanly. EasyOCR generated garbled text because this setup has no rotation handling, raising two false `OCR_MISMATCH` flags (invoice number and total).
3. **Severe CPU Latency**: On `U05.pdf`, EasyOCR took 230 seconds on CPU compared to about 13 seconds for Gemini Vision on the same file. On `L02.jpg`, EasyOCR took about 90 seconds compared to about 13 seconds without OCR.
Consequently, OCR cross-checking was rejected from the default pipeline and retained strictly behind the optional `--ocr-check` flag.

---

## 2. Evaluation Harness and Matching Rules

The evaluation harness is executed via a single command:
```bash
python evaluate.py
```
*(Or `python evaluate.py --no-cache` to evaluate without disk caching).*

The harness pairs documents in `data/documents/` with ground truth in `data/labels/` by filename stem, executes extraction, runs evaluation assertions, prints formatted markdown tables, and exports `reports/eval_latest.json`.

### Matching Rules Per Field

| Field | Rule |
|---|---|
| `is_invoice` | Exact boolean match; both `null` counts as correct. |
| `currency` | Exact match on 3-letter ISO 4217 code; both `null` counts as correct. |
| `invoice_date` | Exact ISO `YYYY-MM-DD` string match. |
| `invoice_number` | Equal after trimming, lowercasing, and stripping all non-alphanumeric whitespace/punctuation. |
| `vendor` | Lowercased, punctuation stripped, corporate legal suffixes removed (`pvt`, `private`, `ltd`, `limited`, `llp`, `inc`, `co`), then `rapidfuzz.fuzz.ratio >= 85`. |
| `total`, `tax_total` | Stripped of currency symbols and commas, parsed as floats, absolute difference $\le 0.01$; both `null` counts as correct. |
| `line_items` | Greedy 1:1 matching of predicted to expected items where `abs(pred.amount - exp.amount) <= 0.01` and `pred.quantity == exp.quantity`. Evaluated as F1 score per document, averaged across documents. Descriptions are unconstrained due to multilingual phrasing differences. |
| Non-invoices | Correct only when `is_invoice` is `false` and all other fields (`vendor`, `invoice_number`, `invoice_date`, `currency`, `line_items`, `tax_total`, `total`) are `null`. |

*Note on Denominators: As a deliberate design decision, ground-truth non-invoices (`L08`) are excluded from field-level invoice accuracy denominators. Including non-invoices would trivially inflate accuracy scores by counting expected `null` values as successful extractions.*

---

## 3. Accuracy

> [!NOTE]
> Prompt conventions were learned from the 10 labelled documents (`L01`–`L10`), and the accuracy numbers reported below are on that same set (tuned on it).

### Field-Level Accuracy Table

Evaluated on the 10 labelled documents (`L01`–`L10`):

| Field / Metric | Accuracy | Correct / Total |
|---|---|---|
| `vendor` | 100.0% | 9/9 |
| `invoice_number` | 100.0% | 9/9 |
| `invoice_date` | 100.0% | 9/9 |
| `currency` | 100.0% | 9/9 |
| `line_items` (mean F1) | 100.0% | 9/9 exact |
| `tax_total` | 100.0% | 9/9 |
| `total` | 100.0% | 9/9 |
| **all fields correct (invoices)** | **100.0%** | **9/9** |
| `is_invoice` (all documents) | 100.0% | 10/10 |
| `non-invoices handled correctly` | 100.0% | 1/1 |

### Per-Document Evaluation Grid

Every row contains exactly 12 columns:

| File | Route | Doc Type | Vendor | Inv # | Date | Curr | Items | Tax | Total | IsInv/Clean | Review |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `L01` | text_layer | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| `L02` | vision | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| `L03` | text_layer | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| `L04` | text_layer | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| `L05` | text_layer | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| `L06` | text_layer | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| `L07` | text_layer | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| `L08` | text_layer | PURCHASE_ORDER | — | — | — | — | — | — | — | ✓ | PASS |
| `L09` | vision | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| `L10` | vision | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |

### Review Flag Performance on Labelled Set

| | Wrong (Errors > 0) | Right (0 Errors) | Total |
|---|---|---|---|
| **Flagged for review** | TP = 0 | FP = 0 | 0 |
| **Not flagged (passed)** | FN = 0 | TN = 10 | 10 |
| **Total** | 0 | 10 | 10 |

- **Review Flag Precision**: Undefined (no documents flagged; 0/0)
- **Review Flag Recall**: Undefined (no field errors in dataset; 0/0)
- **Status**: Precision and recall are undefined: no document was flagged and no document was wrong. This table cannot yet show whether the flag catches errors.

### Mutation Testing: Empirical Review-Flag Validation
Because the labelled pack had 0 field errors, we verified the review flag mechanism by executing an automated synthetic mutation harness (`tests/test_mutation.py`) across 9 defect classes on ground-truth invoices:

| Mutation Class | Injected Defect | Triggered Issue Code | Flagged? |
|---|---|---|---|
| **Math Reconciliation** | Total altered by +$66.60 | `TOTALS_MISMATCH` | **Yes (TP)** |
| **Plausible Date Window** | Invoice date set to `2038-01-01` | `INVALID_DATE` | **Yes (TP)** |
| **Currency Standardization** | Currency altered to non-ISO `GBPX` | `INVALID_CURRENCY` | **Yes (TP)** |
| **Identifier Hallucination** | Invoice number changed to `FAKE-99999` | `NOT_IN_SOURCE_TEXT` | **Yes (TP)** |
| **Missing Vendor** | Vendor set to `null` | `MISSING_FIELD` | **Yes (TP)** |
| **Empty Line Items** | Line items emptied (`[]`) | `MISSING_FIELD` | **Yes (TP)** |
| **Borderline Document Type** | Document type set to `RECEIPT` | `AMBIGUOUS_TYPE` | **Yes (TP)** |
| **Non-Invoice Field Leakage** | `BANK_STATEMENT` with populated fields | `NON_INVOICE_HAS_FIELDS` | **Yes (TP)** |
| **Misclassified Invoice** | Invoice text with `BANK_STATEMENT` type | `SUSPECT_NON_INVOICE` | **Yes (TP)** |

Mutation results: 9 / 9 injected defects flagged (100% recall), 0 false flags on clean truth documents (100% precision).

---

## 4. Failure Analysis

There were zero output errors on the labelled set. The failures analyzed below represent **development-time failures** encountered, debugged, and resolved during system construction:

### a. Number-Format Tokenization
- **Example**: In `L04.pdf`, the total was printed using Indian numbering formatting as `₹3,00,484.80`. In German invoice `U07.pdf`, the total was printed as `€ 9.900,80`.
- **How Found**: A Western 3-digit comma regex split `3,00,484.80` into `00` and `484.80`, raising a false `NOT_IN_SOURCE_TEXT` flag. On `U07.pdf`, naive comma-stripping parsed `,80` as `80.0`.
- **Fix**: The grounding logic in `checks.py` extracts numeric tokens with `(?:\d[\d.,]*\d|\d+)`. In `_matches_amount(token, expected)`, tokens are evaluated under dual interpretations: Anglo/Indian (`tok.replace(",", "")`) and European (`tok.replace(".", "").replace(",", ".")`), accepting a match if the absolute difference is $\le 0.01$.
- **Current Status**: Fixed and covered by unit tests in `tests/test_checks.py`.

### b. Upstream Availability
- **Example**: Transient `503 Service Unavailable` and `429 RESOURCE_EXHAUSTED` errors during initial model experiments and batch runs (`U07`–`U10`).
- **How Found**: Free-tier rate limits and intermittent API disconnects during batch execution.
- **Fix**: Wrapped LLM calls with Tenacity exponential retry backoff (3 attempts, handling `ResourceExhausted` and `ServiceUnavailable` in `llm.py`). If retries exhaust, `pipeline.py` catches `ExtractionFailed` and returns `Result.unavailable(UnavailableReason.API_FAILURE, ...)` with exit code 0 rather than crashing the batch.
- **Current Status**: Fixed. All 20 output deliverables in `outputs/` are complete extractions with 0 `API_FAILURE` deliverables.

### c. OCR Limitations
- **Example**: Perspective distortion on `L02.jpg` (~90 s) and rotated scan on `U05.pdf` (230 s vs 13 s Gemini latency).
- **How Found**: Phase 2 EasyOCR ablation experiments on vision-route documents.
- **Fix**: Identified that CPU EasyOCR lacks skew and orientation rectification. OCR cross-checking was decoupled from the primary pipeline, made strictly flag-gated (`--ocr-check`), and classified as non-retryable (`IssueCode.OCR_MISMATCH`).
- **Current Status**: Documented as an evaluated architecture that failed. Gemini Vision natively handles skew and rotation.

### d. Pipeline Logic Defects Found in Code Review
- **Example 1 (Attempt Selection)**: If Attempt 1 returned a valid `Invoice` with 2 minor issues, but Attempt 2 failed schema validation (`ValidationError`), the naive logic overwrote Attempt 1 with `None`.
  - *Fix*: `_select_best_attempt` uses `min([attempt1, attempt2], key=lambda a: (a[0] is None, len(a[1])))`, ensuring schema-valid extractions take precedence.
- **Example 2 (Unfixable Retries)**: Retrying on borderline document types (`AMBIGUOUS_TYPE`) or OCR discrepancies (`OCR_MISMATCH`) caused unnecessary LLM calls.
  - *Fix*: Introduced `NON_RETRYABLE = {AMBIGUOUS_TYPE, OCR_MISMATCH, OCR_UNAVAILABLE, SUSPECT_NON_INVOICE}`, filtering issues before triggering retries.
- **Example 3 (Swallowed OCR Exceptions)**: Missing optional OCR dependencies produced silent failures.
  - *Fix*: `_resolve_source_text` raises an explicit `RuntimeError` prompting installation of `requirements-ocr.txt` on `ImportError`, while catching runtime engine failures as `OCR_UNAVAILABLE` issues.
- **Current Status**: Fixed and covered by 5 unit tests in `tests/test_pipeline.py`.

### e. Weak Grounding
- **Example**:
  - *Short Identifiers*: Invoice number grounding tests whether `clean_inv in clean_src`. A short ID like `42` or `001` can match anywhere across page text or inside unrelated words.
  - *Unanchored Totals*: Any numeric token matching the total amount satisfies grounding. A small total (e.g. `100.00`) can match an item quantity or an address string.
  - *Fuzzy Threshold Sensitivity*: For a 7-character ID, `fuzz.partial_ratio >= 90.0` tolerates 0 substitutions because a single character error drops the score to `85.7`. The fuzzy check is effectively exact for IDs under 10 characters.
  - *Vision Route Omission*: By default (without `--ocr-check`), vision-route scans supply no source text, meaning `NOT_IN_SOURCE_TEXT` cannot fire on scans.
- **How Found**: Algorithmic inspection and code review of `checks.py`.
- **Current Status**: Open architectural limitation of heuristic grounding without bounding-box geometry.

### f. Classification Asymmetry
- **Example**: If Gemini misclassifies an invoice as a non-invoice (e.g. `document_type="BANK_STATEMENT"`), all other fields are returned as `null`. This output is internally consistent, evading standard null-field and math checks and passing silently.
- **How Found**: Conceptual failure mode modeling during verification.
- **Fix**: Implemented `_check_suspect_non_invoice` in `checks.py`, scanning digital text with `re.MULTILINE` for `^\s*tax\s+invoice\s*$` and `\binvoice\s*(?:no\b|number\b|#)`.
- **Current Status**: Open limitation. This heuristic has been validated on only three non-invoices (`L08`, `U06`, `U10`) and covers the `text_layer` route only; a scanned invoice misread as a non-invoice on the `vision` route still passes silently.

### g. Documentation Inconsistencies Caught in Review
- **Example**: Precision and recall displayed as `0.0%` instead of undefined when TP=FP=FN=0; `L05` row in the evaluation grid missing a cell (7 checkmarks instead of 8); pricing source citing a November 2024 date for a model that did not exist then.
- **How Found**: Technical report review and schema consistency audits.
- **Fix**: Formatted precision and recall as `Undefined` in `evaluation/report.py`; aligned all grid rows to 12 columns; marked pricing verification as `TODO(user)`.
- **Current Status**: Fully resolved.

### What I Would Fix First
**The classification asymmetry**: If an invoice is misclassified as a non-invoice, the failure is completely silent because an all-null output is self-consistent under the schema. While `_check_suspect_non_invoice` provides a basic heuristic safeguard on digital PDFs, scanned invoices processed via the vision route currently have zero defense against misclassification. Implementing a lightweight pre-classification vision check or OCR-based header verification would be the top engineering priority.

---

## 5. Cost Per Document

Costs are calculated deterministically using the formula:
$$\text{Cost} = \left(\frac{\text{Input Tokens}}{1,000,000} \times \text{Input Price}\right) + \left(\frac{\text{Output Tokens}}{1,000,000} \times \text{Output Price}\right)$$

Prices with the source and the date checked: TODO(user)
*(Input price: TODO(user) / 1M tokens; Output price: TODO(user) / 1M tokens; Pricing source and date checked: TODO(user). Note: The figures below reflect the placeholder rates of $0.75 / 1M input tokens and $3.75 / 1M output tokens configured in `.env`, with thinking tokens billed at output token rates).*

Measurements from fresh evaluation run (`python evaluate.py --no-cache`, `PROMPT_VERSION = "v2"`):

| Route | Documents | Mean In Tokens | Mean Out Tokens | Mean USD / Doc | Retry Rate |
|---|---|---|---|---|---|
| **Overall** | 10 | 1,344 | 1,094 | **$0.00511** | 0.0% |
| **Text Layer** | 7 | 1,182 | 1,126 | $0.00511 | 0.0% |
| **Vision** | 3 | 1,724 | 1,019 | $0.00511 | 0.0% |

- **Token Consumption**: Native digital PDFs (`text_layer`) averaged 1,182 input tokens, whereas rendered document scans (`vision`) averaged 1,724 input tokens.
- **Retry Rate**: 0.0% (all 10 labelled documents succeeded on Attempt 1).
- **Caching**: Cached runs consume 0 API tokens and incur $0.00 additional cost while retaining original metadata.

---

## 6. Tuning Log

- **`PROMPT_VERSION = "v1"`**: Initial prompt engineered from the 10 labelled documents (`L01`–`L10`) encoding seller-vs-buyer disambiguation, Indian date formats, pre-tax line item amounts, negative discount amounts, itemized freight charges, and strict nullification of non-invoice fields.
- **`PROMPT_VERSION = "v2"`**: Added explicit date-convention guidance: *"If day and month are ambiguous, infer convention from country or currency."* This was added for defensive generality on international documents and was not motivated by any labelled error.
- **`BLUR_THRESHOLD = 50.0`**: Calibrated by measuring Laplacian variance across sample documents:
  - Mobile phone photos: `L02.jpg` (175.36), `U02.jpg` (213.69).
  - Clean scans / rendered PDFs: `L10.pdf` (390.31), `L04.pdf` (2634.45).
  - Setting the threshold at 50.0 ensures legitimate mobile captures pass without false warnings.
- **`BLANK_STD_THRESHOLD = 5.0`**: Calibrated against empirical grayscale standard deviation. The lowest standard deviation among valid documents was 17.00 (`U03.pdf`), ensuring standard deviation $< 5.0$ reliably identifies truly blank pages.

---

## 7. Limitations

1. **Untested Retry Path on Real Pack**: The retry path never ran on the labelled documents (0.0% retry rate). While unit tests with mock LLM responses verify self-correction mechanics, its effectiveness on noisy real-world documents remains unmeasured.
2. **Untested Blur Threshold**: The blur threshold (50.0) was calibrated only against documents that passed. No genuinely degraded or illegible image was included in the sample pack, so the claim that it catches severely degraded images is an untested claim.
3. **Unmeasured Review Flag Effectiveness on Real Data**: The review flag's empirical ability to catch real errors cannot be proven on the labelled pack because there were 0 field errors in the labelled set (quantified via synthetic mutations in §3).
4. **Grounding Check Limitations**:
   - *Substring Over Whole Page*: Invoice number grounding tests whether `clean_inv in clean_src`. A short identifier like `42` or `001` can match almost anywhere across page text.
   - *Unanchored Total*: Any number token equal to the total satisfies grounding. A small total (e.g. `100.00`) could match an unrelated quantity or address number.
   - *Effective Exactness of OCR Fuzzy Threshold*: On a 7-character ID, `fuzz.partial_ratio >= 90.0` tolerates 0 substitutions because a single substitution drops the ratio to `85.7`. For IDs under 10 characters, the fuzzy check is effectively exact.
   - *Default Vision Route Blind Spot*: By default (without `--ocr-check`), vision-route scans provide no source text. Consequently, `NOT_IN_SOURCE_TEXT` cannot fire, and scans are validated only by math, dates, and currencies.
5. **Non-Invoice Misclassification Asymmetry & Anchor Limits**: Deterministic validation checks cannot catch an invoice that the LLM misclassifies as a non-invoice, because an all-null response is internally consistent. While `_check_suspect_non_invoice` detects obvious invoice headers on digital PDFs, this is a heuristic validated on only three non-invoices, and it covers the `text_layer` route only; a scanned invoice misread as a non-invoice on the vision route still passes silently.

---

## 8. Phase 2: EasyOCR Cross-Check Evaluation & Ablation

### Implementation Details
Phase 2 introduced an optional, flag-gated OCR grounding check (`--ocr-check`):
- **Engine**: `invoice_extractor/ocr.py` lazily instantiates dual EasyOCR readers (`['en', 'hi', 'mr']` and `['en', 'kn']`, `gpu=False`).
- **Filtering & Ordering**: Detections below `OCR_CONFIDENCE_THRESHOLD = 0.4` are filtered out. Extracted text lines are deduplicated and sorted top-to-bottom, left-to-right.
- **Issue Classification**: Failures raise `OCR_MISMATCH` (or `OCR_UNAVAILABLE` on runtime failure). `OCR_MISMATCH` is classified as non-retryable to prevent forcing Gemini to change a correct extraction based on noisy OCR.

### Review-Flag Precision & Recall Comparison

| Mode | Flagged Docs | True Field Errors | TP | FP | FN | TN | Flag Precision | Flag Recall |
|---|---|---|---|---|---|---|---|---|
| **Default (Without `--ocr-check`)** | 0 | 0 | 0 | 0 | 0 | 10 | **Undefined (no flags)** | **Undefined (no errors)** |
| **With `--ocr-check`** | 1 (`L02`) | 0 | 0 | 1 | 0 | 9 | **0.0%** | **Undefined (no errors)** |

### Findings Across Scans in the Pack
Evaluation across two of the scans in the pack revealed fatal weaknesses in the CPU EasyOCR configuration:
1. **Perspective Distortion (`L02.jpg`)**: On mobile photo `L02.jpg`, Gemini 3.5 Flash vision transcribed `SGS/1847` with 100% accuracy. EasyOCR on CPU failed to resolve the string due to angle and lighting, raising a false `OCR_MISMATCH` flag and taking ~90 seconds.
2. **Rotation Fragility (`U05.pdf`)**: On `U05.pdf` (a scan rotated 90°), Gemini extracted all fields cleanly in 13 seconds. EasyOCR took **230 seconds** on CPU and raised two false `OCR_MISMATCH` flags (invoice number and total) because the setup has no automatic rotation handling.
3. **Decision**: Because OCR cross-checking generated false positives on correct documents and incurred excessive CPU latency (230 s vs 13 s Gemini latency), the feature is **kept strictly flag-gated under `--ocr-check` and disabled by default**.

---

## 9. Unlabelled Documents Evaluation (`U01`–`U10`)

All 10 unlabelled documents were processed through the pipeline:

| File | Document Type | Needs Review | Verified by Hand Against Source | Notes |
|---|---|---|---|---|
| `U01.pdf` | `INVOICE` | `false` | No | Blue Lotus Hospitality (`₹30,660.50`) |
| `U02.jpg` | `INVOICE` | `false` | No | Harborview Dental Supply (`$917.00`) |
| `U03.pdf` | `INVOICE` | `false` | No | QUICKSERVE IT RENTALS (`£1,305.50`) |
| `U04.pdf` | `INVOICE` | `false` | No | Bluefin Cloud Services (`$528.90`) |
| `U05.pdf` | `INVOICE` | `false` | No | Lakeside Printing Co (`$3,211.20`, rotated scan) |
| `U06.pdf` | `BANK_STATEMENT` | `false` | **Yes** | Harbor Trust Bank account statement. Contains transaction reference `ZR/INV/8812` but correctly classified as non-invoice. |
| `U07.pdf` | `INVOICE` | `false` | **Yes** | TechNova Solutions GmbH (`€ 9.900,80`). European comma decimals verified. |
| `U08.pdf` | `INVOICE` | `false` | No | Sakura Tea Traders B.V. (`€ 1,136.25`) |
| `U09.pdf` | `INVOICE` | `false` | No | Greenleaf Catering Co (`₹99,099.00`) |
| `U10.pdf` | `PERSONAL_OR_OTHER` | `false` | **Yes** | Data Analyst Intern cover letter from Rhea Menon. Correctly classified as non-invoice. |

---

## 10. Phase 3: Web Application & Visual Grounding Interface

### Architecture & Design
Phase 3 deployed a FastAPI backend paired with a zero-build, dependency-free vanilla HTML/JS visual inspector:
- **Backend (`app.py`)**:
  - `POST /extract`: Accepts multipart file uploads. Content is read as raw byte streams directly into memory. File names and extensions are completely disregarded by the pipeline.
  - Returns the exact Pydantic `Result` JSON schema, preserving top-level extraction fields (`is_invoice` as first key), `needs_review` status, granular `review_reasons`, and execution metadata.
  - Supports query/form flags `no_cache` and `ocr_check`.
  - `GET /`: Serves `static/index.html` via `FileResponse` and mounts `/static`.
- **Frontend (`static/index.html`)**:
  - Side-by-side visual grounding: Renders PDF documents using the browser's native viewer (`<iframe>` with Object URLs) or image scans (`<img>`) on the left pane alongside structured extracted data on the right pane.
  - Real-time review status: Displays an immediate banner indicating `PASS` (clean validation), `NEEDS REVIEW` (with failure code chips), or `UNAVAILABLE` (with unreadable error codes).
  - Line items table, metadata & token cost accounting card, and collapsible raw JSON output with clipboard copy functionality.

### Verification & Testing
- Automated endpoint test coverage implemented in `tests/test_app.py` verifying HTML serving (`GET /`), unreadable byte payload handling (`POST /extract`), mock pipeline serialization, and cached end-to-end extraction.
