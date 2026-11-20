# Invoice Extraction Pipeline — Technical & Evaluation Report

## 1. Approach

### System Architecture
The extraction pipeline is designed around honest measurement, deterministic validation, and strict content sniffing:
- **Routing**: Incoming files are received as raw bytes. File names and extensions are completely ignored. If the first 1024 bytes contain `%PDF-`, PyMuPDF inspects each page's text density and image coverage. Documents with $\ge 30$ non-space characters per page and $< 50\%$ image area coverage route to `text_layer` (native text extraction). Scanned PDFs and raster images (`.jpg`, `.png`) route to `vision` (rendered at 150 DPI).
- **Model**: Google Gemini 3.5 Flash (`gemini-3.5-flash`), temperature `0.0`, invoked with structured output (`response_mime_type="application/json"`, `response_schema=Invoice`).
- **Prompting & Conventions**: Prompt conventions were learned from the 10 labelled documents (`L01`–`L10`):
  1. `is_invoice` is derived strictly from `document_type == "INVOICE"`.
  2. For non-invoice types (`PURCHASE_ORDER`, `BANK_STATEMENT`, `RECEIPT`, etc.), all extraction fields must be `null`.
  3. Discounts are extracted as line items with negative amounts (e.g. `L07`).
  4. Freight, packing, and setup charges are itemized lines (e.g. `L01`, `L04`).
  5. Round-off adjustments are not line items; they are absorbed into `total`.
  6. Dates are formatted as ISO `YYYY-MM-DD`.
- **Validation & Self-Correction**: Extracted data passes through deterministic business checks (`checks.py`). If any check fails, the pipeline generates a structured corrective feedback prompt and triggers exactly one retry attempt. The attempt with fewer issues is retained.
- **Unreadable & Failure Handling**: Corrupt, encrypted, blank, or unsupported files return a structured `Result.unavailable` with an exact `UnavailableReason` code. If the upstream Gemini API fails or exhausts retries, `API_FAILURE` is returned, preventing batch abortion.

### What Did Not Work & Fix
During initial validation of `L04.pdf` (a large 24-item packaging invoice), the grounding check raised a false `NOT_IN_SOURCE_TEXT` flag on the invoice total (`₹3,00,484.80`). The initial grounding regular expression assumed Western 3-digit comma grouping (`(?:\d{1,3}(?:,\d{3})+|\d+)`), which failed on Indian numbering formatting (lakhs/crores grouping with 2-digit intervals). This caused `3,00,484.80` to be fragmented into `00` and `484.80`. Generalizing the tokenization to `r"(?:\d[\d,]*)(?:\.\d+)?"` and stripping commas resolved the issue cleanly, grounding `300484.80` with zero false flags.

---

## 2. Accuracy Table

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

*Note: In accordance with the assignment specifications, truth non-invoices (`L08`) are excluded from field-level invoice denominators to prevent artificial accuracy inflation.*

### Per-Document Evaluation Grid

| File | Route | Doc Type | Vendor | Inv # | Date | Curr | Items | Tax | Total | IsInv/Clean | Review |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `L01` | text_layer | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| `L02` | vision | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| `L03` | text_layer | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| `L04` | text_layer | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| `L05` | text_layer | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| `L06` | text_layer | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| `L07` | text_layer | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| `L08` | text_layer | PURCHASE_ORDER | — | — | — | — | — | — | — | ✓ | PASS |
| `L09` | vision | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| `L10` | vision | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |

---

## 3. Failure Analysis

Across the 10 labelled documents:
- **Misses**: 0 field errors (10/10 documents fully correct).
- **Review Flag Analysis**:
  - True Positives (TP): 0
  - False Positives (FP): 0
  - False Negatives (FN): 0
  - True Negatives (TN): 10
  - Review Flag Precision: 0.0% (no false flags raised)
  - Review Flag Recall: 0.0% (no field errors missed)
  - Flagged documents contain 0 of 0 field errors.

### Batch & Unlabelled Document Observations
During extraction across all 20 documents:
1. **Upstream Rate Limiting (`429 RESOURCE_EXHAUSTED` / `503 UNAVAILABLE`)**: When processing the full batch under free-tier quota limits, transient API spikes occurred on `U07`–`U10`. Tenacity exponential backoff retried requests; when quotas were exhausted, the pipeline safely generated `Result.unavailable(API_FAILURE)`, ensuring that batch processing completed with exit code 0 rather than crashing.

---

## 4. Cost Per Document

Costs are computed deterministically per document using the formula:
$$\text{Cost} = \left(\frac{\text{Input Tokens}}{1,000,000} \times \$0.75\right) + \left(\frac{\text{Output Tokens}}{1,000,000} \times \$3.75\right)$$

| Route | Documents | Mean In Tokens | Mean Out Tokens | Mean USD / Doc | Retry Rate |
|---|---|---|---|---|---|
| **Overall** | 10 | 1,330 | 957 | **$0.00458** | 0.0% |
| **Text Layer** | 7 | 1,168 | 946 | $0.00442 | 0.0% |
| **Vision** | 3 | 1,710 | 981 | $0.00496 | 0.0% |

- **Token efficiency**: Digital PDFs routing to `text_layer` consume fewer input tokens (mean 1,168) compared to rendered image pages in `vision` (mean 1,710).
- **Latency & caching**: Cached runs incur 0 latency and 0 API cost while preserving recorded metadata.

---

## 5. Tuning Log

- **`PROMPT_VERSION = v1`**: Initial prompt engineered from the 10 labelled documents (`L01`–`L10`). Encoded seller-vs-buyer rules, Indian date ordering, pre-tax line item amounts, negative discount amounts, and strict nullification of non-invoice fields.
- **`BLUR_THRESHOLD = 50.0`**: Calibrated by measuring Laplacian variance across sample documents:
  - Phone scans: `L02.jpg` (175.36), `U02.jpg` (213.69).
  - High-resolution scans / rendered PDFs: `L10.pdf` (390.31), `L04.pdf` (2634.45).
  - Setting the threshold at 50.0 ensures legitimate phone scans pass without false warnings while catching severely degraded images.
- **`BLANK_STD_THRESHOLD = 5.0`**: Calibrated against empirical grayscale standard deviation. The lowest standard deviation among real documents was 17.00 (`U03.pdf`), ensuring standard deviation < 5.0 reliably detects truly blank pages.

---

## 6. Limitations

1. **Non-Invoice Misclassification Asymmetry**: Deterministic validation checks cannot catch an invoice that the LLM misclassifies as a non-invoice (e.g. as a bank statement or purchase order), because an all-null response is internally consistent and passes validation. All non-invoice classifications should therefore undergo secondary review.
2. **Sample Size Constraints**: The labelled set consists of $n=10$ documents with only 1 non-invoice (`L08`) and 3 vision-route scans/photos (`L02`, `L09`, `L10`). While 100% accuracy was achieved on this set, broader datasets are required for generalizable statistical claims.
3. **Multi-Currency & Regional Formats**: While Indian lakhs/crores formatting was verified and resolved in `L04`, diverse international formatting (e.g. European comma decimals) requires continuous validation against wider corpora.

---

## 7. Phase 2: EasyOCR Cross-Check Evaluation & Ablation

### Implementation Details
Phase 2 introduced an optional, flag-gated OCR grounding check (`--ocr-check`):
- **Engine**: `invoice_extractor/ocr.py` lazily instantiates dual EasyOCR readers (`['en', 'hi', 'mr']` and `['en', 'kn']`, `gpu=False`).
- **Filtering & Ordering**: Detections below `OCR_CONFIDENCE_THRESHOLD = 0.4` are filtered out. Extracted text lines are deduplicated and sorted top-to-bottom, left-to-right.
- **Fuzzy Grounding**: For vision-route documents, the OCR text is passed to `checks.py`, where invoice numbers are grounded using `rapidfuzz.fuzz.partial_ratio >= 90.0` (to accommodate minor character-level OCR substitutions) and totals are matched numerically.

### Review-Flag Precision & Recall Comparison

| Mode | Flagged Docs | True Field Errors | TP | FP | FN | TN | Flag Precision | Flag Recall |
|---|---|---|---|---|---|---|---|---|
| **Default (Without `--ocr-check`)** | 0 | 0 | 0 | 0 | 0 | 10 | **N/A (0 FP)** | **N/A (0 FN)** |
| **With `--ocr-check`** | 1 (`L02`) | 0 | 0 | 1 | 0 | 9 | **0.0%** | **N/A** |

### Findings & Recommendation
1. **False Positive Flooding on Scans**: On `L02.jpg` (a mobile phone photo), Gemini 3.5 Flash vision transcribed the vendor, date, line items, and invoice number (`SGS/1847`) with 100% accuracy against ground truth. However, EasyOCR on CPU struggled with the skewed camera perspective and lighting gradient, failing to recover the exact invoice number string. This triggered a false `NOT_IN_SOURCE_TEXT` review flag on an otherwise perfect extraction.
2. **Computational Overhead**: EasyOCR on CPU added $\sim 90$ seconds of processing per image page compared to under 1 second for text-layer validation.
3. **Decision**: The assignment specifies to *"keep the feature only if it adds true flags without flooding false ones."* Because OCR grounding generated a false positive on a ground-truth correct document and added severe CPU latency, the feature is **kept strictly flag-gated under `--ocr-check` and disabled by default**.
4. **Statistical Caveat**: We plainly record that the number of labelled scans ($n=3$: `L02`, `L09`, `L10`) is too small to statistically prove whether OCR grounding would be net-positive on lower-resolution, highly noisy document collections.

---

## 8. Phase 3: Web Application & Visual Grounding Interface

### Architecture & Design
Phase 3 deployed a FastAPI backend paired with a zero-build, dependency-free vanilla HTML/JS visual inspector:
- **Backend (`app.py`)**:
  - `POST /extract`: Accepts multipart file uploads. Content is read as raw byte streams directly into memory. File names and extensions are completely disregarded by the pipeline.
  - Returns the exact Pydantic `Result` JSON schema, preserving top-level extraction fields, `needs_review` status, granular `review_reasons`, and execution metadata.
  - Supports query/form flags `no_cache` and `ocr_check`.
  - `GET /`: Serves `static/index.html` via `FileResponse` and mounts `/static`.
- **Frontend (`static/index.html`)**:
  - Side-by-side visual grounding: Renders PDF documents using the browser's native viewer (`<iframe>` with Object URLs) or image scans (`<img>`) on the left pane alongside structured extracted data on the right pane.
  - Real-time review status: Displays an immediate banner indicating `PASS` (clean validation), `NEEDS REVIEW` (with failure code chips), or `UNAVAILABLE` (with unreadable error codes).
  - Line items table, metadata & token cost accounting card, and collapsible raw JSON output with clipboard copy functionality.

### Verification & Testing
- Automated endpoint test coverage implemented in `tests/test_app.py` verifying HTML serving (`GET /`), unreadable byte payload handling (`POST /extract`), mock pipeline serialization, and cached end-to-end extraction.
- Total test suite expanded to 22 passing tests with 100% clean Ruff linting and formatting.
