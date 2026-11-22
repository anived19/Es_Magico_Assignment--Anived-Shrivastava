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
  6. Dates are formatted as ISO `YYYY-MM-DD`. If day/month are ambiguous, conventions are inferred from currency or country.
- **Validation & Self-Correction**: Extracted data passes through deterministic business checks (`checks.py`). If fixable issues occur, the pipeline formats a corrective feedback prompt and triggers exactly one retry attempt. The superior attempt (prioritizing successful schema parsing over issue count) is retained. Non-retryable issues (`AMBIGUOUS_TYPE`, `OCR_MISMATCH`, `SUSPECT_NON_INVOICE`) do not trigger retries.
- **Unreadable & Failure Handling**: Corrupt, encrypted, blank, or unsupported files return a structured `Result.unavailable` with an exact `UnavailableReason` code. If the upstream Gemini API fails or exhausts retries, `API_FAILURE` is returned, preventing batch abortion.

### Architecture Trial That Failed: EasyOCR Grounding Cross-Check
We hypothesized that running a local, CPU-based OCR pass on vision-route scans and cross-referencing extracted values would catch vision hallucinations. In practice, this architectural addition failed:
1. **Perspective & Skew Fragility**: On mobile photo `L02.jpg`, Gemini Vision accurately extracted `SGS/1847`, whereas EasyOCR struggled with camera angle and lighting gradients, failing to recover the string.
2. **Rotation Blindness**: On `U05.pdf` (a scan rotated 90°), EasyOCR produced garbage text due to lack of rotation handling, raising two false `OCR_MISMATCH` flags (invoice number and total).
3. **Severe Latency**: On `U05.pdf`, EasyOCR took 230 seconds on CPU compared to 13 seconds for Gemini Vision on the same file.
Consequently, OCR grounding was rejected as a default component and kept strictly flag-gated behind `--ocr-check`.

### Implementation Debugging & Regex Tuning
During development, two number tokenization edge cases were identified and resolved:
- **Indian Lakhs/Crores Grouping**: In `L04.pdf`, the total was printed as `₹3,00,484.80`. A standard 3-digit comma regex fragmented the number into `00` and `484.80`, raising a false grounding flag.
- **European Comma Decimals**: In German invoice `U07.pdf`, the total was printed as `€ 9.900,80`. Comma-stripping logic treated `,80` as `80.0`.
- **Solution in Code**: The codebase tokenizes numbers with `(?:\d[\d.,]*\d|\d+)` and evaluates dual interpretations in `_matches_amount`: Anglo/Indian format (`tok.replace(",", "")`) and European format (`tok.replace(".", "").replace(",", ".")`), grounding both formats cleanly.

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

*Note: As a design decision, to avoid inflating field accuracy by scoring trivial nulls, truth non-invoices (`L08`) are excluded from field-level invoice denominators.*

### Per-Document Evaluation Grid

| File | Route | Doc Type | Vendor | Inv # | Date | Curr | Items | Tax | Total | IsInv/Clean | Review |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `L01` | text_layer | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| `L02` | vision | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| `L03` | text_layer | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| `L04` | text_layer | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
| `L05` | text_layer | INVOICE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PASS |
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
  - Precision and recall are undefined: no document was flagged and no document was wrong. This table cannot yet show whether the flag catches errors.

### Batch & Unlabelled Document Observations
During an initial full batch run under free-tier quota limits, transient `429` rate-limit errors occurred on documents `U07`–`U10`. Following quota reset, all four documents were re-extracted cleanly without cache. All 20 output deliverables in `outputs/` are complete extractions with 0 `API_FAILURE` deliverables.

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

- **`PROMPT_VERSION = "v1"`**: The prompt was engineered upfront from the 10 labelled documents (`L01`–`L10`) to encode seller-vs-buyer rules, Indian date ordering, pre-tax line item amounts, negative discount amounts, and strict nullification of non-invoice fields. Because extraction on the labelled set achieved 100% accuracy on Attempt 1 without prompt alterations, the prompt version remained `v1`.
- **`BLUR_THRESHOLD = 50.0`**: Calibrated by measuring Laplacian variance across sample documents:
  - Phone scans: `L02.jpg` (175.36), `U02.jpg` (213.69).
  - High-resolution scans / rendered PDFs: `L10.pdf` (390.31), `L04.pdf` (2634.45).
  - Setting the threshold at 50.0 ensures legitimate phone scans pass without false warnings.
- **`BLANK_STD_THRESHOLD = 5.0`**: Calibrated against empirical grayscale standard deviation. The lowest standard deviation among real documents was 17.00 (`U03.pdf`), ensuring standard deviation < 5.0 reliably detects truly blank pages.

---

## 6. Limitations

1. **Untested Retry Path on Real Pack**: The retry path never ran on the labelled documents (0.0% retry rate). While unit tests with mock LLM responses verify that self-correction works, its real-world value on noisy document sets remains unmeasured.
2. **Untested Blur Threshold**: The blur threshold (50.0) was calibrated only against documents that passed. No genuinely degraded or unreadable image was included in the sample pack, so the claim that it catches severely degraded images is an untested claim.
3. **Unmeasured Review Flag Effectiveness**: The review flag's empirical ability to catch real errors cannot be proven on this pack because there were 0 field errors in the labelled set.
4. **Grounding Check Limitations**:
   - **Substring Over Whole Page**: Invoice number grounding tests whether `clean_inv in clean_src`. A short identifier like `42` or `001` can match almost anywhere across the page or across word boundaries.
   - **Unanchored Total**: Any number token in the source text equal to the total satisfies grounding. A small total (e.g. `100.00`) could match an unrelated quantity or address number.
   - **Effective Exactness of OCR Fuzzy Threshold**: On a 7-character ID, `fuzz.partial_ratio >= 90.0` tolerates 0 substitutions, because a single character substitution drops the Indel-based ratio to `85.7`. For IDs under 10 characters, the fuzzy check is effectively exact.
   - **Default Vision Route Blind Spot**: By default (without `--ocr-check`), vision-route scans provide no source text. Consequently, `NOT_IN_SOURCE_TEXT` cannot fire, and scans are validated only by math, dates, and currencies.
5. **Non-Invoice Misclassification Asymmetry**: Deterministic validation checks cannot catch an invoice that the LLM misclassifies as a non-invoice (e.g. as a bank statement or purchase order), because an all-null response is internally consistent. We introduced `_check_suspect_non_invoice` to detect strong invoice anchors (`Invoice No:`, standalone `Tax Invoice` headings) on digital PDFs, but vision-route scans remain reliant on model classification.

---

## 7. Phase 2: EasyOCR Cross-Check Evaluation & Ablation

### Implementation Details
Phase 2 introduced an optional, flag-gated OCR grounding check (`--ocr-check`):
- **Engine**: `invoice_extractor/ocr.py` lazily instantiates dual EasyOCR readers (`['en', 'hi', 'mr']` and `['en', 'kn']`, `gpu=False`).
- **Filtering & Ordering**: Detections below `OCR_CONFIDENCE_THRESHOLD = 0.4` are filtered out. Extracted text lines are deduplicated and sorted top-to-bottom, left-to-right.
- **Issue Classification**: Failures raise `OCR_MISMATCH` (or `OCR_UNAVAILABLE` on runtime failure). `OCR_MISMATCH` is classified as non-retryable to prevent forcing Gemini to change a correct extraction based on noisy OCR.

### Review-Flag Precision & Recall Comparison

| Mode | Flagged Docs | True Field Errors | TP | FP | FN | TN | Flag Precision | Flag Recall |
|---|---|---|---|---|---|---|---|---|
| **Default (Without `--ocr-check`)** | 0 | 0 | 0 | 0 | 0 | 10 | **N/A (0 FP)** | **N/A (0 FN)** |
| **With `--ocr-check`** | 1 (`L02`) | 0 | 0 | 1 | 0 | 9 | **0.0%** | **N/A** |

### Findings Across Scans in the Pack
Evaluation across two of the scans in the pack revealed fatal weaknesses in the CPU EasyOCR configuration:
1. **Perspective Distortion (`L02.jpg`)**: On `L02.jpg` (a mobile phone photo), Gemini 3.5 Flash vision transcribed `SGS/1847` with 100% accuracy. EasyOCR on CPU failed to resolve the string due to angle and lighting, raising a false `OCR_MISMATCH` flag and taking ~90 seconds.
2. **Rotation Fragility (`U05.pdf`)**: On `U05.pdf` (a scan rotated 90°), Gemini extracted all fields cleanly in 13 seconds. EasyOCR took **230 seconds** on CPU and raised two false `OCR_MISMATCH` flags (invoice number and total) because the setup has no automatic rotation handling.
3. **Decision**: Because OCR cross-checking generated false positives on correct documents and incurred excessive CPU latency (230s vs 13s Gemini latency), the feature is **kept strictly flag-gated under `--ocr-check` and disabled by default**.

---

## 8. Unlabelled Documents Evaluation (`U01`–`U10`)

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

## 9. Phase 3: Web Application & Visual Grounding Interface

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
