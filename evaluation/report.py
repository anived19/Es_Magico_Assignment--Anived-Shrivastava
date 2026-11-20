"""Evaluation reporting, metric tables formatting, and JSON artifact generation."""

import json
from pathlib import Path

from tabulate import tabulate

from evaluation.matching import (
    is_non_invoice_correct,
    match_invoice_number,
    match_line_items,
    match_number,
    match_vendor,
)


def evaluate_document(predicted: dict, expected: dict) -> dict:
    """Evaluate predicted extraction against ground truth for one document."""
    is_truth_invoice = expected.get("is_invoice") is True

    is_inv_match = predicted.get("is_invoice") == expected.get("is_invoice")
    if not is_truth_invoice:
        non_inv_ok = is_non_invoice_correct(predicted)
        return {
            "is_truth_invoice": False,
            "is_invoice": is_inv_match,
            "non_invoice_correct": non_inv_ok,
            "all_fields_correct": is_inv_match and non_inv_ok,
            "field_scores": {"is_invoice": 1.0 if is_inv_match else 0.0},
            "misses": [] if (is_inv_match and non_inv_ok) else ["Non-invoice not handled cleanly"],
        }

    vendor_ok = match_vendor(predicted.get("vendor"), expected.get("vendor"))
    inv_num_ok = match_invoice_number(
        predicted.get("invoice_number"), expected.get("invoice_number")
    )
    date_ok = predicted.get("invoice_date") == expected.get("invoice_date")
    curr_ok = predicted.get("currency") == expected.get("currency")
    tax_ok = match_number(predicted.get("tax_total"), expected.get("tax_total"))
    tot_ok = match_number(predicted.get("total"), expected.get("total"))
    li_f1 = match_line_items(predicted.get("line_items"), expected.get("line_items"))
    li_ok = li_f1 >= 0.999

    field_scores = {
        "vendor": 1.0 if vendor_ok else 0.0,
        "invoice_number": 1.0 if inv_num_ok else 0.0,
        "invoice_date": 1.0 if date_ok else 0.0,
        "currency": 1.0 if curr_ok else 0.0,
        "line_items": li_f1,
        "tax_total": 1.0 if tax_ok else 0.0,
        "total": 1.0 if tot_ok else 0.0,
        "is_invoice": 1.0 if is_inv_match else 0.0,
    }

    misses = []
    if not is_inv_match:
        misses.append(
            f"is_invoice expected {expected.get('is_invoice')}, got {predicted.get('is_invoice')}"
        )
    if not vendor_ok:
        misses.append(
            f"vendor expected '{expected.get('vendor')}', got '{predicted.get('vendor')}'"
        )
    if not inv_num_ok:
        exp_num = expected.get("invoice_number")
        got_num = predicted.get("invoice_number")
        misses.append(f"invoice_number expected '{exp_num}', got '{got_num}'")
    if not date_ok:
        exp_date = expected.get("invoice_date")
        got_date = predicted.get("invoice_date")
        misses.append(f"invoice_date expected '{exp_date}', got '{got_date}'")
    if not curr_ok:
        misses.append(
            f"currency expected '{expected.get('currency')}', got '{predicted.get('currency')}'"
        )
    if not li_ok:
        misses.append(f"line_items F1 score={li_f1:.2f}")
    if not tax_ok:
        misses.append(
            f"tax_total expected {expected.get('tax_total')}, got {predicted.get('tax_total')}"
        )
    if not tot_ok:
        misses.append(f"total expected {expected.get('total')}, got {predicted.get('total')}")

    all_ok = (
        is_inv_match
        and vendor_ok
        and inv_num_ok
        and date_ok
        and curr_ok
        and li_ok
        and tax_ok
        and tot_ok
    )
    return {
        "is_truth_invoice": True,
        "is_invoice": is_inv_match,
        "vendor": vendor_ok,
        "invoice_number": inv_num_ok,
        "invoice_date": date_ok,
        "currency": curr_ok,
        "line_items": li_ok,
        "line_items_f1": li_f1,
        "tax_total": tax_ok,
        "total": tot_ok,
        "all_fields_correct": all_ok,
        "field_scores": field_scores,
        "misses": misses,
    }


def build_evaluation_report(records: list[dict], output_json_path: Path | None = None) -> str:
    """Compute aggregate accuracy metrics, print markdown tables, and export detailed JSON."""
    total_docs = len(records)
    inv_docs = [r for r in records if r["eval"]["is_truth_invoice"]]
    non_inv_docs = [r for r in records if not r["eval"]["is_truth_invoice"]]
    n_inv = len(inv_docs)
    n_non_inv = len(non_inv_docs)

    # 1. Field-level accuracy table
    field_rows = []
    fields = [
        "vendor",
        "invoice_number",
        "invoice_date",
        "currency",
        "line_items",
        "tax_total",
        "total",
    ]
    for f in fields:
        if f == "line_items":
            mean_f1 = sum(r["eval"]["line_items_f1"] for r in inv_docs) / n_inv if n_inv else 0.0
            field_rows.append(
                [
                    "line_items (mean F1)",
                    f"{mean_f1 * 100:.1f}%",
                    f"{sum(r['eval']['line_items'] for r in inv_docs)}/{n_inv} exact",
                ]
            )
        else:
            correct_cnt = sum(1 for r in inv_docs if r["eval"].get(f))
            pct = (correct_cnt / n_inv * 100) if n_inv else 0.0
            field_rows.append([f, f"{pct:.1f}%", f"{correct_cnt}/{n_inv}"])

    all_corr_cnt = sum(1 for r in inv_docs if r["eval"]["all_fields_correct"])
    all_corr_pct = (all_corr_cnt / n_inv * 100) if n_inv else 0.0
    field_rows.append(
        ["all fields correct (invoices)", f"{all_corr_pct:.1f}%", f"{all_corr_cnt}/{n_inv}"]
    )

    is_inv_cnt = sum(1 for r in records if r["eval"]["is_invoice"])
    is_inv_pct = (is_inv_cnt / total_docs * 100) if total_docs else 0.0
    field_rows.append(
        ["is_invoice (all documents)", f"{is_inv_pct:.1f}%", f"{is_inv_cnt}/{total_docs}"]
    )

    non_inv_corr_cnt = sum(1 for r in non_inv_docs if r["eval"].get("non_invoice_correct"))
    non_inv_pct = (non_inv_corr_cnt / n_non_inv * 100) if n_non_inv else 0.0
    field_rows.append(
        ["non-invoices handled correctly", f"{non_inv_pct:.1f}%", f"{non_inv_corr_cnt}/{n_non_inv}"]
    )

    table1 = tabulate(
        field_rows, headers=["Field / Metric", "Accuracy", "Correct / Total"], tablefmt="github"
    )

    # 2. Per-document grid table
    grid_rows = []
    for r in records:
        ev = r["eval"]
        pred = r["predicted"]
        meta = pred.get("meta", {})
        row = [r["stem"], meta.get("route", "-"), pred.get("document_type", "-")]
        if ev["is_truth_invoice"]:
            for f in [
                "vendor",
                "invoice_number",
                "invoice_date",
                "currency",
                "line_items",
                "tax_total",
                "total",
            ]:
                row.append("✓" if ev.get(f) else "✗")
            row.append("✓" if ev["is_invoice"] else "✗")
        else:
            row.extend(["-"] * 7)
            row.append("✓" if ev.get("non_invoice_correct") else "✗")
        row.append("FLAGGED" if pred.get("needs_review") else "PASS")
        grid_rows.append(row)

    headers2 = [
        "File",
        "Route",
        "Doc Type",
        "Vendor",
        "Inv #",
        "Date",
        "Curr",
        "Items",
        "Tax",
        "Total",
        "IsInv/Clean",
        "Review",
    ]
    table2 = tabulate(grid_rows, headers=headers2, tablefmt="github")

    # 3. Review flag analysis
    tp = fp = fn = tn = 0
    total_field_errors = 0
    flagged_field_errors = 0

    for r in records:
        ev = r["eval"]
        is_wrong = not ev["all_fields_correct"]
        is_flagged = bool(r["predicted"].get("needs_review"))
        doc_errors = len(ev.get("misses", []))
        total_field_errors += doc_errors

        if is_flagged and is_wrong:
            tp += 1
            flagged_field_errors += doc_errors
        elif is_flagged and not is_wrong:
            fp += 1
        elif not is_flagged and is_wrong:
            fn += 1
        else:
            tn += 1

    recall = (tp / (tp + fn) * 100) if (tp + fn) else 0.0
    precision = (tp / (tp + fp) * 100) if (tp + fp) else 0.0

    review_matrix = [
        ["Flagged for review", f"TP = {tp}", f"FP = {fp}", tp + fp],
        ["Not flagged (passed)", f"FN = {fn}", f"TN = {tn}", fn + tn],
        ["Total", tp + fn, fp + tn, total_docs],
    ]
    table3 = tabulate(
        review_matrix,
        headers=["", "Wrong (Errors > 0)", "Right (0 Errors)", "Total"],
        tablefmt="github",
    )
    review_summary = (
        f"Review Flag Precision: {precision:.1f}%\n"
        f"Review Flag Recall:    {recall:.1f}%\n"
        f"Flagged documents contain {flagged_field_errors} of {total_field_errors} field errors.\n"
        "Note: Checks cannot catch an invoice misclassified as non-invoice; "
        f"n={total_docs} is too small for statistical claims."
    )

    # 4. Cost and performance summary
    costs = [r["predicted"].get("meta", {}).get("cost_usd", 0.0) for r in records]
    in_tokens = [r["predicted"].get("meta", {}).get("input_tokens", 0) for r in records]
    out_tokens = [r["predicted"].get("meta", {}).get("output_tokens", 0) for r in records]
    retries = [1 for r in records if r["predicted"].get("meta", {}).get("attempts", 1) > 1]

    text_records = [
        r for r in records if r["predicted"].get("meta", {}).get("route") == "text_layer"
    ]
    vis_records = [r for r in records if r["predicted"].get("meta", {}).get("route") == "vision"]

    cost_rows = [
        [
            "Overall",
            f"{sum(in_tokens) / total_docs:.0f}",
            f"{sum(out_tokens) / total_docs:.0f}",
            f"${sum(costs) / total_docs:.5f}",
            f"{len(retries) / total_docs * 100:.1f}%",
        ],
    ]
    if text_records:
        t_in = sum(r["predicted"]["meta"]["input_tokens"] for r in text_records) / len(text_records)
        t_out = sum(r["predicted"]["meta"]["output_tokens"] for r in text_records) / len(
            text_records
        )
        t_cost = sum(r["predicted"]["meta"]["cost_usd"] for r in text_records) / len(text_records)
        cost_rows.append(["Text Layer", f"{t_in:.0f}", f"{t_out:.0f}", f"${t_cost:.5f}", "-"])
    if vis_records:
        v_in = sum(r["predicted"]["meta"]["input_tokens"] for r in vis_records) / len(vis_records)
        v_out = sum(r["predicted"]["meta"]["output_tokens"] for r in vis_records) / len(vis_records)
        v_cost = sum(r["predicted"]["meta"]["cost_usd"] for r in vis_records) / len(vis_records)
        cost_rows.append(["Vision", f"{v_in:.0f}", f"{v_out:.0f}", f"${v_cost:.5f}", "-"])

    table4 = tabulate(
        cost_rows,
        headers=["Route", "Mean In Tokens", "Mean Out Tokens", "Mean USD / Doc", "Retry Rate"],
        tablefmt="github",
    )

    full_output = (
        f"### 1. Field-Level Accuracy (n={n_inv} labelled invoices, {total_docs} total)\n\n"
        f"{table1}\n\n"
        f"### 2. Per-Document Evaluation Grid\n\n"
        f"{table2}\n\n"
        f"### 3. Review-Flag Analysis\n\n"
        f"{table3}\n\n"
        f"{review_summary}\n\n"
        f"### 4. Cost and Performance Summary\n\n"
        f"{table4}\n"
    )

    if output_json_path:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        export_data = {
            "summary": {
                "total_documents": total_docs,
                "labelled_invoices": n_inv,
                "labelled_non_invoices": n_non_inv,
                "precision": precision,
                "recall": recall,
                "field_errors_total": total_field_errors,
                "field_errors_flagged": flagged_field_errors,
            },
            "documents": [
                {
                    "file": r["stem"],
                    "route": r["predicted"].get("meta", {}).get("route"),
                    "needs_review": r["predicted"].get("needs_review"),
                    "review_reasons": r["predicted"].get("review_reasons"),
                    "all_fields_correct": r["eval"]["all_fields_correct"],
                    "misses": r["eval"]["misses"],
                    "predicted": r["predicted"],
                    "expected": r["expected"],
                }
                for r in records
            ],
        }
        output_json_path.write_text(json.dumps(export_data, indent=2), encoding="utf-8")

    return full_output
