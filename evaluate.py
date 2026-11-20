"""Evaluation CLI harness: runs extraction against data/labels and outputs tables."""

import argparse
import json
import logging
import sys
from pathlib import Path

from evaluation.report import build_evaluation_report, evaluate_document
from invoice_extractor import cache
from invoice_extractor.pipeline import process

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("evaluate")


def find_document_for_stem(stem: str, doc_dir: Path) -> Path | None:
    """Locate matching document file by stem regardless of file extension."""
    candidates = list(doc_dir.glob(f"{stem}.*"))
    return candidates[0] if candidates else None


def main() -> int:
    """Execute evaluation harness across all labelled files and report findings."""
    parser = argparse.ArgumentParser(description="Evaluate pipeline against ground truth labels.")
    parser.add_argument("--labels-dir", type=str, default="data/labels", help="Labels folder")
    parser.add_argument("--docs-dir", type=str, default="data/documents", help="Documents folder")
    parser.add_argument(
        "--report-out",
        type=str,
        default="reports/eval_latest.json",
        help="JSON report destination",
    )
    parser.add_argument("--no-cache", action="store_true", help="Bypass response disk caching")
    args = parser.parse_args()

    if args.no_cache:
        cache.disable()

    labels_dir = Path(args.labels_dir)
    docs_dir = Path(args.docs_dir)
    if not labels_dir.exists() or not docs_dir.exists():
        logger.error("Labels (%s) or Documents (%s) dir missing", labels_dir, docs_dir)
        return 1

    label_files = sorted(labels_dir.glob("*.json"))
    if not label_files:
        logger.error("No label JSON files found in %s", labels_dir)
        return 1

    pairs: list[tuple[Path, Path]] = []
    missing_docs: list[str] = []
    for lf in label_files:
        stem = lf.stem
        doc_path = find_document_for_stem(stem, docs_dir)
        if not doc_path:
            missing_docs.append(stem)
        else:
            pairs.append((lf, doc_path))

    if missing_docs:
        logger.error("Missing document files for stems: %s", missing_docs)
        return 1

    records: list[dict] = []
    logger.info("Evaluating %d labelled document pairs...", len(pairs))
    for lf, doc_path in pairs:
        logger.info("Evaluating %s (%s)", lf.stem, doc_path.name)
        expected = json.loads(lf.read_text(encoding="utf-8"))
        data = doc_path.read_bytes()
        res = process(data)
        predicted = res.model_dump()
        eval_result = evaluate_document(predicted, expected)
        records.append(
            {
                "stem": lf.stem,
                "doc_name": doc_path.name,
                "predicted": predicted,
                "expected": expected,
                "eval": eval_result,
            }
        )

    report_path = Path(args.report_out)
    tables_str = build_evaluation_report(records, report_path)
    sys.stdout.write(tables_str + "\n")
    logger.info("Evaluation complete. Detailed report saved to %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
