"""CLI for extracting invoice data from single files or directories."""

import argparse
import json
import logging
import sys
from pathlib import Path

from invoice_extractor import cache
from invoice_extractor.pipeline import process

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("extract")


def _process_file(file_path: Path) -> dict:
    """Read a single file as raw bytes, process it, and return its dictionary representation."""
    data = file_path.read_bytes()
    result = process(data)
    return result.model_dump()


def main() -> int:
    """Parse command line arguments and execute batch or single file extraction."""
    parser = argparse.ArgumentParser(
        description="Extract structured invoice data from files or directories."
    )
    parser.add_argument("path", type=str, help="Path to input document file or directory")
    parser.add_argument("--out", type=str, default=None, help="Output directory for batch mode")
    parser.add_argument("--no-cache", action="store_true", help="Bypass response disk caching")
    args = parser.parse_args()

    if args.no_cache:
        cache.disable()

    target = Path(args.path)
    if not target.exists():
        logger.error("Target path does not exist: %s", target)
        return 1

    if target.is_file():
        res_dict = _process_file(target)
        sys.stdout.write(json.dumps(res_dict, indent=2) + "\n")
        return 0

    if target.is_dir():
        if not args.out:
            logger.error("--out directory must be specified when processing a directory")
            return 1
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted([p for p in target.iterdir() if p.is_file()])
        logger.info("Found %d files to process in %s", len(files), target)
        for p in files:
            logger.info("Processing %s", p.name)
            res_dict = _process_file(p)
            out_file = out_dir / f"{p.stem}.json"
            out_file.write_text(json.dumps(res_dict, indent=2), encoding="utf-8")
        logger.info("Completed extraction for %d files into %s", len(files), out_dir)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
