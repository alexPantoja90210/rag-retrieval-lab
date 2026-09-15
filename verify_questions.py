"""Check the ground truth against the PDF itself.

A question set is a claim about where an answer lives. Like any other claim in
this project it is verified against the artifact, not asserted. This extracts
the text of every page named in the question set and confirms the recorded
probe string is actually on it.

Run this whenever the corpus changes. A question set that has quietly stopped
matching its corpus turns every metric downstream into a number about nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import common


def main() -> int:
    qpath = Path(sys.argv[1] if len(sys.argv) > 1 else "eval/attention-paper.questions.jsonl")
    pdfs = sorted(common.DATA_PATH.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"{common.DATA_PATH} contains no PDF. See README, section 'The corpus'.")
    if len(pdfs) > 1:
        sys.exit(f"expected exactly one PDF for this question set, found {len(pdfs)}")

    # The same reader ingest.py uses, so this verifies the text that is
    # actually indexed rather than a second, possibly different, extraction.
    docs = common.load_pdf_pages(common.DATA_PATH)
    pages = {d.metadata["page"] + 1: " ".join(d.page_content.split()) for d in docs}
    print(f"corpus   {pdfs[0].name}  ({len(pages)} pages with extractable text)")

    failures = 0
    checked = 0
    for line in qpath.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        row = json.loads(line)
        if not row["pages"]:
            continue
        probe = row.get("probe")
        if not probe:
            print(f"  {row['id']}  NO PROBE, unverifiable")
            failures += 1
            continue
        checked += 1
        for page in row["pages"]:
            if page not in pages:
                print(f"  {row['id']}  FAIL  page {page} has no extractable text")
                failures += 1
            elif probe.lower() not in pages[page].lower():
                print(f"  {row['id']}  FAIL  {probe!r} not on page {page}")
                failures += 1

    print(f"verified {checked} answerable questions, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
