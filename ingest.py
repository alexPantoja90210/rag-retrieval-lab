"""Build the vector store from the PDFs in data/.

Differences from the reference, all of them deliberate:

  * ids are derived from content, so a second run changes nothing
  * source and page survive into the store, so a result can be cited
  * the run reports the count before and after, so "changed nothing" is
    an observation rather than a claim
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import common


def load_chunks(data_path: Path) -> list:
    if not data_path.is_dir():
        sys.exit(f"no such directory: {data_path}. See README, section 'The corpus'.")

    pdfs = sorted(data_path.glob("*.pdf"))
    if not pdfs:
        sys.exit(
            f"{data_path} contains no PDF. This repository deliberately ships no "
            "corpus. See README, section 'The corpus'."
        )

    raw = PyPDFDirectoryLoader(str(data_path)).load()
    if not raw:
        sys.exit(f"{len(pdfs)} PDF(s) found but no text extracted. Scanned images need OCR.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=common.CHUNK_SIZE,
        chunk_overlap=common.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(raw)

    # Number the chunks within their page so the id is stable across runs even
    # when the same text appears twice on a page.
    seen: dict[tuple[str, int], int] = {}
    for c in chunks:
        source = Path(c.metadata.get("source", "unknown")).name
        page = int(c.metadata.get("page", -1))
        index = seen.get((source, page), 0)
        seen[(source, page)] = index + 1
        c.metadata["source"] = source
        c.metadata["page"] = page
        c.metadata["chunk_index"] = index
        c.metadata["chunk_id"] = common.chunk_id(source, page, index, c.page_content)
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default=common.DEFAULT_BACKEND, choices=common.EMBEDDING_BACKENDS)
    ap.add_argument("--data", default=str(common.DATA_PATH))
    args = ap.parse_args()

    chunks = load_chunks(Path(args.data))
    store = common.get_store(args.backend)

    before = store._collection.count()
    store.add_documents(documents=chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    after = store._collection.count()

    sources = sorted({c.metadata["source"] for c in chunks})
    print(f"backend        {args.backend}")
    print(f"documents      {len(sources)}  ({', '.join(sources)})")
    print(f"chunks built   {len(chunks)}")
    print(f"store before   {before}")
    print(f"store after    {after}")
    print(f"net added      {after - before}")
    if before and after == before:
        print("\nidempotent: a re-ingest of the same corpus added nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
