"""Query the store and show what was retrieved, with its source.

This is the half of the reference's chatbot.py that can be checked. There is no
language model here on purpose: retrieval is the part that can be scored
against ground truth without a judge, and it is the part that decides whether
an answer could have been correct at all.

Every result carries its document and page. The reference had that information
and threw it away, and then told the model not to mention where the answer came
from.
"""

from __future__ import annotations

import argparse
import textwrap

import common


def search(query: str, k: int, backend: str, threshold: float | None):
    store = common.get_store(backend)
    hits = store.similarity_search_with_score(query, k=k)
    scored = [(doc, common.similarity_from_distance(dist)) for doc, dist in hits]
    if threshold is not None:
        scored = [(d, s) for d, s in scored if s >= threshold]
    return scored


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query", nargs="+")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--backend", default=common.DEFAULT_BACKEND, choices=common.EMBEDDING_BACKENDS)
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Minimum cosine similarity to return. Default is none. A number "
            "here should come from evaluate.py --sweep, not from taste."
        ),
    )
    args = ap.parse_args()
    query = " ".join(args.query)

    results = search(query, args.k, args.backend, args.threshold)

    print(f"query      {query}")
    print(f"backend    {args.backend}")
    print(f"threshold  {args.threshold if args.threshold is not None else 'none'}")
    print()

    if not results:
        print("No passage met the threshold. Nothing retrieved, so nothing to answer from.")
        return 0

    for rank, (doc, score) in enumerate(results, start=1):
        m = doc.metadata
        print(f"[{rank}] {score:.4f}  {m.get('source')} p.{int(m.get('page', -1)) + 1}")
        body = " ".join(doc.page_content.split())
        print(textwrap.indent(textwrap.fill(body[:400], width=88), "     "))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
