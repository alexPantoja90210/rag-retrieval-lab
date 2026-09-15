"""The invariants that do not depend on which embedding model is used.

These run anywhere, with no model download and no network, because they use the
hashing backend. They are what makes it honest to hand this over from an
environment where the real model cannot be fetched: the plumbing is verified
here, and only embedding quality is deferred.

    python test_invariants.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import common
from evaluate import load_questions, retrieval_metrics, sweep

BACKEND = "hashing"
FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "ok  " if condition else "FAIL"
    print(f"  {status} {name}{('  ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def main() -> int:
    from ingest import load_chunks

    tmp = Path(tempfile.mkdtemp(prefix="raglab-"))
    try:
        chunks = load_chunks(common.DATA_PATH)
        ids = [c.metadata["chunk_id"] for c in chunks]

        print("ids")
        check("chunk ids are unique", len(set(ids)) == len(ids), f"{len(set(ids))}/{len(ids)}")
        check(
            "chunk ids are deterministic across a reload",
            ids == [c.metadata["chunk_id"] for c in load_chunks(common.DATA_PATH)],
        )
        check(
            "an edited chunk gets a different id",
            common.chunk_id("a.pdf", 0, 0, "x") != common.chunk_id("a.pdf", 0, 0, "x "),
        )

        print("citations")
        check(
            "every chunk carries source and page",
            all(c.metadata.get("source") and c.metadata.get("page", -1) >= 0 for c in chunks),
        )

        print("idempotency")
        store = common.get_store(BACKEND, chroma_path=tmp / "store")
        store.add_documents(documents=chunks, ids=ids)
        first = store._collection.count()
        store.add_documents(documents=chunks, ids=ids)
        second = store._collection.count()
        check("first ingest populates the store", first == len(chunks), f"{first} rows")
        check("a second identical ingest adds nothing", second == first, f"{first} -> {second}")

        print("scores")
        hits = store.similarity_search_with_score("scaled dot-product attention", k=5)
        sims = [common.similarity_from_distance(d) for _, d in hits]
        check("k results returned", len(hits) == 5, f"{len(hits)}")
        check("similarities are within [0, 1]", all(-1e-6 <= s <= 1 + 1e-6 for s in sims),
              f"{min(sims):.3f}..{max(sims):.3f}")
        check("results are ordered best first", sims == sorted(sims, reverse=True))
        check(
            "retrieved passages still carry a page",
            all(d.metadata.get("page", -1) >= 0 for d, _ in hits),
        )

        print("metrics arithmetic")
        rows = [
            {"id": "a", "question": "", "pages": [2],
             "results": [{"page": 9, "score": 0.9}, {"page": 2, "score": 0.5}]},
            {"id": "b", "question": "", "pages": [4],
             "results": [{"page": 4, "score": 0.8}]},
            {"id": "n", "question": "", "pages": [],
             "results": [{"page": 7, "score": 0.6}]},
        ]
        m = retrieval_metrics(rows, k=5)
        check("hit@k counts only answerable rows", m["answerable"] == 2)
        check("hit@k = 1.0 when both are found", abs(m["hit@5"] - 1.0) < 1e-9, f"{m['hit@5']}")
        check("mrr = (1/2 + 1/1) / 2 = 0.75", abs(m["mrr"] - 0.75) < 1e-9, f"{m['mrr']:.3f}")

        # Worked by hand from the fixture above, which is the point of a fixture.
        #   a: wants p2; sees p9 at 0.90 and p2 at 0.50
        #   b: wants p4; sees p4 at 0.80
        #   n: wants nothing; sees p7 at 0.60
        s = {r["threshold"]: r for r in sweep(rows, 5, [0.0, 0.55, 0.65, 0.85])}
        check("t=0.00  keeps everything, declines nothing",
              s[0.0]["answerable_still_found"] == 1.0
              and s[0.0]["unanswerable_correctly_declined"] == 0.0)
        check("t=0.55  drops a's correct passage but still answers n",
              s[0.55]["answerable_still_found"] == 0.5
              and s[0.55]["unanswerable_correctly_declined"] == 0.0,
              f"{s[0.55]['answerable_still_found']}/{s[0.55]['unanswerable_correctly_declined']}")
        check("t=0.65  declines n at no further cost, so it is the balanced optimum",
              s[0.65]["answerable_still_found"] == 0.5
              and s[0.65]["unanswerable_correctly_declined"] == 1.0
              and s[0.65]["balanced"] == 0.75,
              f"balanced {s[0.65]['balanced']}")
        check("t=0.85  has traded all remaining recall away",
              s[0.85]["answerable_still_found"] == 0.0,
              f"{s[0.85]['answerable_still_found']}")
        check("the sweep's optimum is not at either extreme",
              max(s.values(), key=lambda r: r["balanced"])["threshold"] == 0.65)

        print("question set")
        qs = load_questions(Path("eval/attention-paper.questions.jsonl"))
        check("the set contains unanswerable questions",
              any(not q["pages"] for q in qs),
              f"{sum(1 for q in qs if not q['pages'])} of {len(qs)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {', '.join(FAILURES)}")
        return 1
    print("all invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
