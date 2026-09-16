"""Score the retriever against a question set. This is the part that can fail.

None of the five RAG repositories audited in IA-125 had anything like this.
Without it, every tuning decision (chunk size, k, threshold, which embedding
model) is taste, and "it seems to work" is the only available verdict.

Two things this measures that a single accuracy number would hide:

  1. Whether the right passage is retrieved at all, at page granularity:
     hit@k, recall@k and MRR over questions the corpus can answer.

  2. Where the abstention threshold should sit. That cannot be measured from
     answerable questions alone, because with only answerable questions the
     best threshold is always zero. The question set therefore contains
     questions the corpus cannot answer, and the sweep reports both sides:
     how much retrieval survives, and how many unanswerable questions still
     come back with something. Choosing the number is then arithmetic rather
     than taste, which is the specific failure recorded against pixegami's
     inherited 0.7.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import common


def load_questions(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        row = json.loads(line)
        for field in ("id", "question", "pages"):
            if field not in row:
                raise ValueError(f"{path}:{line_no} missing field {field!r}")
        rows.append(row)
    return rows


def retrieve_all(questions: list[dict], k: int, backend: str,
                 search: str = "ann") -> list[dict]:
    store = common.get_store(backend)
    out = []
    for q in questions:
        hits = (common.exact_search(store, q["question"], k) if search == "exact"
                else store.similarity_search_with_score(q["question"], k=k))
        results = [
            {
                "page": int(doc.metadata.get("page", -1)) + 1,
                "source": doc.metadata.get("source"),
                "score": common.similarity_from_distance(dist),
            }
            for doc, dist in hits
        ]
        out.append({**q, "results": results})
    return out


def retrieval_metrics(rows: list[dict], k: int) -> dict:
    answerable = [r for r in rows if r["pages"]]
    hits = recall = mrr = 0.0
    for r in answerable:
        expected = set(r["pages"])
        got_pages = [x["page"] for x in r["results"][:k]]
        found = expected & set(got_pages)
        hits += 1.0 if found else 0.0
        recall += len(found) / len(expected)
        for rank, p in enumerate(got_pages, start=1):
            if p in expected:
                mrr += 1.0 / rank
                break
    n = len(answerable) or 1
    return {
        "answerable": len(answerable),
        f"hit@{k}": hits / n,
        f"recall@{k}": recall / n,
        "mrr": mrr / n,
    }


def sweep(rows: list[dict], k: int, steps: list[float]) -> list[dict]:
    answerable = [r for r in rows if r["pages"]]
    unanswerable = [r for r in rows if not r["pages"]]
    table = []
    for t in steps:
        kept_hit = 0
        for r in answerable:
            expected = set(r["pages"])
            kept = [x for x in r["results"][:k] if x["score"] >= t]
            if expected & {x["page"] for x in kept}:
                kept_hit += 1
        false_answers = 0
        for r in unanswerable:
            if any(x["score"] >= t for x in r["results"][:k]):
                false_answers += 1
        a = kept_hit / (len(answerable) or 1)
        b = 1.0 - (false_answers / (len(unanswerable) or 1))
        table.append(
            {
                "threshold": t,
                "answerable_still_found": a,
                "unanswerable_correctly_declined": b,
                "balanced": (a + b) / 2,
            }
        )
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", default="eval/attention-paper.questions.jsonl")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--search", default="ann", choices=common.SEARCH_MODES,
                    help="ann uses Chroma's approximate index; exact is brute "
                         "force over the same stored vectors. check_index.py "
                         "measures the gap. IA-140.")
    ap.add_argument("--backend", default=common.DEFAULT_BACKEND, choices=common.EMBEDDING_BACKENDS)
    ap.add_argument("--sweep", action="store_true", help="print the threshold sweep")
    ap.add_argument("--json-out", default=None, help="write the full result to this path")
    args = ap.parse_args()

    questions = load_questions(Path(args.questions))
    rows = retrieve_all(questions, args.k, args.backend, args.search)
    metrics = retrieval_metrics(rows, args.k)

    n_unans = len([r for r in rows if not r["pages"]])
    print(f"backend        {args.backend}")
    print(f"questions      {len(rows)}  ({metrics['answerable']} answerable, {n_unans} not)")
    print(f"k              {args.k}")
    print()
    for key in (f"hit@{args.k}", f"recall@{args.k}", "mrr"):
        print(f"{key:<14} {metrics[key]:.3f}")

    print("\nper question")
    for r in rows:
        expected = set(r["pages"])
        got = [x["page"] for x in r["results"][: args.k]]
        if not expected:
            top = r["results"][0]["score"] if r["results"] else 0.0
            print(f"  {r['id']:<6} (unanswerable)  top score {top:.3f}")
        else:
            mark = "ok " if expected & set(got) else "MISS"
            print(f"  {r['id']:<6} {mark}  expected {sorted(expected)}  got {got}")

    sweep_table = []
    if args.sweep:
        if not n_unans:
            print("\nno unanswerable questions in the set, so no threshold can be justified")
        else:
            steps = [round(0.05 * i, 2) for i in range(0, 19)]
            sweep_table = sweep(rows, args.k, steps)
            print("\nthreshold sweep")
            print("  thresh  found  declined  balanced")
            for row in sweep_table:
                print(
                    f"  {row['threshold']:<7.2f} {row['answerable_still_found']:<6.2f} "
                    f"{row['unanswerable_correctly_declined']:<9.2f} {row['balanced']:.2f}"
                )
            best = max(sweep_table, key=lambda r: (r["balanced"], r["threshold"]))
            print(
                f"\nbest balanced point: threshold {best['threshold']:.2f} "
                f"({best['balanced']:.2f})"
            )
            print(
                "That number is only valid for this corpus, this question set, this "
                "embedding backend and hnsw:space=cosine. Copying it anywhere else "
                "reproduces the defect it was written to avoid."
            )

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {"backend": args.backend, "k": args.k, "metrics": metrics,
                 "rows": rows, "sweep": sweep_table},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
