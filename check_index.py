"""How much of a retrieval miss is the index rather than the embedding?

Chroma's HNSW is approximate. It can return a different shortlist from the one
brute force would, and when it does, recall@k reports the sum of two errors and
attributes all of it to the embedding. Those need different fixes: a worse
embedding is a modelling problem, a missed neighbour is a parameter problem.

This measures the gap directly by running both searches over the SAME stored
vectors. Nothing is re-embedded, so any disagreement is the index and only the
index.

    python check_index.py --backend minilm -k 5

Run it whenever the corpus grows. At 48 chunks with ef_search=100 the search
parameter exceeds the corpus and the index is exact in practice, so the answer
is boring and that is the point: it establishes that the numbers recorded at
this size owe nothing to the index. At 13,732 passages it will not be boring,
and the decision about whether to keep an approximate index should be made on
this number rather than on habit. IA-140.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import common
import generate
from evaluate import load_questions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", default="eval/attention-paper.questions.jsonl")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--backend", default=common.DEFAULT_BACKEND,
                    choices=common.EMBEDDING_BACKENDS)
    args = ap.parse_args()

    qs = load_questions(Path(args.questions))
    store = common.get_store(args.backend)
    n_vectors = store._collection.count()

    identical = 0
    rank1_same = 0
    worst_gap = 0.0
    recall_ann = recall_exact = 0
    disagreements = []

    for q in qs:
        ann = generate.retrieve(q["question"], args.k, args.backend, None, "ann")
        exact = generate.retrieve(q["question"], args.k, args.backend, None, "exact")
        ia = [p.chunk_id for p in ann]
        ie = [p.chunk_id for p in exact]
        if ia == ie:
            identical += 1
        else:
            disagreements.append((q["id"], [p.page for p in ann], [p.page for p in exact],
                                  len(set(ie) - set(ia))))
        if ia[:1] == ie[:1]:
            rank1_same += 1
        if ann and exact:
            worst_gap = max(worst_gap,
                            max(abs(x.score - y.score) for x, y in zip(ann, exact)))
        if q["pages"]:
            want = set(q["pages"])
            recall_ann += bool(want & {p.page for p in ann})
            recall_exact += bool(want & {p.page for p in exact})

    n_ans = len([q for q in qs if q["pages"]])
    print(f"backend            {args.backend}")
    print(f"vectors in store   {n_vectors}")
    print(f"queries            {len(qs)}   k={args.k}")
    print()
    print(f"identical top-{args.k} lists   {identical}/{len(qs)}")
    print(f"same rank-1 result     {rank1_same}/{len(qs)}")
    print(f"largest score gap      {worst_gap:.3e}")
    print(f"hit@{args.k}  approximate {recall_ann}/{n_ans}   exact {recall_exact}/{n_ans}")

    if disagreements:
        print("\nwhere they differ")
        for qid, pa, pe, missed in disagreements:
            print(f"  {qid}  ann {pa}  exact {pe}   the index missed {missed} "
                  f"neighbour(s) brute force found")
        print("\nThe hit@k gap between the two columns above is index error. Any "
              "conclusion\nabout the embedding drawn from the approximate column "
              "carries it.")
    else:
        print("\nNo disagreement. At this size the approximate index is exact in "
              "practice,\nso nothing measured on it owes anything to the index. "
              "Re-run when the corpus grows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
