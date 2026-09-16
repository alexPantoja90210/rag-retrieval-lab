"""Load ObliQA: 13,732 passages of ADGM financial regulation, with ground truth.

Not committed and never redistributed. ObliQA has no LICENSE, so it gets the
same treatment as the reference PDF: used locally, cited, absent from git.
Put a clone of RegNLP/ObliQADataset under data/obliqa/. See README.

Why this is a separate module rather than another branch inside ingest.py.
The corpus is the one thing that changes when the experiment moves, and
keeping each corpus's quirks in its own file means a reader can see exactly
what was done to this one without reading around the quirks of the other.

WHAT THIS MODULE DOES NOT DO, deliberately: it does not set a `page`. The
retrieval path currently assumes ground truth is an int page, and pretending a
regulation clause id is a page number would make that assumption survive by
disguise. Emitting an honest field that nothing reads yet is what makes the
next step visible instead of silently wrong.

Four things were found in the data before any of this was written, and each
one is handled here rather than discovered later as a strange metric:

1. (DocumentID, PassageID) IS NOT UNIQUE. 17 keys carry different text under
   the same key; document 17's "Part 12.Chapter 1.134." has seven. Ground
   truth that points at a non-unique key does not identify a passage. Every
   record does carry a uuid `ID` that is unique across all 13,732, and the
   question files include each supporting passage's TEXT, so the triple
   (DocumentID, PassageID, text) resolves all 17. Identity here is the uuid;
   the ground-truth key is resolved to it through the text.

2. 5% OF PASSAGES EXCEED THE MODEL WINDOW. all-MiniLM-L6-v2 stops at 256
   word-pieces and truncates the rest in silence. 687 passages are longer.
   Among the passages dev's ground truth points at, 147 of 1862, and 260 of
   2788 questions have every supporting passage in that set. Decision on
   record: truncate and record it. Splitting breaks the unit the ground truth
   names; excluding changes the question population. Truncation degrades both
   retrieval criteria equally, so it cancels in a comparison instead of
   biasing it. See `over_window` below for what "record it" means here.

3. 720 PASSAGES ARE EMPTY and 2,203 are under 40 characters, mostly headings
   like "INTRODUCTION". They are dropped or kept by argument, never silently,
   and the loader reports how many of each it saw.

4. ZERO DANGLING REFERENCES. Every ground-truth pointer in dev resolves to a
   passage that exists. Checked, not assumed, and re-checked by
   `resolve_ground_truth` on every run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

from langchain_core.documents import Document

import common

# all-MiniLM-L6-v2 stops at 256 word-pieces. The exact boundary is the
# tokenizer's and the tokenizer only runs on a machine that can reach
# huggingface, so this is a character proxy using the same 3.5 chars/token rule
# as common.estimate_input_tokens, which errs high on purpose.
#
# Erring high here means erring toward flagging MORE passages as at risk. A
# flag that misses a truncated passage is worse than one that warns about a
# passage that fits, because the first hides a ceiling and the second only
# widens the subset you report separately.
MODEL_WINDOW_TOKENS = 256
WINDOW_CHARS = int(MODEL_WINDOW_TOKENS * 3.5)  # 896

MIN_CHARS = 40  # below this a passage is a heading, not content


class CorpusStats(NamedTuple):
    files: int
    records: int
    emitted: int
    empty: int
    too_short: int
    over_window: int
    duplicate_keys: int


def load_passages(root: Path, drop_short: bool = True
                  ) -> tuple[list[Document], CorpusStats]:
    """One Document per passage. Full text kept; nothing is cut here.

    The model truncates the VECTOR at embed time whether we like it or not.
    Cutting the stored text as well would degrade what gets displayed and cited
    for no gain, so the text stays whole and `over_window` records that its
    vector will not represent all of it. That is the difference between
    truncating and knowing you truncated.
    """
    docs_dir = root / "StructuredRegulatoryDocuments"
    if not docs_dir.is_dir():
        raise SystemExit(
            f"no such directory: {docs_dir}. Clone RegNLP/ObliQADataset into "
            f"{root} . This repository ships no corpus. See README."
        )

    files = sorted(docs_dir.glob("*.json"), key=lambda p: int(p.stem)
                   if p.stem.isdigit() else 0)
    out: list[Document] = []
    seen_keys: dict[tuple[int, str], int] = {}
    n_records = n_empty = n_short = n_over = 0

    for f in files:
        for r in json.loads(f.read_text(encoding="utf-8")):
            n_records += 1
            text = (r.get("Passage") or "").strip()
            doc_id = r["DocumentID"]
            passage_id = r["PassageID"]
            key = (doc_id, passage_id)
            seen_keys[key] = seen_keys.get(key, 0) + 1

            if not text:
                n_empty += 1
                continue
            if drop_short and len(text) < MIN_CHARS:
                n_short += 1
                continue

            over = len(text) > WINDOW_CHARS
            n_over += over
            out.append(Document(
                page_content=text,
                metadata={
                    "source": f"obliqa-doc-{doc_id}",
                    "document_id": doc_id,
                    # The regulation's own clause id. Not unique on its own.
                    "passage_id": passage_id,
                    # The uuid. This IS unique across all 13,732 and is what
                    # identity means in this corpus.
                    "uid": r["ID"],
                    "chars": len(text),
                    # Its vector will not represent the whole passage. Kept as
                    # a field so recall can be reported with and without this
                    # subset, instead of the ceiling being invisible.
                    "over_window": over,
                    "chunk_index": 0,
                    "chunk_id": common.chunk_id(
                        f"obliqa-doc-{doc_id}", passage_id, 0, text),
                },
            ))

    stats = CorpusStats(
        files=len(files), records=n_records, emitted=len(out), empty=n_empty,
        too_short=n_short, over_window=n_over,
        duplicate_keys=sum(1 for v in seen_keys.values() if v > 1),
    )
    return out, stats


def load_questions(root: Path, split: str = "dev") -> list[dict]:
    path = root / f"ObliQA_{split}.json"
    if not path.is_file():
        raise SystemExit(f"no such file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_ground_truth(questions: list[dict], passages: list[Document]
                         ) -> tuple[list[dict], list[str]]:
    """Turn each question's (DocumentID, PassageID) pointers into uids.

    This is where finding 1 is paid for. The key alone is ambiguous for 17
    passages, so the passage TEXT that the question file carries is used to
    pick which record was meant. Falling back to the key when the text does not
    match would quietly reintroduce the ambiguity, so it does not: an
    unresolvable pointer is returned as a problem, not guessed at.

    Returns (rows, problems). A non-empty problems list means the ground truth
    and the corpus have drifted apart and no metric computed on them means
    anything, which is the same rule verify_questions.py enforces on the PDF.
    """
    by_key: dict[tuple[int, str], list[Document]] = {}
    for d in passages:
        by_key.setdefault(
            (d.metadata["document_id"], d.metadata["passage_id"]), []).append(d)

    truncated = {d.metadata["uid"]: bool(d.metadata["over_window"])
                 for d in passages}

    rows, problems = [], []
    for q in questions:
        uids = []
        for p in q["Passages"]:
            key = (p["DocumentID"], p["PassageID"])
            cands = by_key.get(key, [])
            if not cands:
                problems.append(
                    f"{q['QuestionID']}: {key} is not in the corpus")
                continue
            if len(cands) == 1:
                hit = cands[0]
            else:
                want = (p.get("Passage") or "").strip()
                exact = [c for c in cands if c.page_content == want]
                if len(exact) != 1:
                    problems.append(
                        f"{q['QuestionID']}: {key} matches {len(cands)} "
                        f"passages and the text picks {len(exact)} of them")
                    continue
                hit = exact[0]
            uids.append(hit.metadata["uid"])
        rows.append({
            "id": q["QuestionID"],
            "question": q["Question"],
            "uids": uids,
            # True when EVERY supporting passage is truncated. These are the
            # questions whose recall ceiling is set by the model window rather
            # than by the embedding, and reporting them separately is the
            # whole reason `over_window` exists.
            "all_support_truncated": bool(uids) and all(
                truncated[u] for u in uids),
        })
    return rows, problems


def report(stats: CorpusStats) -> str:
    kept = stats.emitted
    return "\n".join([
        f"files                 {stats.files}",
        f"records read          {stats.records}",
        f"passages emitted      {kept}",
        f"  dropped, empty      {stats.empty}",
        f"  dropped, < {MIN_CHARS} chars {stats.too_short}",
        f"over the model window {stats.over_window} of {kept} "
        f"({100 * stats.over_window / kept:.1f}%)  vector truncated, text kept",
        f"ambiguous (doc,passage) keys  {stats.duplicate_keys}  "
        f"resolved by text against the question file",
    ])


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1] if len(sys.argv) > 1 else common.DATA_PATH / "obliqa")
    passages, stats = load_passages(root)
    print(report(stats))
    qs = load_questions(root, "dev")
    rows, problems = resolve_ground_truth(qs, passages)
    print(f"\nquestions (dev)       {len(rows)}")
    resolved = sum(1 for r in rows if r["uids"])
    print(f"  ground truth resolved to uids   {resolved}")
    print(f"  unresolvable pointers           {len(problems)}")
    for p in problems[:10]:
        print(f"    {p}")
    ceiling = sum(1 for r in rows if r["all_support_truncated"])
    print(f"  every supporting passage truncated  {ceiling} "
          f"({100 * ceiling / len(rows):.1f}%)   <- a recall ceiling the "
          f"embedding cannot be blamed for")
