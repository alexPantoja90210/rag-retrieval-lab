"""Score the generation layer, and catch the model answering from memory.

The reference this was forked from demos its RAG on the most famous paper in
machine learning, and tries to stop the model using what it already knows with a
line in the prompt:

    "you don't use your internal knowledge, but solely the information in
     the 'The knowledge' section"

That is an instruction. Nothing enforces it and nothing detects a breach, so the
demo behaves identically with the retriever unplugged.

This turns it into a measurement. Every answerable question is asked twice:

  grounded   the passages retrieval actually returned
  sabotaged  passages deliberately taken from unrelated pages

Declining under sabotage is correct. Answering CORRECTLY under sabotage means
the answer came from the model's memory, not from the corpus, and that question
proves nothing about the retrieval.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import common
import generate
from evaluate import load_questions


def evidence_strings(row: dict) -> list[str]:
    """Text that would let a reader answer without prior knowledge.

    Defaults to the probe, which verify_questions.py has already confirmed sits
    on the expected page. `answer_evidence` extends it where the answer also
    appears in another form elsewhere in the corpus, such as a symbol in a
    table. That is not hypothetical: q14's answer is "label smoothing", the
    phrase appears only on page 8, and page 9 carries a Table 3 column headed
    with the symbol for it. Excluding page 8 did not exclude the evidence.
    """
    out = [row["probe"]] if row.get("probe") else []
    out += row.get("answer_evidence", [])
    if row.get("answer_contains") and len(row["answer_contains"]) >= 5:
        out.append(row["answer_contains"])
    return [e for e in dict.fromkeys(out) if e]


def is_pure_sabotage(passages: list, row: dict) -> bool:
    """No expected page, no answer chunk, and no answer evidence in the text.

    Page exclusion alone was what the first run used, and it let a fragment of
    q14's answer through. Purity is now a property that gets checked rather
    than a property the construction is assumed to have. IA-144.
    """
    if not passages:
        return False
    if set(row["pages"]) & {ps.page for ps in passages}:
        return False
    if set(row.get("_answer_chunk_ids", ())) & {ps.chunk_id for ps in passages}:
        return False
    hay = " ".join(" ".join(ps.text.split()) for ps in passages).lower()
    return not any(e.lower() in hay for e in evidence_strings(row))


def sabotage_passages(rows: list[dict], i: int) -> list:
    """A donor question's passages, verified pure for this question.

    Deterministic: walks forward from a fixed offset and takes the first donor
    that passes every purity check, so a rerun compares like with like.
    """
    n = len(rows)
    for step in range(1, n):
        donor = rows[(i + 6 + step) % n]
        if is_pure_sabotage(donor["_passages"], rows[i]):
            return donor["_passages"]
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", default="eval/attention-paper.questions.jsonl")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--backend", default=common.DEFAULT_BACKEND,
                    choices=common.EMBEDDING_BACKENDS)
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--max-usd", type=float, default=0.25,
                    help="worst-case ceiling for the WHOLE run; it refuses to "
                         "start rather than stopping halfway")
    ap.add_argument("--max-tokens", type=int, default=common.MAX_OUTPUT_TOKENS)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--prompt", default="grounded", choices=list(generate.PROMPTS),
                    help="which system prompt to use for this run (IA-145)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="run every arm N times. Temperature 0 lowers variance "
                         "The SDK exposes no temperature setting, so variance "
                         "is measured rather than suppressed. IA-148.")
    ap.add_argument("--all-arms", action="store_true",
                    help="run every prompt arm and tabulate. One variable moves: "
                         "the system prompt. Same model, retrieval, questions and "
                         "sabotage passages throughout.")
    args = ap.parse_args()

    rows = load_questions(Path(args.questions))

    # Chunk-level ground truth, derived rather than hand-written: the chunks
    # whose text actually contains the probe. verify_questions.py has already
    # confirmed the probe is on the expected page, so this cannot drift from
    # the corpus without that check failing first.
    from ingest import load_chunks
    chunks = load_chunks(common.DATA_PATH)
    for r in rows:
        pr = (r.get("probe") or "").lower()
        # Restricted to the expected page as well as the probe. Without that
        # restriction q03's probe, "identical layers", matches a chunk on page 2
        # as well as the expected page 3, and a run that retrieved page 2 was
        # being classified as a generation failure when the answer chunk had
        # never been supplied at all. Caught while testing this very fix.
        r["_answer_chunk_ids"] = [
            c.metadata["chunk_id"] for c in chunks
            if pr and pr in c.page_content.lower()
            and (c.metadata["page"] + 1) in r["pages"]
        ] if r["pages"] else []

    for r in rows:
        r["_passages"] = generate.retrieve(r["question"], args.k, args.backend,
                                           args.threshold)

    answerable = [r for r in rows if r["pages"]]
    scorable = [r for r in answerable if r.get("answer_contains")]

    # --- control 2: price the whole run before sending anything -------------
    calls, est_tokens = [], []
    for i, r in enumerate(rows):
        if r["_passages"]:
            calls.append(("grounded", i))
            est_tokens.append(common.estimate_input_tokens(
                generate.SYSTEM + generate.build_prompt(r["question"], r["_passages"])))
    for i, r in enumerate(rows):
        if r["pages"]:
            sab = sabotage_passages(rows, i)
            if sab:
                calls.append(("sabotaged", i))
                est_tokens.append(common.estimate_input_tokens(
                    generate.SYSTEM + generate.build_prompt(r["question"], sab)))
    worst_tokens = max(est_tokens) if est_tokens else 0

    impure = [r["id"] for i, r in enumerate(rows)
              if r["pages"] and not is_pure_sabotage(sabotage_passages(rows, i), r)]
    n_ans_ch = sum(1 for r in rows if r["pages"] and r["_answer_chunk_ids"])

    print(f"model        {args.model}"
          f"{'  repeats=' + str(args.repeats) if args.repeats > 1 else ''}")
    print(f"retrieval    {args.backend}, k={args.k}, threshold="
          f"{args.threshold if args.threshold is not None else 'none'}")
    print(f"questions    {len(rows)}  ({len(answerable)} answerable, "
          f"{len(scorable)} of those auto-scorable)")
    n_arms_p = (len(generate.PROMPTS) if args.all_arms else 1) * args.repeats
    n_arm_kinds = len(generate.PROMPTS) if args.all_arms else 1
    if n_arm_kinds > 1:
        print(f"arms         {n_arm_kinds}  ({', '.join(generate.PROMPTS)})")
    print(f"calls        {len(calls) * n_arms_p}  "
          f"({sum(1 for c,_ in calls if c=='grounded') * n_arms_p} grounded, "
          f"{sum(1 for c,_ in calls if c=='sabotaged') * n_arms_p} sabotaged)")
    print(f"ground truth  page level for all, chunk level for {n_ans_ch}/"
          f"{len([r for r in rows if r['pages']])} answerable "
          f"(derived from the probe, not authored)")
    if impure:
        print(f"\nSABOTAGE STOP: no pure sabotage set exists for {', '.join(impure)}. "
              f"Nothing was sent.")
        return 3
    print("sabotage      purity verified for every answerable question: no expected "
          "page,\n              no answer chunk, no answer evidence in the text")
    skipped = len(rows) - sum(1 for c, _ in calls if c == "grounded")
    if skipped:
        print(f"skipped      {skipped} question(s) declined by the retrieval gate, "
              f"costing nothing")
    try:
        n_arms = (len(generate.PROMPTS) if args.all_arms else 1) * args.repeats
        est = common.check_budget(args.model, len(calls) * n_arms, worst_tokens,
                                  args.max_usd, args.max_tokens)
    except common.BudgetExceeded as e:
        print(f"\nBUDGET STOP: {e}")
        return 2
    print(f"budget       worst case ${est:.4f}, limit ${args.max_usd:.4f}  ok\n")

    # --- run ----------------------------------------------------------------
    model = generate.get_model(args.model)
    spent = 0.0
    results = []
    arms = list(generate.PROMPTS) if args.all_arms else [args.prompt]
    for rep in range(args.repeats):
     for arm in arms:
      system = generate.PROMPTS[arm]
      for cond, i in calls:
        r = rows[i]
        passages = r["_passages"] if cond == "grounded" else sabotage_passages(rows, i)
        a = model(r["question"], passages, expect=r.get("answer_contains"),
                  expected_pages=r["pages"], max_tokens=args.max_tokens,
                  system=system)
        spent += a.usd
        correct = None
        if r.get("answer_contains"):
            # Word-boundary match, so "8" does not match "128", and applied to
            # the model's ANSWER only. Never to the passages: a short expected
            # string matches almost any chunk of a technical paper, which is the
            # defect this run caught on its first execution.
            pat = re.compile(r"(?<![\w.])" + re.escape(r["answer_contains"]) + r"(?![\w])",
                             re.IGNORECASE)
            correct = (not a.abstained) and bool(pat.search(a.text))
        results.append({"id": r["id"], "arm": arm, "repeat": rep, "condition": cond,
                        "abstained": a.abstained,
                        "correct": correct, "text": a.text[:300],
                        "pages_shown": [ps.page for ps in passages],
                        "chunks_shown": [ps.chunk_id for ps in passages],
                        "answer_chunk_shown": bool(
                            set(r.get("_answer_chunk_ids", ()))
                            & {ps.chunk_id for ps in passages}),
                        "expected_pages": r["pages"], "usd": a.usd})

    def pick(cond, qid, arm=None):
        for x in results:
            if x["id"] == qid and x["condition"] == cond \
               and (arm is None or x["arm"] == arm):
                return x
        return None

    # --- the numbers, per arm -----------------------------------------------
    print(f"{'arm':<24} {'correct':>8} {'leaked':>8} {'leak rate':>10} "
          f"{'declined(sab)':>14} {'unans declined':>15}")
    summary = {}
    for arm in arms:
        g_ok = [r for r in scorable
                if (pick("grounded", r["id"], arm) or {}).get("correct")]
        # A leak is a CORRECT ANSWER UNDER SABOTAGE. Full stop.
        #
        # The first version required the question to also be correct when
        # grounded, which silently excluded the purest case there is: answering
        # correctly from passages that cannot contain the answer, having
        # declined when shown the passages that can. That is doing worse with
        # the right evidence than with the wrong evidence, and it can only come
        # from memory. The metric was blind to exactly the instance it was
        # built to catch, and the first three-arm run contained one. IA-147.
        per_rep = []
        for rep in range(args.repeats):
            n = sum(1 for r in scorable
                    if any(x["correct"] for x in results
                           if x["id"] == r["id"] and x["arm"] == arm
                           and x["condition"] == "sabotaged" and x["repeat"] == rep))
            per_rep.append(n)
        # The union across repeats, not one sample. pick() returns the first
        # matching row, so with --repeats this column reported repeat 0 and
        # silently dropped the rest. The final three-repeat run leaked once, in
        # repeat 1 of the reference arm, and the headline said zero while the
        # per-repeat detail printed [0, 1, 0] right next to it. Third time this
        # family of bug has appeared: a summary computed on a sample of the
        # thing it claims to summarise. IA-150.
        leaked = [r for r in scorable
                  if any(x["correct"] for x in results
                         if x["id"] == r["id"] and x["arm"] == arm
                         and x["condition"] == "sabotaged")]
        leaked_only_sabotaged = [r for r in leaked if r not in g_ok]
        dec = [r for r in answerable
               if (pick("sabotaged", r["id"], arm) or {}).get("abstained")]
        und = [r for r in rows if not r["pages"]
               and (pick("grounded", r["id"], arm) or {"abstained": False})["abstained"]]
        n_un = len([r for r in rows if not r["pages"]])
        # Denominator is the scorable questions, not the ones it happened to
        # get right when grounded, so the rate cannot be improved by getting
        # more questions wrong.
        rate = len(leaked) / len(scorable) if scorable else 0.0
        summary[arm] = {"correct": len(g_ok), "leaked": len(leaked),
                        "leak_rate": rate, "declined_sabotage": len(dec),
                        "unanswerable_declined": len(und),
                        "leaked_ids": [r["id"] for r in leaked],
                        "leaked_while_failing_when_grounded":
                            [r["id"] for r in leaked_only_sabotaged]}
        # Printed even at one repeat, so a reader can never mistake a single
        # sample for a measurement.
        spread = f"  per repeat {per_rep}"
        print(f"{arm:<24} {len(g_ok):>3}/{len(scorable):<4} {len(leaked):>8} "
              f"{rate:>9.0%} {len(dec):>10}/{len(answerable):<3} "
              f"{len(und):>11}/{n_un}{spread}")
    for arm in arms:
        wrong_chunk = [r["id"] for r in scorable
                       if (pick("grounded", r["id"], arm) or {}).get("correct") is False
                       and not (pick("grounded", r["id"], arm) or {}).get(
                           "answer_chunk_shown", False)]
        had_it = [r["id"] for r in scorable
                  if (pick("grounded", r["id"], arm) or {}).get("correct") is False
                  and (pick("grounded", r["id"], arm) or {}).get(
                      "answer_chunk_shown", False)]
        if wrong_chunk or had_it:
            print(f"\n  {arm} grounded failures, split by cause (IA-144):")
            if wrong_chunk:
                print(f"    retrieval never supplied the answer chunk: "
                      f"{', '.join(wrong_chunk)}")
            if had_it:
                print(f"    the answer chunk WAS supplied and it still failed: "
                      f"{', '.join(had_it)}   <- a generation failure")
    for arm in arms:
        if summary[arm]["leaked_ids"]:
            print(f"\n  {arm}: answered from memory on "
                  + ", ".join(summary[arm]["leaked_ids"]))
        worse = summary[arm]["leaked_while_failing_when_grounded"]
        if worse:
            print(f"    of those, {', '.join(worse)} were answered correctly with "
                  f"the WRONG passages and declined with the right ones,")
            print(f"    which is the strongest form the evidence takes.")

    print("\nNote: 'leaked' means ANSWERED CORRECTLY while shown passages that")
    print("cannot contain the answer. It is measured by the answer_contains check,")
    print("not by the abstention detector, so it is comparable across arms even")
    print("though only two of the three arms have a refusal token at all.")
    print("The 'declined(sab)' column is NOT comparable: the reference arm has no")
    print("way to produce the exact token, so a low number there means the prompt")
    print("offers no refusal, not that the model refused to refuse.")

    print(f"\nspent this run   ${spent:.6f}")
    if args.model.startswith("stub"):
        print("(stub model: no request was sent and nothing was charged)")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"model": args.model, "backend": args.backend, "k": args.k,
             "threshold": args.threshold, "spent_usd": spent,
             "arms": arms, "summary": summary,
             "results": results}, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
