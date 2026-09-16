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


def sabotage_passages(rows: list[dict], i: int) -> list:
    """Passages from a different question, chosen so no expected page overlaps.

    Deterministic, so a rerun compares like with like. Walks forward from a
    fixed offset until it finds a donor whose pages miss the target's entirely.
    """
    want = set(rows[i]["pages"])
    n = len(rows)
    for step in range(1, n):
        donor = rows[(i + 6 + step) % n]
        donor_pages = {p for _, p, _, _ in donor["_passages"]}
        if not (want & donor_pages) and donor["_passages"]:
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
    ap.add_argument("--all-arms", action="store_true",
                    help="run every prompt arm and tabulate. One variable moves: "
                         "the system prompt. Same model, retrieval, questions and "
                         "sabotage passages throughout.")
    args = ap.parse_args()

    rows = load_questions(Path(args.questions))
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

    print(f"model        {args.model}")
    print(f"retrieval    {args.backend}, k={args.k}, threshold="
          f"{args.threshold if args.threshold is not None else 'none'}")
    print(f"questions    {len(rows)}  ({len(answerable)} answerable, "
          f"{len(scorable)} of those auto-scorable)")
    n_arms_p = len(generate.PROMPTS) if args.all_arms else 1
    if n_arms_p > 1:
        print(f"arms         {n_arms_p}  ({', '.join(generate.PROMPTS)})")
    print(f"calls        {len(calls) * n_arms_p}  "
          f"({sum(1 for c,_ in calls if c=='grounded') * n_arms_p} grounded, "
          f"{sum(1 for c,_ in calls if c=='sabotaged') * n_arms_p} sabotaged)")
    skipped = len(rows) - sum(1 for c, _ in calls if c == "grounded")
    if skipped:
        print(f"skipped      {skipped} question(s) declined by the retrieval gate, "
              f"costing nothing")
    try:
        n_arms = len(generate.PROMPTS) if args.all_arms else 1
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
        results.append({"id": r["id"], "arm": arm, "condition": cond,
                        "abstained": a.abstained,
                        "correct": correct, "text": a.text[:300],
                        "pages_shown": [p for _, p, _, _ in passages],
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
        leaked = [r for r in scorable
                  if (pick("sabotaged", r["id"], arm) or {}).get("correct")]
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
        print(f"{arm:<24} {len(g_ok):>3}/{len(scorable):<4} {len(leaked):>8} "
              f"{rate:>9.0%} {len(dec):>10}/{len(answerable):<3} "
              f"{len(und):>11}/{n_un}")
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
