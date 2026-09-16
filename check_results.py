"""Check that a saved run's summary still agrees with its own rows. Costs nothing.

Why this exists. `eval/results/experiment-final.json` is the file the runbook
points at as the evidence for the leak measurement. Its `summary` block said
the reference arm leaked zero times. Its own 324 rows contain a correct answer
under sabotage. The rows were right and the summary was stale: it was written
before IA-150 changed the definition from "repeat 0" to "the union across
repeats", and nothing ever went back to the file.

So the repository shipped a conclusion and, one directory down, the evidence
that contradicted it. A reader who opened the file to check the runbook would
have concluded the runbook was lying.

A results file is two different things stapled together: rows, which are raw
evidence from a run that cost money, and a summary, which is derived. Derived
values go stale when the derivation changes. Rows do not. So the rows are the
truth and this script re-derives from them, using `common.leaked_ids`, the same
function the evaluator calls, rather than a copy written for the occasion.

    python check_results.py                 # every file in eval/results/
    python check_results.py FILE            # one file
    python check_results.py --fix           # rewrite stale summaries, rows untouched

Exit code is 1 when any file disagrees with itself, so this can gate a push.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import common

RESULTS_DIR = Path("eval/results")

# Only these are re-derived. Everything else in a summary block is left exactly
# as the run wrote it, because this script is a corrector of one known drift,
# not a re-summariser that quietly rewrites numbers it never verified.
DERIVED = ("leaked", "leak_rate", "leaked_ids")


def recompute(d: dict, arm: str) -> dict:
    rows = d["results"]
    ids = common.leaked_ids(rows, arm)
    # The denominator the evaluator uses is the scorable questions. It is not
    # in the file, so it is recovered as the questions that were ever marked
    # correct or incorrect under the grounded condition for this arm: exactly
    # the ones the correctness check could score.
    scorable = {x["id"] for x in rows
                if x.get("arm") == arm and x.get("condition") == "grounded"
                and x.get("correct") is not None}
    rate = len(ids) / len(scorable) if scorable else 0.0
    return {"leaked": len(ids), "leak_rate": rate, "leaked_ids": ids}


def check_file(path: Path, fix: bool = False) -> list[str]:
    """Return a list of human-readable disagreements. Empty means consistent."""
    d = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(d, dict) or "summary" not in d or "results" not in d:
        return []
    summary = d["summary"]
    rows = d["results"]
    if not isinstance(summary, dict) or not rows or "arm" not in rows[0]:
        return []

    problems = []
    changed = False
    for arm in d.get("arms", sorted(summary)):
        if arm not in summary:
            continue
        got = recompute(d, arm)
        for key in DERIVED:
            if key not in summary[arm]:
                continue
            stored = summary[arm][key]
            fresh = got[key]
            if key == "leak_rate":
                same = abs(float(stored) - fresh) < 1e-9
            else:
                same = stored == fresh
            if not same:
                problems.append(
                    f"{path.name}  {arm}.{key}: stored {stored!r}, "
                    f"rows say {fresh!r}")
                if fix:
                    summary[arm][key] = fresh
                    changed = True

    if fix and changed:
        d["summary_recomputed"] = {
            "on": _dt.date.today().isoformat(),
            "by": "check_results.py --fix",
            "fields": list(DERIVED),
            "why": ("the stored summary predated IA-150, which redefined a leak "
                    "as the union across repeats rather than repeat 0"),
            "rows": "unchanged; only derived fields in `summary` were rewritten",
        }
        # Written back byte-for-byte the way the evaluator writes it: indent=2,
        # ASCII-escaped, no trailing newline, and the line ending the file
        # already had. Not cosmetics. If this reformatted the file, every one
        # of the 324 rows would show as changed in the diff, and "the rows were
        # untouched" would become a claim you have to take on trust instead of
        # one you can see.
        raw = path.read_bytes()
        nl = "\r\n" if b"\r\n" in raw else "\n"
        with open(path, "w", encoding="utf-8", newline=nl) as fh:
            fh.write(json.dumps(d, indent=2))
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--fix", action="store_true",
                    help="rewrite the stale derived fields. Rows are never touched.")
    args = ap.parse_args()

    files = args.files or sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        print(f"no result files under {RESULTS_DIR}")
        return 0

    all_problems = []
    checked = 0
    for f in files:
        probs = check_file(f, fix=args.fix)
        if probs or (f.name and json.loads(f.read_text(encoding="utf-8")).get("summary")):
            checked += 1
        all_problems.extend(probs)

    print(f"checked {checked} file(s) with a summary block, "
          f"out of {len(files)} under {RESULTS_DIR}\n")
    if not all_problems:
        print("every stored summary agrees with its own rows")
        return 0

    verb = "rewrote" if args.fix else "found"
    print(f"{verb} {len(all_problems)} disagreement(s) between a summary and "
          f"the rows beneath it:\n")
    for p in all_problems:
        print(f"  {p}")
    if args.fix:
        print("\nrows untouched. `summary_recomputed` records what changed and why.")
        return 0
    print("\nthe rows are the evidence. Run with --fix to re-derive the summary.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
