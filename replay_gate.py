"""Re-score a saved evaluation with the shipped citation gate. Costs nothing.

Why this exists as a script rather than a paste. The citation gate was first
checked by replaying the recorded results through a COPY of its logic written
inline for the occasion. That verifies the idea and not the artifact, which is
the distinction this repository is about. This imports `check_citations` from
generate.py, so what gets verified is the code that runs.

It also means any saved run can be re-scored after the gate changes, without
sending anything to an API.

    python replay_gate.py eval/results/experiment-final.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from generate import Passage, check_citations


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1
                else "eval/results/experiment-final.json")
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d["results"]

    rejected = defaultdict(int)
    reasons = defaultdict(int)
    totals = defaultdict(int)
    leaks = []

    for x in rows:
        # Only the page matters to the gate, so a minimal Passage is honest
        # here: reconstructing the text would not change the verdict and
        # pretending to have it would be worse.
        shown = [Passage(d.get("source", "d.pdf"), p, 0.0, "", -1, "")
                 for p in x["pages_shown"]]
        ok, why = check_citations(x["text"], shown)
        totals[x["arm"]] += 1
        if not ok:
            rejected[x["arm"]] += 1
            reasons[why.split(" that")[0]] += 1
        if x["condition"] == "sabotaged" and x.get("correct"):
            leaks.append((x["arm"], x["repeat"], x["id"], ok, why))

    print(f"replaying {path}  ({len(rows)} recorded responses, 0 API calls)\n")
    print(f"{'arm':<26} {'rejected':>12}")
    for arm in d.get("arms", sorted(totals)):
        print(f"{arm:<26} {rejected[arm]:>5}/{totals[arm]:<6}")

    print("\nreasons")
    for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {why}")

    print(f"\nleaks recorded in this run: {len(leaks)}")
    for arm, rep, qid, ok, why in leaks:
        verdict = "PASSED THE GATE" if ok else "rejected by the gate"
        print(f"  {arm} / repeat {rep} / {qid}: {verdict}  ({why})")
    if not leaks:
        print("  none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
