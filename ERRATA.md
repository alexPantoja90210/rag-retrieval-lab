# Errata

Corrections to this repository's published record. Nothing here is edited out of
the history it corrects. A commit that shipped keeps its hash, because the hash
is what the issue tracker records as evidence, and rewriting it to hide a
mistake would damage the thing the mistake was found in.

If you are reading a commit message and it points somewhere that does not exist,
this file is where the real reference lives.

---

## E1. Commit messages cite Jira keys that did not exist

**Raised as IA-151. Discovered 16 Sep 2026 by a routine status query.**

Five commits were written with the issue key I expected the issue to receive,
not the key it received. Two of those keys had never been created at all.

| Commit | Subject | Cites | What it should say |
| --- | --- | --- | --- |
| `0b77e93` | Pin temperature and add repeats | IA-148 | Correct |
| `ae0143f` | Remove a control that does not exist, and check the SDK contract instead | IA-148, IA-149 | IA-148 only. The IA-149 reference is wrong: IA-149 is the citation gate |
| `864f259` | The headline leak column reported one repeat and dropped the rest | IA-147, IA-148, IA-150 | Correct, but IA-150 was created after the commit, not before |
| `defad39` | Citation gate | IA-145, IA-151 | IA-145, **IA-149** |
| `1e2ce96` | Add replay_gate.py | IA-151 | **IA-149** |

IA-150 and IA-151 now both exist. IA-150 is the union-fix defect, which makes
`864f259` resolve correctly. IA-151 is the record of this erratum, which means
the two commits that mis-cite it now lead a reader to the explanation instead of
to nothing. That is the only one of these references the erratum can repair in
place, and it repairs it by occupying the key rather than by creating a
duplicate of IA-149 to make a typo true.

### Why this is worth a file

It is the same defect this repository was built to study, at the smallest
possible scale: an artifact that describes something it does not deliver. The
experiment measured a model asserting facts its passages could not support. A
commit message asserting IA-151 before IA-151 existed is the same shape of
claim, made by the author of the experiment, in the artifact that documents it.

### What would have caught it

Nothing in the workflow checks that a key cited in a commit message exists on
the board. The obvious guard is a pre-push hook that extracts `IA-\d+` from the
outgoing commits and fails on any key the board does not know.

**It is not built.** It is recorded as not built, on IA-151, rather than being
described as future work with no issue behind it. A guard that is only
described is exactly the failure mode named above.

---

## E2. The erratum for E1 was itself incomplete

**Raised on IA-151, reopened. Discovered by the PO pasting the test output.**

E1 mapped five commit messages, the Confluence runbook and the Jira board. It
did not cover the source code, which carries the same wrong key in two places:

| File | Was | Is |
| --- | --- | --- |
| `test_invariants.py` | `the outgoing API call binds against the real SDK (IA-149)` | IA-148 |
| `generate.py` | `it checked for the presence of a word, not for a contract. IA-149.` | IA-148 |

The first of those **prints on screen every time the suite runs**. It was the
most visible copy of the wrong reference in the entire project, and the erratum
that set out to find every copy missed it, because the search was done by
remembering where the key had been written rather than by grepping for it.

An erratum that misses the most visible instance of the thing it corrects is the
same defect one level up: a claim of completeness with no mechanism behind it.
E1's own wording, "the full mapping for anyone who has the repo", was not true
when it was written.

The remaining `IA-149` in `generate.py` is correct: it labels the citation gate,
which is what IA-149 is. `test_invariants.py` now labels that section with the
key too, so the gate and the SDK contract are no longer distinguishable only by
reading the code around them.

### How it was found, and what that says

Not by a check. The PO ran `python test_invariants.py` because the previous
answer asked him to confirm the 63 assertions, and the wrong key was sitting in
line 229 of the output. The pre-push hook named in E1 as not built would not
have caught this either: it validates keys that exist, and IA-149 exists. The
check that would have caught it is a grep of the working tree, which is what E1
should have run and did not.

---

## E3. Two shipped results files contradicted their own rows

**Raised as IA-152. Found while writing the generation results section of the
README, which the README had never had.**

`eval/results/experiment-final.json` is the file the runbook names as the
evidence for the leak measurement. Its `summary` block said the reference arm
leaked zero times. Its own 324 rows contain a correct answer under sabotage:
`reference`, repeat 1, q02. `experiment-arms.json` had the same shape of
disagreement, at q14.

The rows were right. The summary was stale. It was written before IA-150
redefined a leak as the union across repeats rather than repeat 0, and nothing
ever went back to the file. **IA-150 fixed the code and left the artifact.**

So the repository shipped a conclusion and, one directory down, the evidence
that appeared to refute it. Anyone opening the file to check the runbook would
have concluded the runbook was wrong.

### What was changed, and what was not

The 324 rows are **byte-identical** to what the paid run produced. That is not a
promise, it is checkable: compare the `results` array against the previous
commit. Only three derived fields in one arm moved, and a `summary_recomputed`
block records the date, the tool and the reason.

The correction was made by `check_results.py --fix`, which calls
`common.leaked_ids`, the same function the evaluator calls. It is not a second
implementation of the leak definition written for the occasion. A definition
kept in two places is a definition that will drift, and this one already had.

### The mechanism, which is the actual deliverable

`check_results.py` re-derives the summary from the rows and reports every
disagreement. `test_invariants.py` runs it over every shipped results file, and
the assertion **failed on two of them the day it was written**, which is the
only reason to trust it now. A control builds a file whose summary denies its
own rows and requires the checker to catch all three fields, plus the inverse
case so the checker is not simply always angry.

### The pattern, stated plainly

E1 was a commit citing an issue that did not exist. E2 was an erratum that
missed the most visible copy of what it was correcting. E3 is a summary that
outlived the definition it was computed with. All three are the same thing: a
derived artifact that stopped tracking its source and nothing noticed, because
nothing was watching.
