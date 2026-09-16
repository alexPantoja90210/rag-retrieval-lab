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

