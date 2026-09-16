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


def _raises_not(fn) -> bool:
    """True when fn does not raise. Named so the assertion reads forwards."""
    try:
        fn()
        return True
    except Exception:
        return False


def _raises(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


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

        print("abstention detection")
        # IA-125 recorded a tutorial whose abstention check was `result == "NA"`,
        # exact equality against free-form model output, defeated by "NA.".
        # These are the cases that must not defeat this one.
        for txt in [common.ABSTAIN, common.ABSTAIN + ".", common.ABSTAIN.lower(),
                    "  " + common.ABSTAIN + "  ",
                    common.ABSTAIN + ", the passages cover a different topic."]:
            check(f"declines on {txt[:34]!r}", common.looks_like_abstention(txt))
        for txt in ["The model uses 8 heads [p.5]", "", "There is insufficient time"]:
            check(f"does NOT declare abstention on {txt[:30]!r}",
                  not common.looks_like_abstention(txt))

        print("spend controls")
        # Haiku 4.5 at $1/MTok in, $5/MTok out, priced 16 Sep 2026.
        one = common.worst_case_usd("claude-haiku-4-5", 1, 1_000_000, 1_000_000)
        check("worst case of 1M in + 1M out on Haiku is $6.00", abs(one - 6.0) < 1e-9,
              f"${one:.4f}")
        check("an unpriced model is refused, not guessed",
              _raises(lambda: common.worst_case_usd("gpt-9", 1, 10, 10), ValueError))
        check("the budget stops a run it cannot afford",
              _raises(lambda: common.check_budget("claude-haiku-4-5", 1000, 5000,
                                                  0.0001), common.BudgetExceeded))
        check("the budget allows a run it can afford",
              common.check_budget("claude-haiku-4-5", 1, 1000, 1.0) > 0)
        check("the token estimate errs high (3.5 chars/token, not 4)",
              common.estimate_input_tokens("x" * 3500) > 900)

        print("sabotage construction")
        import evaluate_answers as ea
        from generate import Passage as _P
        fake = [{"id": f"q{i}", "pages": [i], "question": "?", "probe": f"marker{i}",
                 "_answer_chunk_ids": [f"chunk{i}"],
                 "_passages": [_P("d.pdf", i, 0.5, f"body text {i}", 0, f"chunk{i}")]}
                for i in range(1, 8)]
        clean = True
        for i, r in enumerate(fake):
            sab = ea.sabotage_passages(fake, i)
            if not sab or (set(r["pages"]) & {ps.page for ps in sab}):
                clean = False
        check("sabotage passages never include an expected page", clean)
        check("sabotage selection is deterministic",
              ea.sabotage_passages(fake, 0) == ea.sabotage_passages(fake, 0))

        print("model backends accept what the evaluator passes")
        # The bug this exists to prevent: adding a keyword to the call site and
        # to one backend's body, but not to another backend's signature. The
        # stubs are exercised constantly and the API backend never is, because
        # running it needs a key, so a signature drift there survives every
        # test in this file. inspect.bind checks the contract without calling
        # anything and without a key.
        import inspect

        import generate as _g
        call_kwargs = dict(expect="x", expected_pages=[1], max_tokens=10,
                           system="a system prompt")
        for backend in ("stub", "stub-memoriser", "claude-haiku-4-5"):
            m = _g.get_model(backend)
            try:
                inspect.signature(m.__call__).bind("a question", [], **call_kwargs)
                ok = True
            except TypeError:
                ok = False
            check(f"{backend} accepts the evaluator's keywords", ok)

        check("every prompt arm is a non-empty string",
              all(isinstance(v, str) and v.strip() for v in _g.PROMPTS.values()),
              f"{len(_g.PROMPTS)} arms")
        check("the reference arm really has no refusal token",
              common.ABSTAIN not in _g.PROMPTS["reference"])
        check("the plus-abstain arm is the reference plus exactly that",
              _g.PROMPTS["reference-plus-abstain"].startswith(_g.PROMPTS["reference"])
              and common.ABSTAIN in _g.PROMPTS["reference-plus-abstain"])

        print("sabotage purity (IA-144)")
        import evaluate_answers as ea2
        from generate import Passage
        row = {"id": "t", "pages": [8], "probe": "label smoothing",
               "answer_evidence": ["\u03f5ls"], "_answer_chunk_ids": ["deadbeef"]}
        clean = [Passage("d.pdf", 2, .5, "unrelated prose about convolutions", 0, "aaa")]
        onpage = [Passage("d.pdf", 8, .5, "unrelated prose", 0, "bbb")]
        onchunk = [Passage("d.pdf", 2, .5, "unrelated prose", 0, "deadbeef")]
        eviden = [Passage("d.pdf", 2, .5, "Table 3 ... \u03f5ls 0.1 ...", 0, "ccc")]
        probey = [Passage("d.pdf", 2, .5, "we used label smoothing here", 0, "ddd")]
        check("a clean donor is accepted", ea2.is_pure_sabotage(clean, row))
        check("a donor on the expected page is rejected",
              not ea2.is_pure_sabotage(onpage, row))
        check("a donor holding the answer chunk is rejected",
              not ea2.is_pure_sabotage(onchunk, row))
        check("a donor holding a declared evidence symbol is rejected",
              not ea2.is_pure_sabotage(eviden, row), "the q14 contamination")
        check("a donor holding the probe verbatim is rejected",
              not ea2.is_pure_sabotage(probey, row))
        check("an empty donor is rejected", not ea2.is_pure_sabotage([], row))

        import inspect as _i
        print("the outgoing API call binds against the real SDK (IA-148)")
        # The previous version of this section asserted that the string
        # "temperature=temperature" appeared in the source of the call site. It
        # passed while the call was invalid, because anthropic 1.6.0 removed the
        # parameter. Checking for a word is not checking a contract.
        #
        # This binds the exact keywords we send against the SDK's own signature.
        # No call, no network, no key, and it fails on precisely the mistake
        # that got through.
        try:
            import anthropic.resources.messages as _M
            have_sdk = True
        except ImportError:
            have_sdk = False
        if not have_sdk:
            print("  SKIP anthropic is not installed, so the SDK contract is "
                  "UNCHECKED here")
            FAILURES.append("sdk-contract-unchecked")
        else:
            kw = _g.AnthropicModel("claude-haiku-4-5").send_kwargs("p", "sys", 300)
            check("we send max_tokens, the spend cap", "max_tokens" in kw)
            check("we send a system prompt, the arm's independent variable",
                  "system" in kw)
            check("every keyword we send is accepted by Messages.create",
                  _raises_not(lambda: _i.signature(_M.Messages.create).bind(None, **kw)))
            check("the control fails on a keyword the SDK does not have",
                  not _raises_not(lambda: _i.signature(_M.Messages.create).bind(
                      None, **{**kw, "temperature": 0})),
                  "temperature, the one that got through")

        print("variance is measured, not assumed away")
        check("there is no temperature setting to pretend to pin",
              not hasattr(common, "TEMPERATURE"))

        print("citation gate, levels 1 and 2 (IA-149)")
        from generate import Passage as _PG, check_citations as _cc
        shown = [_PG("d.pdf", 6, .5, "t", 0, "a"), _PG("d.pdf", 7, .5, "t", 1, "b")]
        check("an answer citing a supplied page is accepted",
              _cc("8 heads [d.pdf p.6].", shown)[0])
        check("two citations, both supplied, accepted",
              _cc("A [d.pdf p.6] and B [d.pdf p.7].", shown)[0])
        check("an assertion with no citation is REJECTED",
              not _cc("Based on the passages, it dispenses with recurrence.", shown)[0],
              "level 1, the form the real leak took")
        check("a citation to a page never supplied is REJECTED",
              not _cc("The answer is X [d.pdf p.1].", shown)[0], "level 2")
        check("declining needs no citation",
              _cc(common.ABSTAIN + " The passages cover another topic.", shown)[0])
        check("an empty answer is REJECTED rather than waved through",
              not _cc("", shown)[0])
        check("the gate runs on the stubs too, not only the real backend",
              _g.get_model("stub")("q", shown, expect="x",
                                   expected_pages=[6]).citation_reason != "")

        print("passages carry chunk identity")
        ps = Passage("d.pdf", 3, 0.5, "text", 2, "abc123")
        check("a Passage exposes chunk_index and chunk_id",
              ps.chunk_index == 2 and ps.chunk_id == "abc123")

        print("exact vs approximate search (IA-140)")
        qs_ = load_questions(Path("eval/attention-paper.questions.jsonl"))[:6]
        same = 0
        for q_ in qs_:
            a_ = _g.retrieve(q_["question"], 5, BACKEND, None, "ann")
            e_ = _g.retrieve(q_["question"], 5, BACKEND, None, "exact")
            if [p.chunk_id for p in a_] == [p.chunk_id for p in e_]:
                same += 1
        check("at this corpus size the index agrees with brute force",
              same == len(qs_), f"{same}/{len(qs_)} queries")
        e_ = _g.retrieve(qs_[0]["question"], 5, BACKEND, None, "exact")
        check("exact search returns k results", len(e_) == 5)
        check("exact search is ordered best first",
              [p.score for p in e_] == sorted([p.score for p in e_], reverse=True))
        check("exact search carries the same metadata as the index path",
              all(p.chunk_id and p.page >= 0 for p in e_))

        print("a saved summary agrees with its own rows (IA-152)")
        import json as _json
        import check_results as _cr
        # The real files. This assertion failed on two of them the day it was
        # written, which is the only reason to trust it now.
        _stale = []
        for _f in sorted(Path("eval/results").glob("*.json")):
            _stale.extend(_cr.check_file(_f))
        check("no shipped results file contradicts itself",
              not _stale,
              f"{len(_stale)} disagreement(s)" if _stale else "all consistent")

        # The control. A checker that has only ever been shown consistent files
        # cannot be distinguished from one that returns nothing. So: build a
        # file whose rows contain a leak and whose summary denies it, and
        # require the checker to say so.
        _rows = [{"id": "qX", "arm": "a", "condition": "sabotaged",
                  "correct": True, "repeat": 0},
                 {"id": "qX", "arm": "a", "condition": "grounded",
                  "correct": False, "repeat": 0}]
        _bad = Path(tmp) / "drifted.json"
        _bad.write_text(_json.dumps(
            {"arms": ["a"], "results": _rows,
             "summary": {"a": {"leaked": 0, "leak_rate": 0.0,
                               "leaked_ids": []}}}, indent=2), encoding="utf-8")
        check("the control fails on a summary that denies its own rows",
              len(_cr.check_file(_bad)) == 3,
              "leaked, leak_rate and leaked_ids all caught")

        # And the other direction, so it is not simply always angry.
        _ok = Path(tmp) / "consistent.json"
        _ok.write_text(_json.dumps(
            {"arms": ["a"], "results": _rows,
             "summary": {"a": {"leaked": 1, "leak_rate": 1.0,
                               "leaked_ids": ["qX"]}}}, indent=2), encoding="utf-8")
        check("and passes the same file once the summary tells the truth",
              _cr.check_file(_ok) == [])

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
