"""The chat the fork promised. History, and what breaks when turns are unbounded.

IA-159. `ask.py` is one question and one answer, which let it avoid the only
interesting problem a chat has. This is that problem.

WHY A REWRITE STEP EXISTS

"And the second one?" cannot be embedded. As a string it means nothing: the
noun it refers to is in the previous turn. Retrieval on it returns whatever is
nearest to the word "second", which is noise. So the question is rewritten into
a standalone one before it reaches the store.

That is the piece the reference tutorial handles worst and the piece a
one-shot interface never has to handle at all.

WHY THE REWRITE IS ALSO A NEW WAY TO LEAK

The rewriter is the same model with the same weights. Asked to make
"and the second one?" standalone, it can quietly produce "what is the second
of the eight attention heads", inserting a fact the corpus was never asked
for. Retrieval then answers a question the user did not ask, grounded in
passages that match the invented detail.

Three things hold it down, and none of them is an instruction:

  1. The rewriter is shown the QUESTIONS only. Not the answers, not the
     passages. It cannot echo corpus content it was never given.
  2. It is asked for a question, and its output is used only as a query. It
     never reaches the user as an answer.
  3. The rewritten query is DISPLAYED. A rewrite that invents is visible on
     screen instead of buried one layer down. This is the actual control: the
     other two reduce the chance, this one makes it observable.

WHY THE BUDGET WORKS DIFFERENTLY HERE

Every other script in this repository knows its call count before it starts,
which is what lets common.check_budget price the whole run and refuse to
begin. A chat does not know how long it will be.

So the rule changes from "refuse to start a run you cannot afford" to "refuse
the next TURN whose worst case does not fit in what is left". Same principle,
different shape, and it is the first time this repository has needed it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import common
import generate

# Shown only the questions, never the answers or the passages. Asked for a
# question, never an answer.
REWRITE_SYSTEM = """You rewrite a follow-up question so it can stand alone.

You are given earlier questions from a conversation and the newest one. The
newest one may refer to the earlier ones ("and the second?", "why is that?").

Rewrite it into a single self-contained question that means the same thing
without the conversation.

Rules:
- Use only what the earlier questions contain. Do not add facts, numbers,
  names or details that are not in them.
- If the newest question already stands alone, return it unchanged.
- Return the question and nothing else. No answer, no explanation, no quotes.
"""

MAX_HISTORY_TURNS = 6


@dataclass
class Turn:
    question: str                 # what the user typed
    query: str                    # what was actually retrieved on
    rewritten: bool               # was the rewrite step used
    passages: list = field(default_factory=list)
    answer: str = ""
    accepted: bool = True         # what the SYSTEM returns, not what the model said
    citation_reason: str = ""
    abstained: bool = False
    usd: float = 0.0
    note: str = ""                # why there is no answer, when there is none


class StubRewriter:
    """Identity. Lets the whole loop be exercised with no key and no spend.

    Deliberately not clever. Its job is to prove the loop calls the rewriter
    when it should and skips it when it should, which a clever stub would
    obscure.
    """

    name = "stub-rewriter"

    def __call__(self, questions: list[str], max_tokens: int = 100):
        return questions[-1], 0.0


class AnthropicRewriter:
    def __init__(self, name: str) -> None:
        self.name = name

    def send_kwargs(self, questions: list[str], max_tokens: int) -> dict:
        joined = "\n".join(f"- {q}" for q in questions[:-1])
        prompt = (f"EARLIER QUESTIONS\n\n{joined}\n\n"
                  f"NEWEST QUESTION\n\n{questions[-1]}")
        return {
            "model": self.name,
            "max_tokens": max_tokens,
            "system": REWRITE_SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
        }

    def __call__(self, questions: list[str], max_tokens: int = 100):
        import os

        from anthropic import Anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set.")
        resp = Anthropic().messages.create(**self.send_kwargs(questions, max_tokens))
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        tin, tout = resp.usage.input_tokens, resp.usage.output_tokens
        usd = common.actual_usd(self.name, tin, tout)
        common.log_usage({"model": self.name, "question": questions[-1][:120],
                          "prompt_arm": "rewrite", "input_tokens": tin,
                          "output_tokens": tout, "usd": round(usd, 6)})
        return (text or questions[-1]), usd


class EchoStub:
    """Not a model. Returns the top passage with its citation. IA-162.

    Why this exists. `generate.StubModel` decides whether to answer by
    comparing the expected page against the retrieved ones, which is exactly
    right inside an evaluation, where ground truth exists. A chat has none, so
    `expected_pages` arrives empty and the stub can only ever decline. It was
    offered in the interface as "the control: if it leaks, the detector is
    broken", a role it only holds in evaluate_answers.py. Here it was an option
    whose label promised something it structurally could not do.

    This one does the job the interface actually needs: it lets the citation
    gate, the passage viewer, the rewrite display and the budget be exercised
    with no API key and nothing spent.

    It is not a model and says so. It answers nothing. It hands back what an
    answer would have been built from, cited, so you can see the machinery
    work before deciding to pay for a real one.

    It goes through the same gate as everything else, because a stand-in
    exempt from a control cannot exercise it.
    """

    name = "stub-echo"

    def __call__(self, question, passages, max_tokens: int = 0, **_):
        if not passages:
            return generate._gated(common.ABSTAIN, self.name, passages)
        top = passages[0]
        body = " ".join(top.text.split())
        if len(body) > 400:
            body = body[:400].rsplit(" ", 1)[0] + "..."
        return generate._gated(
            f"{body} [{top.source} p.{top.page}]", self.name, passages)


def get_model(name: str):
    """The chat's model list, which is not the evaluator's.

    stub and stub-memoriser are evaluation instruments. They need a question
    set with expected pages to mean anything, and a chat does not have one.
    They are deliberately not offered here.
    """
    return EchoStub() if name == "stub-echo" else generate.get_model(name)


def get_rewriter(name: str):
    return StubRewriter() if name.startswith("stub") else AnthropicRewriter(name)


class ChatSession:
    def __init__(self, model: str = common.DEFAULT_MODEL, backend: str | None = None,
                 k: int = 5, threshold: float | None = None,
                 max_tokens: int = common.MAX_OUTPUT_TOKENS,
                 max_usd: float = 0.25, search: str = "ann") -> None:
        self.model_name = model
        self.model = get_model(model)
        self.rewriter = get_rewriter(model)
        self.backend = backend or common.DEFAULT_BACKEND
        self.k = k
        self.threshold = threshold
        self.max_tokens = max_tokens
        self.max_usd = max_usd
        self.search = search
        self.turns: list[Turn] = []
        self.spent = 0.0

    # --- preconditions ----------------------------------------------------
    def missing_key(self) -> bool:
        """True when this session needs a key and the shell has none.

        Only presence is checked. The value is never read, never logged and
        never leaves the environment.

        This exists because the backends raise SystemExit when the key is
        absent, which is correct for a CLI and silent in a UI: SystemExit is a
        BaseException, Streamlit does not catch it, the script run ends, the
        turn is never recorded and the page comes back blank with no error.
        A failure that produces an empty screen is worse than a crash. IA-164.
        """
        import os

        return (not self.model_name.startswith("stub")
                and not os.environ.get("ANTHROPIC_API_KEY"))

    # --- budget -----------------------------------------------------------
    @property
    def remaining(self) -> float:
        return max(0.0, self.max_usd - self.spent)

    def _worst_case_next_turn(self, question: str) -> float:
        """Priced before spending, as everywhere else. Two calls after the first.

        Deliberately over-stated, same as common.check_budget: it assumes the
        answer exhausts max_tokens and that the prompt is as long as k full
        passages. Erring high means stopping early rather than overshooting,
        and overshooting is the failure that matters.
        """
        n_calls = 1 if not self.turns else 2
        est_in = common.estimate_input_tokens(
            generate.SYSTEM + question + "x" * (self.k * common.CHUNK_SIZE))
        return common.worst_case_usd(self.model_name, n_calls, est_in,
                                     self.max_tokens)

    # --- one turn ---------------------------------------------------------
    def ask(self, question: str) -> Turn:
        question = (question or "").strip()
        if not question:
            return Turn(question, question, False, note="empty question")

        if self.missing_key():
            return Turn(question, question, False,
                        note=("ANTHROPIC_API_KEY is not set in the shell that "
                              "started this app, and `" + self.model_name +
                              "` needs it. Nothing was sent. Set it and "
                              "restart, or switch Model to `stub-echo`, which "
                              "needs no key. The key is never read from a file "
                              "in this repository and never committed."))

        # Control: refuse the TURN, not the run. A chat cannot be priced in
        # advance, so the budget is checked against what is left each time.
        worst = self._worst_case_next_turn(question)
        if worst > self.remaining:
            return Turn(question, question, False,
                        note=(f"BUDGET STOP: this turn's worst case "
                              f"${worst:.6f} exceeds the ${self.remaining:.6f} "
                              f"left of ${self.max_usd:.2f}. Nothing was sent."))

        # The rewrite is skipped on the first turn. There is no history to
        # resolve against, so the call would cost money to return its input.
        usd = 0.0
        rewritten = False
        query = question
        if self.turns:
            earlier = [t.question for t in self.turns][-MAX_HISTORY_TURNS:]
            query, r_usd = self.rewriter(earlier + [question],
                                         max_tokens=min(120, self.max_tokens))
            usd += r_usd
            rewritten = query.strip() != question

        passages = generate.retrieve(query, self.k, self.backend,
                                     self.threshold, self.search)

        # Control 4, unchanged: nothing above the threshold means nothing to
        # answer from, so no request is made and the turn costs only the
        # rewrite.
        if not passages:
            # The first version of this said "no passage met the threshold"
            # whatever the reason. With the threshold off that sentence cannot
            # be true, and it sent the diagnosis in the wrong direction for an
            # afternoon. A message that names a cause it did not check is the
            # defect this repository is about. IA-163.
            #
            # So the cause is established rather than assumed: ask the store
            # how many vectors it holds and what it returned before filtering.
            self.spent += usd
            store = common.get_store(self.backend)
            try:
                n_vectors = store._collection.count()
                where = store._client.get_settings().persist_directory
            except Exception as e:
                n_vectors, where = -1, f"unreadable: {e}"
            raw = []
            if n_vectors > 0:
                try:
                    raw = store.similarity_search_with_score(query, k=self.k)
                except Exception:
                    raw = []

            if n_vectors == 0:
                why = (f"The store is EMPTY. `{where}` holds 0 vectors for "
                       f"backend `{self.backend}`. Nothing was retrieved "
                       f"because there is nothing to retrieve. Run "
                       f"`python ingest.py --backend {self.backend}` from the "
                       f"same directory this app was started in.")
            elif not raw:
                why = (f"The store holds {n_vectors} vectors at `{where}` and "
                       f"the search returned nothing for this query, which "
                       f"should not happen with k={self.k}.")
            else:
                best = max(common.similarity_from_distance(d) for _, d in raw)
                why = (f"All {len(raw)} results scored below the threshold "
                       f"{self.threshold}. Best was {best:.4f}. Lower the "
                       f"threshold to see them.")

            t = Turn(question, query, rewritten, [], "", True,
                     "nothing retrieved", True, usd,
                     note="Declined without calling the model. " + why)
            self.turns.append(t)
            return t

        try:
            a = self.model(query, passages, max_tokens=self.max_tokens)
        except BaseException as e:          # SystemExit included, deliberately
            self.spent += usd
            t = Turn(question, query, rewritten, passages, "", True,
                     "the call failed", True, usd,
                     note=f"The model call failed: {type(e).__name__}: {e}")
            self.turns.append(t)
            return t
        usd += a.usd
        self.spent += usd

        t = Turn(question, query, rewritten, passages, a.text, a.accepted(),
                 a.citation_reason, a.abstained, usd)
        if not a.accepted():
            t.note = ("The model produced an answer and the system does not "
                      "return it: " + a.citation_reason + ". An assertion the "
                      "corpus cannot be shown to support is not an answer.")
        self.turns.append(t)
        return t
