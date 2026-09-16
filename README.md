# rag-retrieval-lab

**A chat that refuses to answer what it cannot cite**, forked from a working
tutorial and changed until its behaviour can be measured instead of described.

Ask it something about the corpus and every claim comes back carrying the
document and page it was built from. When the passages do not support an
answer it says so. An assertion the corpus cannot be shown to support is
rejected by the system rather than returned, whatever the model produced.

That is the product. The rest of this repository is the reason to believe it.

Retrieval was built and measured first, with no language model at all, because
it is the half of a RAG that decides whether a correct answer was ever
possible and the half that can be scored against ground truth with set
arithmetic and no judge. Generation is the half that cannot, so it came second
and had to prove its answers came from the corpus rather than from what the
model already knew.

**Two halves, two costs.** Retrieval, the metrics, the invariant suite, the
sabotage construction, the citation gate and a keyless chat all run with no
API key and no spend. Answering with a real model needs a key and costs money,
about two tenths of a cent per turn, and every control around that is in
"Spending is bounded" below.

## Quick start

```
python -m venv venv
venv\Scripts\Activate          # Windows
source venv/bin/activate       # macOS, Linux
pip install -r requirements.txt

# put one PDF in data/  (see "The corpus" below)

python verify_questions.py               # is the ground truth still true?
python ingest.py --backend hashing       # build the store
python ingest.py --backend hashing       # run it again: it must add nothing
python evaluate.py --backend hashing -k 5 --sweep
python search.py "how many attention heads does the model use" --backend hashing

# the generation layer, with no API key and no spend
python evaluate_answers.py --backend hashing --model stub             # must leak 0
python evaluate_answers.py --backend hashing --model stub-memoriser   # must catch all

# re-read anything already run, no key, no spend
python replay_gate.py eval/results/experiment-final.json   # re-score with the shipped gate
python check_results.py                                    # does each summary match its rows?

# the chat, with no key and nothing spent
streamlit run app.py           # pick Model: stub-echo in the sidebar

# with a real model
export ANTHROPIC_API_KEY=...
python ask.py "how many attention heads does the model use" --max-usd 0.01
python evaluate_answers.py --backend minilm --model claude-haiku-4-5 \
    --max-usd 0.15 --json-out eval/results/answers-haiku.json
```

**The two questions worth asking the chat first**, in this order:

1. `how many attention heads does the model use`
2. `and how many layers?`

The second is the whole point and is explained under "The chat" below.

Then swap `--backend hashing` for `--backend minilm` and compare. The first
MiniLM run downloads about 90 MB of model weights once, after which it is
offline.

No compiler is needed on any platform. Everything above the "with a real
model" block runs without a key and spends nothing, the chat included.

## The generation layer, and the test that makes it mean something

`ask.py` answers a question from the retrieved passages, citing the document and
page for every claim, or replies `INSUFFICIENT EVIDENCE` when the passages do
not support an answer.

The hard part is not writing that. It is proving it works. The reference this
was forked from demos its RAG on *Attention Is All You Need*, the most famous
paper in machine learning, which every current model knows by heart. Ask it how
many attention heads the model uses and it answers "eight" whether the retriever
supplied page 5 or the bibliography. The reference tries to prevent that with a
line in the prompt:

> *"you don't use your internal knowledge, but solely the information in the
> 'The knowledge' section"*

That is an instruction, not a mechanism. Nothing enforces it and nothing detects
a breach, so the demo behaves identically with the retriever unplugged.

`evaluate_answers.py` turns it into a measurement. Every answerable question is
asked twice:

- **grounded**: the passages retrieval actually returned
- **sabotaged**: passages deliberately taken from pages that cannot contain the answer

Declining under sabotage is correct. Answering *correctly* under sabotage means
the answer came from memory, and that question proves nothing about retrieval.
The headline number is **what fraction of the correct answers is attributable to
retrieval rather than to recall**.

Correctness is scored without an LLM judge: the question set carries an
`answer_contains` string that a correct answer must include, matched on word
boundaries. **This is a weak check and that is stated rather than hidden.** It
can pass on a wrong answer containing the string and fail on a right answer
phrased differently. 11 of the 15 answerable questions have an answer crisp
enough for it; the rest are reported as answered-or-declined only.

### Spending is bounded, and the bound is demonstrated

Not because the amounts are large. A full 36-call run on Claude Haiku 4.5 prices
at a worst case of **$0.1123**. Because "it is cheap" is not a control.

1. `--max-tokens` caps output per call.
2. `--max-usd` prices the whole run before the first request and **refuses to
   start**, rather than stopping halfway. Set it to `0.0001` and watch it stop
   with `Nothing was sent.`
3. Every response's real `usage` is appended to `eval/usage-log.jsonl`, so spend
   is observed rather than estimated. That file is gitignored.
4. **The abstention gate is a spend control.** A question declined by retrieval
   never reaches the API. At `--threshold 0.45` the hashing backend skips 10 of
   21 questions, and those cost nothing.

Prices are recorded in `common.py` with the date they were read
([Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing),
16 Sep 2026). An unpriced model raises rather than being guessed at, and a test
asserts that.

### Verifiable with no API key

`--model stub` is a perfectly grounded model and `--model stub-memoriser` is one
that ignores the passages entirely. The first must produce **zero** memory leaks
and the second must be caught on **every** question. Two stubs, one that should
trip nothing and one that should trip everything, is what makes the detector
falsifiable in both directions rather than merely present.

## The corpus

**No corpus is committed here, deliberately.** `data/` is in `.gitignore`.

The question set in `eval/` is written against Vaswani et al., *Attention Is All
You Need*, NIPS 2017, 11 pages. To reproduce the numbers below, put that PDF in
`data/`. It is freely available from the NeurIPS proceedings site.

The reason it is not in this repository is worth stating, because it is the same
class of problem this project is about. The reference repository ships that PDF
under a repository-wide MIT LICENSE. MIT is the author's grant over the author's
work, and a third party's published paper is not the author's work to grant. The
licence therefore states a guarantee wider than the one behind it. Shipping no
corpus is the cheapest way not to repeat that.

`verify_questions.py` exists so the ground truth is checkable rather than
asserted: it extracts the text of every page the question set names and confirms
the recorded probe string is actually on it. A question set that has drifted from
its corpus turns every number downstream into a measurement of nothing.

## What changed from the reference

Forked from [ThomasJanssen-tech/Chatbot-with-RAG-and-LangChain](https://github.com/ThomasJanssen-tech/Chatbot-with-RAG-and-LangChain)
(MIT, Thomas Janssen). It is a clear tutorial and the two-step shape is its idea.
Five changes, each fixing something found by reading its source.

**1. The ingest is idempotent, and that is proven rather than claimed.**

The reference generates a fresh `uuid4` for every chunk on every run:

```python
uuids = [str(uuid4()) for _ in range(len(chunks))]
vector_store.add_documents(documents=chunks, ids=uuids)
```

Chroma accepts explicit ids precisely so that re-adding identical content
upserts. A random id removes that property while appearing to use it. Because
the store persists, a second ingest appends the corpus to itself, and at `k=5`
a duplicated store returns the same passage several times, so the model sees
*less* distinct context than on a clean run. Nothing reports it.

Here the id is `sha256(source | page | chunk_index | text)`. Running `ingest.py`
twice prints `net added 0`, and `test_invariants.py` asserts it.

**2. Citations survive.**

`PyPDFDirectoryLoader` attaches `source` and `page` to every chunk. The
reference reads `doc.page_content`, drops the metadata, and then instructs the
model, in the prompt: *"You don't mention anything to the user about the povided
knowledge."* The system is built not to say where its answer came from.

Here every result prints its document and page, and a test fails if metadata
stops surviving the round trip.

**3. There is an abstention threshold, and it was measured.**

The reference retrieves an unconditional `k=5` with no score filter, so it can
never decline. A related tutorial declines below `0.7`, a number that is never
justified and that silently depends on the store's distance function.

Two things are needed to do better. The distance function is pinned explicitly
(`hnsw:space=cosine`, and Chroma's default is `l2`, so a threshold tuned in one
space is meaningless in the other). And the question set contains **questions the
corpus cannot answer**, because with only answerable questions the best possible
threshold is always zero and the exercise is circular.

`evaluate.py --sweep` then reports both sides of the trade at every threshold.

**4. There is an evaluation, and it can come back bad.**

None of the five RAG repositories audited for this had any. Without one, chunk
size, `k`, the threshold and the choice of embedding model are all taste.

**5. One definition of the embedding model, imported by both scripts.**

The reference constructs `OpenAIEmbeddings(model="text-embedding-3-large")`
separately in `ingest_database.py` and in `chatbot.py`. Change it in one file and
the stored vectors and the query vectors come from different models. Here it
lives in `common.py`, and the backend name is part of the store directory, so two
models cannot land in one collection.

## The two backends

`hashing` is a deterministic bag of words, L2 normalised, no model and no
download. It is not a stand-in for MiniLM. It is a **floor**: it has no notion of
synonymy, so any question phrased differently from the passage it should find
will miss. If a sentence embedding model cannot beat it, the embedding step is
not earning its place.

`minilm` is `sentence-transformers/all-MiniLM-L6-v2`, downloaded once and then
local. No API key, no per-query cost, no network at inference.

## Results: retrieval

Two backends, same corpus, same question set. 48 chunks from the 11-page paper,
chunk 900 / overlap 150, k=5, 21 questions of which 6 are unanswerable. Full
output in `eval/results/`.

| metric | hashing (floor) | MiniLM |
| --- | --- | --- |
| hit@5 | 0.733 | 0.733 |
| recall@5 | 0.733 | 0.733 |
| MRR | 0.449 | **0.539** |
| right page at rank 1 | 4 of 15 | **6 of 15** |
| best balanced threshold | 0.30 | 0.50 |
| balanced score at it | **0.70** | 0.65 |

### The aggregate did not move, and almost everything underneath it did

Both backends find the right page for 11 of 15 answerable questions. That single
number hides the fact that **four of fifteen questions changed outcome**:

| | hashing | MiniLM | |
| --- | --- | --- | --- |
| q03, how many identical layers | MISS | rank 2 | fixed |
| q14, what regularisation on the output | MISS | rank 4 | fixed |
| q02, what does it dispense with | rank 5 | MISS | broken |
| q10, what function encodes position | rank 2 | MISS | broken |

Two fixed, two broken, on fifteen questions. An aggregate that stays still is
not evidence that the system stayed still, and with a set this small the
sensible reading of `hit@5` is that it did not distinguish the two backends at
all.

MRR is the more defensible improvement, and it comes from somewhere specific:
not from finding more pages, but from ranking better the ones both backends
find. Rank-1 hits go from 4 to 6, and q06 moves from rank 5 to rank 2, q07 from
3 to 1, q12 from 2 to 1. That is what a sentence embedding is supposed to buy,
and here it bought it.

### The result that matters: a better embedding made abstention harder

The six unanswerable questions, by top score:

| | | hashing | MiniLM | |
| --- | --- | --- | --- | --- |
| u01 | BLEU for English to Spanish | 0.470 | **0.709** | +0.239 |
| u02 | BERT fine-tuning learning rate | 0.226 | **0.483** | +0.257 |
| u03 | GPT-4 parameter count | 0.172 | **0.272** | +0.100 |
| u04 | ImageNet top-1 accuracy | 0.379 | **0.427** | +0.048 |
| u05 | S3 bucket policy for CloudFront | 0.231 | 0.110 | -0.121 |
| u06 | oven temperature for sourdough | 0.290 | 0.170 | -0.120 |

The split is clean and it is not a coincidence. The four questions MiniLM got
**more** confident about are the four that are about machine translation, BERT,
GPT-4 and ImageNet: adjacent to the corpus and absent from it. The two it got
**less** confident about are the two from another world entirely.

A sentence embedding is good at recognising what a question is about. Being
about the same thing is precisely what makes an unanswerable question dangerous.
So the better model pulls near-misses up and pushes far-misses down, and the
near-misses were always the hard part.

The consequence is visible in the bottom row of the first table. On the combined
objective of answering when it can and declining when it cannot, **MiniLM scores
worse than hashed bag-of-words**, 0.65 against 0.70, and it needs a threshold of
0.50 rather than 0.30 to get there, by which point it has already discarded a
third of its own correct retrievals.

**Semantic similarity is not answerability.** Every system in this family
computes the first and then uses it as though it were the second, which is why
"retrieve the top k and tell the model to answer from them" produces confident
answers to questions the corpus cannot support. A better retriever does not fix
that. It sharpens it.

### What is still missed by both

q01 asks who the authors are; q15 asks what happens when the number of attention
heads is varied. Neither backend finds them, which points away from the
embedding and towards chunk size, k, or the fact that page 1 is mostly a block
of names and affiliations that no prose question resembles. That is a separate
fix and it is not one a different model would make.

## Results: the generation half

Everything above is retrieval. This section is the part the README used to
describe and never report, which is its own small instance of the defect this
repository is about: it announced a headline number and did not print it.

`claude-haiku-4-5`, MiniLM backend, k=5, three prompt arms, three repeats. 324
calls, **$0.5915 measured**, not estimated. Raw output in
`eval/results/experiment-final.json`.

One variable moves: the system prompt.

| arm | what it says |
| --- | --- |
| `grounded` | cite every claim, or reply `INSUFFICIENT EVIDENCE` |
| `reference` | the forked tutorial's prompt, verbatim, typos included. No way to refuse |
| `reference-plus-abstain` | the reference prompt plus exactly that refusal token |

| | correct | leaked | declined under sabotage | rejected by the citation gate |
| --- | --- | --- | --- | --- |
| `grounded` | 7/11 | **0** | 45/45 | 8/108 |
| `reference` | 7/11 | **1** (q02, repeat 1) | 0/45 | **108/108** |
| `reference-plus-abstain` | 7/11 | **0** | 45/45 | 33/108 |

Reproduce both right-hand columns with no API key and no spend:

```bash
python replay_gate.py eval/results/experiment-final.json   # the gate column
python check_results.py                                    # summary vs its own rows
```

### The hypothesis was mine and the data refuted it

The prediction on record was that the reference arm would leak heavily, because
it forbids using internal knowledge while leaving no way to decline: told to
answer from passages that cannot answer, a model has one route to producing
output at all. **It leaked once in 33 scorable question-repeats.** It never
emitted a refusal token, because it has none, and it still declined in prose
most of the time.

How often it declined in prose is deliberately not a number here. Counting it
needs a classifier for free-text refusal, and there isn't one: a regex written
for the occasion caught 23 of 45 while missing "I cannot answer", "I cannot
find information" and "I don't see information in the provided passages" in the
first three rows it was shown. An instrument that bad does not get to produce a
figure. What is measured is the leak, and the leak is 1.

The wrong prediction is recorded here rather than quietly replaced by the
result, because a project about unbacked claims does not get to hide its own.

### What the affordance actually buys is auditability

Look at the last column. The reference arm is rejected by the citation gate
**108 times out of 108**. It is not wrong 108 times. It is **unauditable by
construction**: its prompt forbids mentioning the supplied knowledge, so it
cites nothing, so nothing downstream can tell a grounded answer from a
remembered one. The one real leak is in there, indistinguishable from the 107
answers that were fine.

`grounded` is rejected 8 times out of 108, and those 8 are findable.

**The prompt does not make the model honest. It makes the model checkable.**
That conclusion rests on 108 observations rather than on the single leak, which
is why it is the one worth keeping.

### Our own prompt was not exempt

`grounded` has always said that an uncited claim is a failure, and 8 of its 108
responses carried no citation while nothing failed. An instruction with no
mechanism is not a rule. The gate is the mechanism it was missing, and it was
written after the prompt had been claiming the property for weeks.

### A finding that was retracted

An earlier run had `reference` at 8/11 against `grounded` at 7/11, and a
paragraph was written about the refusal affordance costing an answer. It was one
question flipping between runs. The SDK exposes no temperature setting, and 48
of 63 byte-identical inputs produced different text across two runs. The 8/11 is
not in the numbers above, and `--repeats` with a per-repeat spread exists so
that a single sample can never be read as a measurement again.

### What this does not show

One model, one corpus, 11 scorable questions, three repeats. Nothing here
generalises. The leak count is 1, and a single observation is an anecdote with
an exit code. The 108/108 auditability result is the robust one.

**And none of it makes the model forget.** The knowledge is in the weights. The
gate makes using it without support detectable and refusable. Detection and
incentive, not amnesia.

## The chat

`streamlit run app.py`. Every turn shows the passages it used with their
source, page and score, the citation verdict, the running spend, and which
store it is reading from. When the corpus does not support an answer it
declines instead of bluffing.

Two model options. `stub-echo` is not a model: it hands back the top passage
with its citation, which exercises the gate, the passage viewer and the budget
with no key and nothing spent. `claude-haiku-4-5` answers for real.

### The only interesting problem a chat has

`ask.py` is one question and one answer, which let it avoid this entirely.

"And how many layers?" cannot be embedded. As a string it means nothing: the
noun it refers to is in the previous turn. So it is rewritten into a standalone
question before it reaches the store. Measured, same question both ways:

| | retrieved on | answer |
| --- | --- | --- |
| **with the rewrite** | "How many layers does the model have?" | "a total of 12 layers: the encoder is composed of a stack of N = 6 identical layers, and the decoder is also composed of a stack of N = 6" `[p.3]` |
| **without it** | "and how many layers?" | a passage about **dropout**, `p.7` |

The second is what the tutorial this was forked from does with a follow-up.
Run the chat twice, once on `stub-echo` and once on the real model, and you see
both rows.

### The rewrite is also a new way to leak, and only one control is a mechanism

The rewriter is the same model with the same weights. Asked to make "and the
second one?" standalone, it can produce "what is the second of the eight
attention heads" and insert a fact the corpus was never asked for. Retrieval
then answers a question nobody asked.

Three things hold it down:

1. It is shown the **questions only**. Never the answers, never the passages.
2. Its output is used only as a query, never returned as an answer.
3. **The rewritten query is displayed.**

The first two reduce the chance and enforce nothing. The third is the only
mechanism: a rewrite that invents is visible on screen rather than buried a
layer down. Nothing here measures how often it happens, and on this corpus it
never visibly did, which is an absence of observation and not evidence.

### The budget changes shape when turns are unbounded

Every other script here knows its call count before it starts, which is what
lets the budget price the whole run and refuse to begin. A chat does not know
how long it will be.

So the rule becomes: refuse the **next turn** whose worst case does not fit in
what is left. Same principle, different shape, and the first time this
repository needed it.

### Six defects came out of building it, and none were in the RAG

Retrieval and generation worked throughout. Every defect was in what the
interface tells the person using it: a sidebar that read `stub` while the
session spent real money, a control that could only ever decline, a failure
message blaming a threshold that was switched off, a blank page when the API
key was missing, and a store silently redirected by an environment variable set
hours earlier.

A system can be correct and still lie to its user. That does not show up in any
accuracy number, and it is the most useful thing this repository found.
Recorded as IA-159 through IA-164.

## The second corpus: ObliQA

`corpus_obliqa.py` loads 13,732 passages of ADGM financial regulation with
ground truth, for measuring retrieval at a scale where the reference paper's 48
chunks say nothing. Not committed and never redistributed: it has no LICENSE,
so it gets the same treatment as the reference PDF.

```
git clone --depth 1 https://github.com/RegNLP/ObliQADataset.git data/obliqa
python corpus_obliqa.py
python ingest.py --corpus obliqa --data data/obliqa --backend hashing
```

Four things are wrong with that corpus and the loader handles each at the door
rather than letting it surface later as a strange metric:

- **The ground-truth key is not unique.** 17 keys carry different text under
  the same `(DocumentID, PassageID)`; one has seven. Resolved through the
  passage text the question files carry, and anchored to the record uuid. A
  pointer that cannot be resolved is reported, never guessed.
- **864 of 11,529 passages exceed the model's 256 word-piece window** and are
  truncated silently at embed time. Decision on record: truncate and record it.
  Splitting breaks the unit the ground truth names; excluding changes the
  question population. The full text is kept and an `over_window` flag carries
  the fact, so recall can be reported with and without that subset. 364 of
  2,788 dev questions have every supporting passage in it, which is a recall
  ceiling the embedding cannot be blamed for.
- **720 empty passages and 1,483 under 40 characters**, mostly headings.
  Dropped by argument and counted, never silently.
- **Zero dangling pointers**, re-checked on every run rather than once by hand.

Retrieval on it is **not finished**: the pipeline's ground truth is an integer
page and a regulation clause id is not, so citations currently read `p.0` for
every passage. The ingest warns about it. Tracked as IA-157.

## Every file in this repository

| | |
| --- | --- |
| `common.py` | config, embeddings, the store, prices, budget arithmetic, exact search |
| `ingest.py` | build the store from a PDF or from ObliQA. Idempotent, batched |
| `corpus_obliqa.py` | the second corpus, and the four defects in it |
| `search.py` | retrieval only, no model |
| `evaluate.py` | hit@k, recall@k, MRR, and the threshold sweep |
| `verify_questions.py` | is the ground truth still true of the corpus |
| `generate.py` | prompts, the model backends, and the citation gate |
| `ask.py` | one question, one cited answer |
| `chat.py` | the chat session: history, rewrite, gate, per-turn budget |
| `app.py` | Streamlit over `chat.py` |
| `evaluate_answers.py` | the sabotage experiment and the prompt arms |
| `replay_gate.py` | re-score a saved run with the shipped gate, free |
| `check_results.py` | does a saved summary still agree with its own rows |
| `check_index.py` | is the approximate index costing recall |
| `test_invariants.py` | 81 assertions, no key, no spend |

## What was verified, and where

Two different limits apply, and they are worth keeping apart.

**Verified, on Linux with Python 3.11, using `--backend hashing`:**
chunk ids unique, deterministic and content-sensitive; citations surviving into
and out of the store; a second ingest adding nothing; similarity scores in
`[0,1]` and correctly ordered; the metric arithmetic against a hand-worked
fixture; the sweep's optimum landing strictly between the extremes; the question
set verified probe by probe against the PDF. 18 assertions, all passing, against
chromadb 1.5.9 with `hnsw:space` read back from the live collection rather than
assumed.

**Not verified anywhere reachable from the authoring session: MiniLM.**
`huggingface.co` is blocked at the proxy in both the session container and the
desktop workspace (`403 Forbidden` at CONNECT, confirmed on the model download
itself, while `pypi.org` returns 200 in both). The weights cannot be fetched
from either, so the first `--backend minilm` run happens on a real machine, and
the number it produces should beat the hashing baseline above.

**The pins were wrong once, and the reason is worth recording.** The first
version of `requirements.txt` was checked by installing it on Linux with Python
3.11, where it works. It cannot install on Windows with Python 3.12, because
`chromadb` 0.6.3 requires the `chroma-hnswlib` C++ extension and that package's
Windows wheels stop at cp311. "The pins resolve" was true of one platform and
was written as though it were true in general. The current pins avoid the
compiler entirely; see the note at the bottom of `requirements.txt`. The
platform matrix itself is still checked on one platform at a time, and saying so
is the honest version of the claim.

## Provenance

Forked from Thomas Janssen's MIT-licensed tutorial. See `NOTICE` and `LICENSE`.
Tracked as IA-134.

## Errata

`ERRATA.md` records corrections to this repository's published history. Nothing
there is edited out of the history it corrects, because a commit's hash is the
evidence recorded against its issue.

Currently one entry: five commit messages cite Jira keys that did not exist when
they were written, and the citation gate work is referenced throughout as IA-151
when it is IA-149. If you are following a key out of a commit message and it
goes nowhere, start there.
