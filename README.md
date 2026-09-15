# rag-retrieval-lab

A retrieval-only RAG, forked from a working tutorial and changed until its
behaviour can be measured instead of described.

There is no language model in this repository. That is the point, not an
omission. Retrieval is the half of a RAG that decides whether a correct answer
was ever possible, and it is the half that can be scored against ground truth
with set arithmetic and no judge. Generation is the half that cannot. So
retrieval gets built and measured first.

Nothing here needs an API key, and nothing here costs money to run.

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
```

Then swap `--backend hashing` for `--backend minilm` and compare. The first
MiniLM run downloads about 90 MB of model weights once, after which it is
offline.

No compiler is needed on any platform, and nothing here asks for an API key.

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

## Recorded baseline

`hashing`, `k=5`, chunk 900 / overlap 150, 48 chunks from the 11-page paper,
21 questions of which 6 are unanswerable. Full output in
`eval/results/baseline-hashing-k5.json`.

| metric | value |
| --- | --- |
| hit@5 | 0.733 |
| recall@5 | 0.733 |
| MRR | 0.449 |

Threshold sweep, `found` = answerable questions that still retrieve a correct
page, `declined` = unanswerable questions correctly returning nothing:

| threshold | found | declined | balanced |
| --- | --- | --- | --- |
| 0.00 | 0.73 | 0.00 | 0.37 |
| 0.20 | 0.73 | 0.17 | 0.45 |
| 0.25 | 0.73 | 0.50 | 0.62 |
| **0.30** | **0.73** | **0.67** | **0.70** |
| 0.35 | 0.67 | 0.67 | 0.67 |
| 0.40 | 0.47 | 0.83 | 0.65 |
| 0.50 | 0.13 | 1.00 | 0.57 |
| 0.60 | 0.00 | 1.00 | 0.50 |

Read the shape, not the number. Up to 0.30 the threshold is free: it turns away
two thirds of the unanswerable questions and costs no recall at all. Past 0.35
every further point of caution is paid for in answers that were there and got
thrown away. That is the trade a RAG makes when it decides whether to speak, and
this is what it looks like when it is measured rather than guessed.

The hardest unanswerable question is the one closest to the corpus. "What BLEU
score does the model achieve on English-to-Spanish translation?" scores 0.470,
far above the other five, because the paper is full of BLEU scores for other
language pairs. A threshold that catches "what temperature for sourdough bread"
is easy and worth almost nothing.

**0.30 is not a constant to copy.** It is valid for this corpus, this question
set, this backend and `hnsw:space=cosine`. Copying it elsewhere reproduces
exactly the defect it was written to avoid.

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
