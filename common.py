"""Shared configuration and store access.

The reference this is forked from instantiates its embeddings model separately
in ingest_database.py and in chatbot.py. That is a latent defect: changing the
model in one file and not the other leaves the query vectors and the stored
vectors coming from different models, and a vector store will compute distances
between them without complaining. One definition, imported by both, removes the
possibility.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- configuration ------------------------------------------------------

DATA_PATH = Path(os.environ.get("RAG_DATA_PATH", "data"))
CHROMA_PATH = Path(os.environ.get("RAG_CHROMA_PATH", "chroma_db"))
COLLECTION = os.environ.get("RAG_COLLECTION", "corpus")

# Chunking. The reference uses 300/100. That is roughly 50 words, which rarely
# holds a complete idea in a technical document, and 33% overlap inflates the
# store for little gain. These are the values we start from; they are a
# parameter of the experiment, not a setting, and evaluate.py is what decides
# whether they are right.
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", 900))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", 150))

# The distance function is pinned explicitly. Chroma's default is l2, and a
# threshold tuned against one space is meaningless in the other. IA-125 recorded
# this as the reason pixegami's inherited 0.7 could not be trusted.
HNSW_SPACE = "cosine"

EMBEDDING_BACKENDS = ("minilm", "hashing")
DEFAULT_BACKEND = os.environ.get("RAG_EMBEDDINGS", "minilm")


# --- embeddings ---------------------------------------------------------

def get_embeddings(backend: str = DEFAULT_BACKEND):
    """Return an embeddings object.

    minilm  : sentence-transformers/all-MiniLM-L6-v2, downloaded once, then
              local and offline. No API key and no per-query cost.
    hashing : a deterministic lexical bag-of-words baseline with no model and
              no download. It exists so the pipeline can be tested where the
              model cannot be fetched, and so there is a floor to beat. If
              MiniLM does not beat hashing on the evaluation, the embedding
              step is not earning its place.
    """
    if backend == "minilm":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True},
        )
    if backend == "hashing":
        return HashingEmbeddings()
    raise ValueError(f"unknown backend {backend!r}, expected one of {EMBEDDING_BACKENDS}")


class HashingEmbeddings:
    """Hashed bag of words, L2 normalised. Deterministic, no model, no network.

    This is a baseline, not a stand-in. It has no notion of synonymy, so any
    question phrased differently from the passage it should find will miss.
    That is exactly the gap a sentence embedding is supposed to close, which is
    what makes it a useful floor.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        import math
        import re

        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int.from_bytes(hashlib.sha1(tok.encode()).digest()[:8], "big")
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0:
            v[0] = 1.0
            return v
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


# --- store --------------------------------------------------------------

def get_store(backend: str = DEFAULT_BACKEND, chroma_path: Path | None = None) -> Chroma:
    """Open (or create) the persistent store.

    The backend is part of the directory name. Vectors from two different
    models must never land in the same collection, and relying on a human to
    remember that is how it goes wrong.
    """
    path = Path(chroma_path) if chroma_path else CHROMA_PATH / backend
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(backend),
        persist_directory=str(path),
        collection_metadata={"hnsw:space": HNSW_SPACE},
    )


def load_pdf_pages(data_path: Path) -> list[Document]:
    """Read every PDF in data_path, one Document per page, via pypdf directly.

    Deliberately not langchain_community's PyPDFDirectoryLoader. Two reasons.

    The dependency: that loader is the only thing this project used from
    langchain-community, and it pulls langchain, SQLAlchemy, aiohttp, langsmith
    and dataclasses-json in behind it. That is the reference's "140 pins for six
    imports" problem arriving through a transitive dependency instead of a
    requirements file, and it counts the same.

    The correctness one matters more. verify_questions.py checks the ground
    truth by extracting page text with pypdf. If the ingest extracted text by a
    different path, the question set would be verified against one string and
    the store would be built from another, and the check would quietly be about
    something other than what is indexed. One function, used by both, removes
    the possibility rather than relying on the two agreeing.
    """
    from pypdf import PdfReader

    docs: list[Document] = []
    for pdf in sorted(data_path.glob("*.pdf")):
        for page_no, page in enumerate(PdfReader(str(pdf)).pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": pdf.name, "page": page_no},
                )
            )
    return docs


def chunk_id(source: str, page: int, index: int, text: str) -> str:
    """Content-addressed id.

    This is the whole of the idempotency fix. The reference generates a fresh
    uuid4 for every chunk on every run, which means a second ingest against a
    persisted store appends the corpus to itself. Deriving the id from the
    content makes a re-ingest an upsert of identical rows.
    """
    payload = f"{source}|{page}|{index}|{text}".encode()
    return hashlib.sha256(payload).hexdigest()[:32]


def similarity_from_distance(distance: float) -> float:
    """Chroma returns a distance. With hnsw:space=cosine that is 1 - cosine.

    Reported as a similarity in [0, 1] so that "higher is better" holds and a
    threshold reads the way a person expects. Pinned to the space configured
    above; if HNSW_SPACE changes, this must change with it.
    """
    if HNSW_SPACE != "cosine":
        raise NotImplementedError(f"no similarity mapping defined for {HNSW_SPACE}")
    return 1.0 - float(distance)


# =====================================================================
# Generation layer (IA-143)
# =====================================================================

# Prices in USD per million tokens, read from
# https://platform.claude.com/docs/en/about-claude/pricing on 16 Sep 2026.
# Dated on purpose. A price quoted from memory is not a price.
PRICES_USD_PER_MTOK = {
    "claude-haiku-4-5":  {"in": 1.00, "out": 5.00},
    "claude-sonnet-5":   {"in": 2.00, "out": 10.00},
    "claude-opus-5":     {"in": 5.00, "out": 25.00},
    "stub":              {"in": 0.00, "out": 0.00},
    "stub-memoriser":    {"in": 0.00, "out": 0.00},
}

DEFAULT_MODEL = os.environ.get("RAG_MODEL", "claude-haiku-4-5")
MAX_OUTPUT_TOKENS = int(os.environ.get("RAG_MAX_OUTPUT_TOKENS", 300))

# Temperature 0 by default. This was not set for the first two experiment runs
# and it should have been: with identical inputs, 48 of 63 calls produced
# different text between runs, and one verdict flipped, which was the whole of
# an 8/11-versus-7/11 difference that had already been written up as a finding.
# An experiment whose arms are compared on one sample each needs its only source
# of randomness pinned. Temperature 0 reduces variance; it does not abolish it,
# so repeats still matter. IA-148.
TEMPERATURE = float(os.environ.get("RAG_TEMPERATURE", 0.0))
USAGE_LOG = Path(os.environ.get("RAG_USAGE_LOG", "eval/usage-log.jsonl"))

# The exact string the model is told to produce when the passages do not
# support an answer.
ABSTAIN = "INSUFFICIENT EVIDENCE"


def looks_like_abstention(text: str) -> bool:
    """Did the model decline?

    Deliberately not `text == ABSTAIN`. IA-125 recorded a tutorial whose
    abstention check was exact string equality against free-form model output,
    which the single character in "NA." defeats. Normalise, then look at the
    opening, because a model that declines and then explains why has still
    declined.
    """
    import re as _re

    head = _re.sub(r"[^a-z ]", " ", (text or "").lower())
    head = " ".join(head.split())
    return head.startswith(ABSTAIN.lower())


def estimate_input_tokens(text: str) -> int:
    """Upper-leaning estimate: 3.5 characters per token.

    Anthropic's real count is only available from the API, and the point of a
    pre-flight budget is to decide BEFORE spending anything. English prose runs
    nearer 4 characters per token, so 3.5 over-counts, which is the direction an
    estimate guarding a budget should err in. Every logged figure afterwards is
    the real count from the response, not this.
    """
    return int(len(text) / 3.5) + 1


def worst_case_usd(model: str, n_calls: int, est_input_tokens: int,
                   max_output_tokens: int = MAX_OUTPUT_TOKENS) -> float:
    p = PRICES_USD_PER_MTOK.get(model)
    if p is None:
        raise ValueError(f"no recorded price for {model!r}; refusing to guess")
    return n_calls * (est_input_tokens / 1e6 * p["in"]
                      + max_output_tokens / 1e6 * p["out"])


class BudgetExceeded(RuntimeError):
    pass


def check_budget(model: str, n_calls: int, est_input_tokens: int,
                 max_usd: float, max_output_tokens: int = MAX_OUTPUT_TOKENS) -> float:
    """Refuse to start a run that cannot be afforded. Control 2 of 4."""
    est = worst_case_usd(model, n_calls, est_input_tokens, max_output_tokens)
    if est > max_usd:
        raise BudgetExceeded(
            f"worst case ${est:.4f} for {n_calls} call(s) on {model} exceeds "
            f"the --max-usd limit of ${max_usd:.4f}. Nothing was sent."
        )
    return est


def log_usage(record: dict) -> None:
    """Append the real usage reported by the API. Control 3 of 4."""
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with USAGE_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def actual_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICES_USD_PER_MTOK[model]
    return input_tokens / 1e6 * p["in"] + output_tokens / 1e6 * p["out"]
