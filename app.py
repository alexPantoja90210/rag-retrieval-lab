"""Streamlit front end for the chat. IA-159.

    streamlit run app.py

What this shows that the reference does not, and it is the whole point.

The fork returns prose. A reader has no way to tell an answer built from the
corpus from one the model remembered, which is the defect this repository was
built to measure: its prompt even instructs the model not to mention where the
answer came from.

Every turn here shows the passages it used with their source, page and score,
the citation verdict, the rewritten query when there was one, and the running
spend. When the corpus does not support an answer it says so.

All the measurement in eval/results/ is the reason to believe what is on this
screen. Until now it lived in a JSON nobody outside this project would open.
"""

from __future__ import annotations

import streamlit as st

import chat
import common

st.set_page_config(page_title="Grounded RAG chat", layout="wide")


def _new_session():
    st.session_state.chat = chat.ChatSession(
        model=st.session_state.model,
        backend=st.session_state.backend,
        k=st.session_state.k,
        threshold=st.session_state.threshold or None,
        max_usd=st.session_state.max_usd,
    )


with st.sidebar:
    st.header("Settings")
    st.selectbox("Model", [common.DEFAULT_MODEL, "stub-echo"], key="model",
                 help="stub-echo is not a model. It hands back the top "
                      "passage with its citation, so the gate, the passage "
                      "viewer and the budget can be exercised with no API key "
                      "and nothing spent. It answers nothing.\n\n"
                      "The evaluator's `stub` and `stub-memoriser` are not "
                      "offered here: they decide what to say by comparing "
                      "expected pages, and a chat has no ground truth, so "
                      "they could only ever decline. IA-162.")
    st.selectbox("Embeddings", list(common.EMBEDDING_BACKENDS), key="backend")
    st.slider("Passages (k)", 1, 10, 5, key="k")
    st.slider("Score threshold", 0.0, 1.0, 0.0, 0.05, key="threshold",
              help="0 disables it. Above it, a question whose best passage "
                   "scores lower is declined before the API is called, so it "
                   "costs nothing.")
    st.number_input("Session budget (USD)", 0.01, 5.0, 0.25, 0.05,
                    key="max_usd")

    # A widget that changes nothing is a widget that lies. The first version
    # of this file built the session once and never rebuilt it, so switching
    # Model to `stub` left the conversation running on the paid model while
    # the sidebar said otherwise, and a turn cost $0.000222 under a label that
    # promised zero. IA-161.
    #
    # Model and embeddings cannot change inside a conversation: the answers
    # already on screen came from a different system, and keeping them beside
    # new ones would make the transcript unreadable as evidence. Those rebuild.
    # k, threshold and the budget are per-turn settings and apply live.
    identity = (st.session_state.model, st.session_state.backend)
    if "chat" not in st.session_state or st.session_state.get("identity") != identity:
        rebuilt = "chat" in st.session_state
        _new_session()
        st.session_state.identity = identity
        if rebuilt:
            st.info("Model or embeddings changed, so the conversation was "
                    "restarted. The answers above came from a different "
                    "system and cannot be continued by this one.")
    if st.button("New conversation"):
        _new_session()
        st.rerun()

    s = st.session_state.chat
    s.k = st.session_state.k
    s.threshold = st.session_state.threshold or None
    s.max_usd = st.session_state.max_usd

    # Read back off the session object, not off the widgets. What the settings
    # say and what the session is doing are two different facts, and this
    # repository has spent three days on the difference.
    st.divider()
    st.caption("**Actually running**")
    st.caption(f"model `{s.model_name}` · embeddings `{s.backend}` · "
               f"k={s.k} · threshold="
               f"{s.threshold if s.threshold is not None else 'off'}")

    # Which store, and how much is in it. Shown always, not only when
    # something goes wrong. RAG_CHROMA_PATH redirects every entry point in
    # this repository and an empty store is indistinguishable from a corpus
    # with no matches: both return zero passages and the interface used to
    # blame a threshold that was switched off. An afternoon went into that.
    # IA-163.
    try:
        _st = common.get_store(s.backend)
        _n = _st._collection.count()
        _where = _st._client.get_settings().persist_directory
    except Exception as _e:
        _n, _where = -1, f"unreadable: {_e}"
    st.caption(f"store `{_where}` · **{_n} vectors**")

    # Said before a question is typed, not after it silently does nothing.
    if s.missing_key():
        st.error(
            f"`{s.model_name}` needs ANTHROPIC_API_KEY and the shell that "
            f"started this app does not have it. Questions will be refused "
            f"without sending anything. Set it and restart, or switch Model "
            f"to `stub-echo`."
        )
    if _n == 0:
        st.error(
            f"That store is empty, so every question will be declined for "
            f"lack of passages and it will not be the model's fault. Check "
            f"RAG_CHROMA_PATH in the shell that started this app, then run "
            f"`python ingest.py --backend {s.backend}`."
        )
    st.metric("Spent this session", f"${s.spent:.6f}")
    st.caption(f"${s.remaining:.6f} left of ${s.max_usd:.2f}. A turn whose "
               f"worst case does not fit is refused before anything is sent.")

st.title("Ask the corpus, and see what it answered from")
st.caption(
    "Every claim carries the document and page it came from, or the answer is "
    "refused. An assertion the corpus cannot be shown to support is not "
    "returned, whatever the model produced."
)

s = st.session_state.chat

for t in s.turns:
    with st.chat_message("user"):
        st.write(t.question)
        if t.rewritten:
            st.caption(f"retrieved on: {t.query}")
            st.caption(
                "The follow-up was rewritten to stand alone before retrieval. "
                "It is shown because a rewrite is where a model can quietly "
                "insert a detail the corpus never contained."
            )
    with st.chat_message("assistant"):
        if t.note and not t.answer:
            st.warning(t.note)
        elif not t.accepted:
            st.error(t.note)
            with st.expander("What the model produced, which the system did not return"):
                st.write(t.answer)
        elif t.abstained:
            st.info(t.answer)
            st.caption("Declined. The passages did not support an answer, and "
                       "declining is the correct behaviour rather than a failure.")
        else:
            st.write(t.answer)
            st.caption(f"citations accepted: {t.citation_reason}")

        if t.passages:
            with st.expander(f"The {len(t.passages)} passages this answer could use"):
                for i, ps in enumerate(t.passages, start=1):
                    st.markdown(f"**[{i}]** `{ps.source}` p.{ps.page} "
                                f"chunk {ps.chunk_index} score {ps.score:.4f}")
                    st.text(" ".join(ps.text.split())[:700])
        st.caption(f"this turn cost ${t.usd:.6f}")

if q := st.chat_input("Ask something about the corpus"):
    with st.spinner("retrieving and answering"):
        s.ask(q)
    st.rerun()
