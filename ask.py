"""Ask a question and get an answer built only from the retrieved passages.

This is the chatbot, minus the chat. One question, one answer, every claim
carrying the document and page it came from, and a refusal when the passages do
not support an answer.

    python ask.py "how many attention heads does the model use" --max-usd 0.01
"""

from __future__ import annotations

import argparse
import textwrap

import common
import generate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question", nargs="+")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--backend", default=common.DEFAULT_BACKEND,
                    choices=common.EMBEDDING_BACKENDS)
    ap.add_argument("--model", default=common.DEFAULT_MODEL)
    ap.add_argument("--threshold", type=float, default=None,
                    help="retrieval score below which the question is declined "
                         "without calling the model, and therefore for free")
    ap.add_argument("--max-usd", type=float, default=0.02,
                    help="refuse to send anything whose worst case exceeds this")
    ap.add_argument("--max-tokens", type=int, default=common.MAX_OUTPUT_TOKENS)
    args = ap.parse_args()
    question = " ".join(args.question)

    passages = generate.retrieve(question, args.k, args.backend, args.threshold)

    print(f"question   {question}")
    print(f"retrieval  {args.backend}, k={args.k}, threshold="
          f"{args.threshold if args.threshold is not None else 'none'}")
    print(f"model      {args.model}")

    # Control 4. Nothing was retrieved above the threshold, so there is nothing
    # to answer from, so no request is made and no money is spent.
    if not passages:
        print("\nNo passage met the threshold. Declining without calling the model.")
        print("Cost of this answer: $0.000000")
        return 0

    prompt = generate.build_prompt(question, passages)
    est_in = common.estimate_input_tokens(generate.SYSTEM + prompt)

    # Control 2. Decide before spending, and stop rather than warn.
    try:
        est = common.check_budget(args.model, 1, est_in, args.max_usd, args.max_tokens)
    except common.BudgetExceeded as e:
        print(f"\nBUDGET STOP: {e}")
        return 2
    print(f"budget     worst case ${est:.6f}, limit ${args.max_usd:.6f}  ok")

    print("\nretrieved")
    for i, ps in enumerate(passages, start=1):
        print(f"  [{i}] {ps.score:.4f}  {ps.source} p.{ps.page} chunk {ps.chunk_index}")

    model = generate.get_model(args.model)
    answer = model(question, passages, max_tokens=args.max_tokens)

    print("\nanswer")
    print(textwrap.indent(textwrap.fill(answer.text, width=86), "  "))
    if answer.abstained:
        print("\n  (the model declined: the passages did not support an answer)")
    if answer.input_tokens:
        print(f"\nusage      {answer.input_tokens} in / {answer.output_tokens} out"
              f"   ${answer.usd:.6f}   logged to {common.USAGE_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
