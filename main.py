
"""CLI entrypoint for rag_v2."""

from src.rag_v2.app_main import bootstrap_query_engine_v2
from llama_index.core.schema import QueryBundle


def main() -> None:
    qe = bootstrap_query_engine_v2()
    questions = [
        "What is a DAT on Solana?",
        "Return videos about Firedancer",
    ]

    for q in questions:
        print(f"\n=== {q}\n")
        resp = qe.query(QueryBundle(q))
        print(resp)


if __name__ == "__main__":
    main()
