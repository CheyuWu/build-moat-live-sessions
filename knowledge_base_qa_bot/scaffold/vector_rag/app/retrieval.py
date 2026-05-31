import os

from langchain.schema import Document, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import indexer

SYSTEM_PROMPT = """You are a knowledge base Q&A assistant.
Answer questions using ONLY the provided CONTEXT below.

Rules:
1. Only answer using information found in the CONTEXT.
2. Cite sources using only the exact IDs shown as [Source: filename#heading].
3. If the context does not contain the answer, respond with: "I don't know based on the provided knowledge base." Do not attempt to use any information not in the CONTEXT.
4. Do not fabricate sources. Only use the sources provided in the CONTEXT. If you cannot find the answer in the CONTEXT, do not cite any sources.
"""

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            request_timeout=20,
            max_retries=1,
        )
    return _llm


def build_prompt(query: str, ranked_chunks: list[tuple[Document, float]]) -> str:
    context_parts = []

    for doc, _ in ranked_chunks:
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content

        context_parts.append(f"[Source: {source}]\n\n{content}")

    context_text = (
        "\n\n---\n\n".join(context_parts) if context_parts else "(no context)"
    )

    return f"CONTEXT:\n{context_text}\n\nQUESTION:\n{query}"


def query(question: str) -> dict:
    if indexer.vectorstore is None:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    ranked_chunks = indexer.search(question, k=3)
    if not ranked_chunks:
        return {
            "answer": "I cannot confirm from the knowledge base.",
            "sources": [],
        }

    response = get_llm().invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=build_prompt(question, ranked_chunks)),
        ]
    )

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "heading": doc.metadata.get("heading", "unknown"),
            "score": round(float(score), 3),
            "content": doc.page_content[:240],
        }
        for doc, score in ranked_chunks
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }
