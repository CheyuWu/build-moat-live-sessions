import os

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import indexer


MAX_QUESTION_LEN = 512

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


def build_prompt(query: str, ranked_sections: list[tuple[indexer.Section, float]]) -> str:
    context_parts = []
    for section, _ in ranked_sections:
        heading_path = " > ".join(section.heading_path)
        content = section.content.strip()
        context_parts.append(
            f"[Source: {section.id}]\n"
            f"Path: {heading_path}\n\n"
            f"{content}"
        )

    context = "\n\n---\n\n".join(context_parts)
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"


def query(question: str) -> dict:
    question = question.strip()
    if len(question) > MAX_QUESTION_LEN:
        return {"answer": f"Question is too long (max {MAX_QUESTION_LEN} characters).", "sources": []}
    if "[Source:" in question:
        return {"answer": "Invalid question format.", "sources": []}

    if not indexer.sections:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    ranked_sections = indexer.search(question, k=3)
    if not ranked_sections:
        return {
            "answer": "I cannot confirm from the knowledge base.",
            "sources": [],
        }

    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(question, ranked_sections)),
    ])

    sources = [
        {
            "source": section.id,
            "heading": " > ".join(section.heading_path),
            "score": round(score, 3),
            "content": section.content[:240],
        }
        for section, score in ranked_sections
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }
