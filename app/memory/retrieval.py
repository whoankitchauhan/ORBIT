"""RAG helpers layered on top of the vector store.

Two distinct things live here, and the distinction matters at viva time:

*   **Retrieval (RAG)** — fetching documents to ground a single answer.
*   **Episodic memory** — writing the outcome of a finished task back so that
    later tasks can recall what this system has already done.

Both use the same vector store; they differ in what gets written and when.
"""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.memory.vector import MemoryRecord, get_memory


def retrieve_context(
    query: str, k: int | None = None, kinds: list[str] | None = None
) -> list[MemoryRecord]:
    """Fetch the most relevant memories for a query.

    `kinds` narrows retrieval to a slice of memory — passing ["policy"] when
    checking eligibility stops a stale episode from outranking the rule that
    actually governs the decision.
    """
    memory = get_memory()
    k = k or settings.retrieval_top_k
    if not kinds:
        return memory.search(query, k=k)

    results: list[MemoryRecord] = []
    for kind in kinds:
        results.extend(memory.search(query, k=k, where={"kind": kind}))
    results.sort(key=lambda r: r.score, reverse=True)

    deduped: list[MemoryRecord] = []
    seen: set[str] = set()
    for record in results:
        if record.id not in seen:
            seen.add(record.id)
            deduped.append(record)
    return deduped[:k]


def build_rag_prompt(question: str, records: list[MemoryRecord]) -> str:
    """Assemble retrieved chunks into a grounded prompt.

    Sources are numbered so the agent can cite them and a reviewer can trace
    any claim back to the chunk it came from.
    """
    if not records:
        return (
            f"Question: {question}\n\n"
            "No supporting documents were retrieved. Say so explicitly rather "
            "than answering from general knowledge."
        )
    blocks = [
        f"[{i + 1}] (source: {r.source}, relevance: {r.score:.2f})\n{r.text}"
        for i, r in enumerate(records)
    ]
    return (
        "Answer using only the context below. Cite sources as [1], [2]. "
        "If the context does not answer the question, say so.\n\n"
        f"--- CONTEXT ---\n{chr(10).join(blocks)}\n--- END CONTEXT ---\n\n"
        f"Question: {question}"
    )


def remember_task_outcome(state: Any) -> str | None:
    """Write a finished task into long-term memory as an episode."""
    if not state.final_answer:
        return None
    summary = (
        f"Task {state.task_id}: {state.objective}\n"
        f"Outcome ({state.status}, confidence {state.confidence:.0%}): "
        f"{state.final_answer[:600]}"
    )
    return get_memory().add(
        summary,
        source=f"task:{state.task_id}",
        kind="episode",
        metadata={
            "task_id": state.task_id,
            "status": state.status,
            "risk": state.risk,
            "user_id": state.user_id,
        },
    )


def recall_similar_tasks(objective: str, k: int = 3) -> list[MemoryRecord]:
    """Find earlier tasks resembling this one, for the Supervisor's context."""
    return get_memory().search(objective, k=k, where={"kind": "episode"})
