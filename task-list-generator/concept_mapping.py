"""Per-task concept assignment for synthesized task lists.

Decouples the "start here" concept pointer from the single holistic retrieval
query used during task synthesis. That holistic query biases concept picks
toward a project's own (often coarse) concepts and collapses many tasks onto
one concept. Instead, we embed each task individually and search the program's
concept_rollup chunks across the whole corpus to find the most relevant
granular teaching concept per task.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from openai import OpenAI

import semantic_search

# Concept titles that are structural scaffolding (exercises, quizzes, summaries,
# welcomes, project instructions, coarse phase buckets) rather than teaching
# material. We avoid pointing students to these as a "start here" concept when a
# topical teaching concept is available.
_STRUCTURAL_PREFIXES = (
    "exercise:", "solution:", "quiz:", "summary", "welcome",
    "meet your", "knowledge check", "prerequisites", "good luck",
    "congratulations", "course review", "skill scorecard",
    # Project-instruction / coarse structural concepts, not teaching material:
    "instructions", "project overview", "project instructions",
    "project:", "course project", "phase 1", "phase 2", "phase ",
)


def is_structural_concept(title: str) -> bool:
    t = (title or "").strip().lower()
    return any(t.startswith(p) for p in _STRUCTURAL_PREFIXES)


def assign_concept_per_task(
    index: semantic_search.CorpusIndex, qvec: np.ndarray
) -> dict[str, Any] | None:
    """Pick the best teaching concept for a task via semantic search over
    concept_rollup chunks across the whole corpus (no project_key filter).

    Prefers topical teaching concepts over structural scaffolding
    (exercises/quizzes/summaries) so the "start here" pointer lands on a
    teachable concept rather than an exercise or quiz.
    """
    hits = index.search(qvec, k=20)  # no project_key -> whole corpus
    rollups = [h for h in hits if h.get("type") == "concept_rollup"]
    if not rollups:
        return None
    teaching = [h for h in rollups if not is_structural_concept(h.get("concept_title", ""))]
    pool = teaching or rollups
    return pool[0]


def assign_concepts(
    tasks: list[dict[str, Any]],
    *,
    client: OpenAI,
    index: semantic_search.CorpusIndex | None,
    catalog: list[dict[str, str]],
    program_key: str,
    concept_url_fn,
) -> list[dict[str, Any]]:
    """Assign each task a best-fit teaching concept via per-task semantic search
    over the program's concept_rollup chunks (whole corpus, not project-scoped).

    The LLM's concept_key pick is kept only as a fallback when semantic search
    finds nothing. Always populates concept_title + concept_url on each task.
    """
    catalog_by_key = {c["key"]: c for c in (catalog or [])}

    # Build one query per task (title + description + rubric criteria) and
    # batch-embed so we pay one embeddings round-trip instead of N.
    queries = [
        " ".join(
            filter(
                None,
                [
                    t.get("title", ""),
                    t.get("description", ""),
                    " ".join(t.get("rubric_criteria") or []),
                ],
            )
        )
        for t in tasks
    ]
    qvecs = None
    if index is not None and queries:
        try:
            qvecs = semantic_search.embed_queries(client, queries)
        except Exception:
            qvecs = None

    for i, t in enumerate(tasks):
        best = None
        if qvecs is not None and qvecs.ndim == 2 and qvecs.shape[0] > i:
            try:
                best = assign_concept_per_task(index, qvecs[i])
            except Exception:
                best = None
        if best:
            t["concept_key"] = best.get("concept_key") or ""
            t["concept_title"] = best.get("concept_title", "")
        else:
            # Fallback: trust the LLM's pick if it is a valid catalog key.
            ck = t.get("concept_key")
            if ck and ck in catalog_by_key:
                t["concept_key"] = ck
                t["concept_title"] = catalog_by_key[ck].get("title", "")
            else:
                t["concept_key"] = t.get("concept_key") or ""
                t["concept_title"] = t.get("concept_title") or ""
        if t.get("concept_key"):
            t["concept_url"] = concept_url_fn(program_key, t["concept_key"])
        else:
            t["concept_url"] = ""
    return tasks
