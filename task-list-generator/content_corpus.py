"""Walk a classroom-content program tree into text chunks for semantic search."""
from __future__ import annotations

from typing import Any


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split())


def _atom_text(atom: dict[str, Any]) -> str:
    """Extract readable text from a single atom node."""
    semantic = atom.get("semantic_type") or atom.get("__typename") or ""
    title = atom.get("title") or ""
    parts: list[str] = []
    if title:
        parts.append(title)
    if atom.get("text"):
        parts.append(_clean(atom["text"]))

    question = atom.get("question")
    if isinstance(question, dict):
        prompt = _clean(question.get("prompt"))
        if prompt:
            parts.append(prompt)
        # complex_prompt (MatchingQuiz) and correct_feedback are extra context.
        cp = question.get("complex_prompt")
        if isinstance(cp, dict) and cp.get("text"):
            parts.append(_clean(cp["text"]))
        cf = question.get("correct_feedback")
        if cf:
            parts.append(_clean(cf))
        answers = question.get("answers")
        if isinstance(answers, list):
            ans_texts = [_clean(a.get("text")) for a in answers if isinstance(a, dict) and a.get("text")]
            if ans_texts:
                parts.append("Answers: " + " | ".join(ans_texts))
        concepts = question.get("concepts")
        if isinstance(concepts, list):
            con_texts = [_clean(c.get("text")) for c in concepts if isinstance(c, dict) and c.get("text")]
            if con_texts:
                parts.append("Concepts: " + " | ".join(con_texts))

    video = atom.get("video")
    if isinstance(video, dict) and video.get("vtt_url"):
        parts.append(f"(video transcript: {video['vtt_url']})")

    if not parts:
        return ""
    prefix = f"[{semantic}] " if semantic else ""
    return prefix + " | ".join(parts)


def _concept_chunks(concept: dict[str, Any], *, project_key: str | None) -> list[dict[str, Any]]:
    concept_title = concept.get("title") or ""
    concept_key = concept.get("key") or ""
    chunks: list[dict[str, Any]] = []
    for atom in concept.get("atoms") or []:
        text = _atom_text(atom)
        if not text:
            continue
        chunks.append(
            {
                "project_key": project_key,
                "concept_key": concept_key,
                "concept_title": concept_title,
                "atom_key": atom.get("key"),
                "type": atom.get("semantic_type") or atom.get("__typename") or "atom",
                "text": f"{concept_title}: {text}" if concept_title else text,
            }
        )
    # One roll-up chunk per concept so a concept is retrievable even if its atoms
    # are individually thin.
    rollup_parts = [c["text"] for c in chunks]
    if rollup_parts:
        chunks.append(
            {
                "project_key": project_key,
                "concept_key": concept_key,
                "concept_title": concept_title,
                "atom_key": None,
                "type": "concept_rollup",
                "text": f"{concept_title}: " + " ".join(rollup_parts),
            }
        )
    return chunks


def _lesson_chunks(lesson: dict[str, Any], *, parent_project_key: str | None) -> list[dict[str, Any]]:
    lesson_title = lesson.get("title") or ""
    lesson_key = lesson.get("key") or ""
    project = lesson.get("project")
    project_key = (project or {}).get("key") or parent_project_key
    chunks: list[dict[str, Any]] = []

    summary = _clean(lesson.get("summary"))
    if summary:
        chunks.append(
            {
                "project_key": project_key,
                "concept_key": None,
                "concept_title": lesson_title,
                "atom_key": None,
                "type": "lesson_summary",
                "text": f"Lesson {lesson_title}: {summary}",
            }
        )

    if project:
        proj_title = project.get("title") or lesson_title
        proj_desc = _clean(project.get("description")) or _clean(project.get("summary"))
        if proj_desc:
            chunks.append(
                {
                    "project_key": project_key,
                    "concept_key": None,
                    "concept_title": proj_title,
                    "atom_key": None,
                    "type": "project_description",
                    "text": f"Project {proj_title}: {proj_desc}",
                }
            )

    for concept in lesson.get("concepts") or []:
        chunks.extend(_concept_chunks(concept, project_key=project_key))
    return chunks


def build_concept_catalog(chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Unique list of {key, title} for every concept in the program.

    Built from concept_rollup chunks (one per concept), which carry
    concept_key + concept_title. Used as the search space the LLM picks a
    best-fit teaching concept from for each task.
    """
    seen: set[str] = set()
    catalog: list[dict[str, str]] = []
    for c in chunks or []:
        if c.get("type") != "concept_rollup":
            continue
        key = c.get("concept_key")
        title = c.get("concept_title") or ""
        if not key or key in seen:
            continue
        seen.add(key)
        catalog.append({"key": key, "title": title})
    return catalog


def build_corpus(program: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (chunks, projects).

    chunks: flat list of text chunks for semantic search.
    projects: list of {key, title, rubric_id, rubric, lesson_key, lesson_title}.

    Shape-based root detection (mirrors the reference app): if `parts[]` is
    populated the root is Nanodegree-shaped; if `modules[]` is populated it is
    Part-shaped (cd root); otherwise treat it as a Lesson root. We do NOT trust
    the `semantic_type` string because it has drifted over time ('Degree' vs
    'Nanodegree').
    """
    chunks: list[dict[str, Any]] = []
    projects: list[dict[str, Any]] = []

    # Program-level overview. summary is on Nanodegree/Part/Lesson; syllabus_overview
    # is Nanodegree-only (only present when the root is a Nanodegree).
    overview = _clean(program.get("summary")) or _clean(program.get("syllabus_overview"))
    if overview:
        chunks.append(
            {
                "project_key": None,
                "concept_key": None,
                "concept_title": program.get("title") or "",
                "atom_key": None,
                "type": "program_overview",
                "text": f"{program.get('title')}: {overview}",
            }
        )

    def _register_project(project: dict[str, Any] | None, lesson_key: str, lesson_title: str):
        if not project:
            return
        projects.append(
            {
                "key": project.get("key"),
                "title": project.get("title") or lesson_title,
                "rubric_id": project.get("rubric_id"),
                "rubric": project.get("rubric"),
                "lesson_key": lesson_key,
                "lesson_title": lesson_title,
            }
        )

    def _walk_lessons(lessons: list[dict[str, Any]], parent_project_key: str | None = None):
        for lesson in lessons or []:
            _register_project(lesson.get("project"), lesson.get("key") or "", lesson.get("title") or "")
            chunks.extend(_lesson_chunks(lesson, parent_project_key=parent_project_key))

    def _walk_modules(modules: list[dict[str, Any]]):
        for module in modules or []:
            _walk_lessons(module.get("lessons"))

    nd_parts = [p for p in (program.get("parts") or []) if p and p.get("key")]
    part_modules = program.get("modules") or []

    if nd_parts:
        # Nanodegree-shaped root.
        for part in nd_parts:
            _walk_modules(part.get("modules"))
    elif part_modules:
        # Part-shaped root (cd key).
        _walk_modules(part_modules)
    elif program.get("concepts") or program.get("project"):
        # Lesson-shaped root.
        _register_project(program.get("project"), program.get("key") or "", program.get("title") or "")
        chunks.extend(_lesson_chunks(program, parent_project_key=(program.get("project") or {}).get("key")))
    else:
        # Last-resort fallback: try lessons directly.
        _walk_lessons(program.get("lessons") or [])

    # Deduplicate projects by key (a course may surface the same project twice).
    seen: set[str] = set()
    unique_projects: list[dict[str, Any]] = []
    for p in projects:
        k = p.get("key") or p.get("title")
        if k in seen:
            continue
        seen.add(k)
        unique_projects.append(p)
    return chunks, unique_projects
