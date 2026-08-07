"""Streamlit app: cd/nd key -> program + rubrics -> synthesized task list -> chat refine."""
from __future__ import annotations

import csv
import io
import json
from typing import Any

import streamlit as st
from openai import OpenAI

import content_corpus
import llm
import rubric
import semantic_search
import udacity_client

st.set_page_config(page_title="Udacity Rubric -> Task List", page_icon=":memo:", layout="wide")

# --------------------------------------------------------------------------- #
# Secrets / config
# --------------------------------------------------------------------------- #


def _secrets() -> tuple[str, str]:
    try:
        s = st.secrets
        api_key = s["OPENAI_API_KEY"]
        jwt = s["UDACITY_JWT"]
    except (KeyError, FileNotFoundError):
        api_key = ""
        jwt = ""
    if not api_key or not jwt:
        st.error(
            "Missing secrets. Copy `.streamlit/secrets.toml.example` to "
            "`.streamlit/secrets.toml` and fill in OPENAI_API_KEY and UDACITY_JWT."
        )
        st.stop()
    return api_key, jwt


OPENAI_API_KEY, UDACITY_JWT = _secrets()
_oa = OpenAI(api_key=OPENAI_API_KEY)

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

if "program" not in st.session_state:
    st.session_state.program = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "projects" not in st.session_state:
    st.session_state.projects = None
if "index" not in st.session_state:
    st.session_state.index = None
if "tasklists" not in st.session_state:
    st.session_state.tasklists = {}  # project_key -> [task, ...]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}  # project_key -> [{"role","content"}, ...]
if "synthesized" not in st.session_state:
    st.session_state.synthesized = {}  # project_key -> bool


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _load_program(key: str) -> None:
    with st.status("Loading program from classroom-content (production)..."):
        program = udacity_client.fetch_program(key, UDACITY_JWT)
        chunks, projects = content_corpus.build_corpus(program)
        with st.spinner("Embedding content for semantic search..."):
            index = semantic_search.build_index(_oa, chunks)
    st.session_state.program = program
    st.session_state.chunks = chunks
    st.session_state.projects = projects
    st.session_state.index = index
    st.session_state.tasklists = {}
    st.session_state.chat_history = {}
    st.session_state.synthesized = {}


def _retrieve(project: dict[str, Any], k: int = 8) -> list[dict[str, Any]]:
    index: semantic_search.CorpusIndex | None = st.session_state.index
    if index is None or not index.chunks:
        return []
    query = (
        f"{project.get('title','')}. "
        + rubric.rubric_to_text(project.get("rubric"), project_title=project.get("title", ""))[:2000]
    )
    qvec = semantic_search.embed_query(_oa, query)
    return index.search(qvec, project_key=project.get("key"), k=k)


def _synthesize(project: dict[str, Any]) -> None:
    retrieved = _retrieve(project)
    result = llm.synthesize_tasklist(OPENAI_API_KEY, project=project, retrieved=retrieved)
    pk = project.get("key") or project.get("title")
    st.session_state.tasklists[pk] = result.get("tasks", [])
    st.session_state.chat_history[pk] = []
    st.session_state.synthesized[pk] = True


def _refine(project: dict[str, Any], feedback: str) -> None:
    pk = project.get("key") or project.get("title")
    current = st.session_state.tasklists.get(pk, [])
    history = st.session_state.chat_history.get(pk, [])
    result = llm.refine_tasklist(
        OPENAI_API_KEY,
        project=project,
        current_tasks=current,
        feedback=feedback,
        history=history,
    )
    st.session_state.tasklists[pk] = result.get("tasks", [])
    st.session_state.chat_history[pk].append({"role": "user", "content": feedback})
    st.session_state.chat_history[pk].append({"role": "assistant", "content": json.dumps(result)})


def _tasks_to_csv(tasks: list[dict[str, Any]]) -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["step", "title", "description", "rubric_criteria", "tips"])
    for i, t in enumerate(tasks, 1):
        crits = "; ".join(t.get("rubric_criteria") or [])
        w.writerow([t.get("step") or i, t.get("title", ""), t.get("description", ""), crits, t.get("tips", "")])
    return out.getvalue()


_TYPE_LABELS = {
    "TextAtom": "text items",
    "VideoAtom": "video transcripts",
    "RadioQuizAtom": "radio quiz items",
    "CheckboxQuizAtom": "checkbox quiz items",
    "MatchingQuizAtom": "matching quiz items",
    "lesson_summary": "lesson summaries",
    "project_description": "project descriptions",
    "concept_rollup": "concept rollups",
    "program_overview": "program overview",
}


def _chunk_breakdown(chunks: list[dict[str, Any]]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for c in chunks or []:
        t = c.get("type") or "other"
        counts[t] = counts.get(t, 0) + 1
    # Order: known types first (by friendly label), then any unknown types.
    labeled = []
    for raw, n in counts.items():
        label = _TYPE_LABELS.get(raw, raw)
        labeled.append((label, n))
    labeled.sort(key=lambda x: (-x[1], x[0]))
    return labeled


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("Udacity Rubric -> Task List")
    st.caption("Production classroom-content + reviews-api. OpenAI for LLM/embeddings.")
    key = st.text_input("cd / nd key", value="", placeholder="e.g. nd006, cd1827")
    if st.button("Load program", type="primary", width="stretch"):
        if not key.strip():
            st.warning("Enter a cd/nd key.")
        else:
            try:
                _load_program(key.strip())
            except udacity_client.UdacityAPIError as e:
                st.error(str(e))
                st.session_state.program = None
    st.divider()
    if st.session_state.program:
        p = st.session_state.program
        st.caption(
            f"Loaded: {p.get('title')} ({p.get('_kind')}, key={p.get('key')}, v{p.get('version')})"
        )
        st.caption(f"{len(st.session_state.projects or [])} projects")
        breakdown = _chunk_breakdown(st.session_state.chunks or [])
        if breakdown:
            st.caption("Content loaded:")
            for label, n in breakdown:
                st.caption(f"· {n} {label}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def _render_rubric(project: dict[str, Any]) -> None:
    items = rubric.flatten_rubric(project.get("rubric"))
    if not items:
        st.info("No rubric attached to this project (it may be hidden or unavailable).")
        return
    rows = [
        {
            "section": it["section"],
            "criteria": it["criteria"],
            "pass": it["passed_description"],
            "exceed": it["exceeded_description"],
            "optional": it["optional"],
            "exceedable": it["exceedable"],
        }
        for it in items
    ]
    st.dataframe(rows, width="stretch", hide_index=True)


def _render_tasklist(project: dict[str, Any]) -> None:
    pk = project.get("key") or project.get("title")
    tasks = st.session_state.tasklists.get(pk, [])
    if not tasks:
        st.info("No task list yet. Click **Synthesize task list** above.")
        return

    view = st.segmented_control(
        "View",
        options=["Card", "Table"],
        default="Card",
        key=f"view-{pk}",
        width="stretch",
    )

    if view == "Table":
        flat = [
            {
                "step": t.get("step") or (i + 1),
                "title": t.get("title", ""),
                "description": t.get("description", ""),
                "rubric_criteria": " | ".join(t.get("rubric_criteria") or []),
                "tips": t.get("tips", ""),
            }
            for i, t in enumerate(tasks)
        ]
        st.dataframe(flat, width="stretch", hide_index=True)
    else:
        for i, t in enumerate(tasks, 1):
            step = t.get("step") or i
            with st.container(border=True):
                st.markdown(f"**Step {step}. {t.get('title','')}**")
                st.write(t.get("description", ""))
                crits = t.get("rubric_criteria") or []
                if crits:
                    st.caption("Rubric criteria: " + " | ".join(crits))
                if t.get("tips"):
                    st.caption(f":bulb: {t['tips']}")

    st.download_button(
        "Download CSV",
        data=_tasks_to_csv(tasks),
        file_name=f"{pk}-tasklist.csv",
        mime="text/csv",
        key=f"csv-{pk}",
    )
    st.download_button(
        "Download JSON",
        data=json.dumps({"tasks": tasks}, indent=2),
        file_name=f"{pk}-tasklist.json",
        mime="application/json",
        key=f"json-{pk}",
    )


def _render_project(project: dict[str, Any]) -> None:
    pk = project.get("key") or project.get("title")
    title = project.get("title") or "(untitled project)"

    with st.container(border=True):
        st.subheader(title)
        st.caption(
            f"key={project.get('key')} · rubric_id={project.get('rubric_id')} · "
            f"reviews_project_id={project.get('reviews_project_id')}"
        )

        st.markdown("**Rubric**")
        _render_rubric(project)

        st.divider()
        st.markdown("**Synthesized task list**")
        if st.button("Synthesize task list", key=f"synth-{pk}", type="primary"):
            with st.spinner("Synthesizing..."):
                _synthesize(project)
            st.rerun()

        # Feedback appears only after the synthesize button has been clicked,
        # below the button and above the result.
        if st.session_state.synthesized.get(pk):
            with st.container(border=True):
                st.markdown("**Refine (feedback)**")
                hist = st.session_state.chat_history.get(pk, [])
                for msg in hist:
                    if msg["role"] == "user":
                        st.chat_message("user").write(msg["content"])
                    else:
                        try:
                            parsed = json.loads(msg["content"])
                            st.chat_message("assistant").write(
                                f"Refined task list updated. {len(parsed.get('tasks', []))} tasks."
                            )
                        except json.JSONDecodeError:
                            st.chat_message("assistant").write(msg["content"])
                fb = st.text_input(
                    "Feedback to refine the task list",
                    key=f"fb-{pk}",
                    placeholder="e.g. split step 2 into setup and test",
                )
                if st.button("Send feedback", key=f"send-{pk}") and fb.strip():
                    with st.spinner("Refining..."):
                        _refine(project, fb.strip())
                    st.rerun()

        st.divider()
        _render_tasklist(project)


def _render_main(program: dict[str, Any]) -> None:
    projects = st.session_state.projects or []
    if not projects:
        st.info("No projects found in this program.")
        return

    st.subheader(f"Projects ({len(projects)})")
    for project in projects:
        _render_project(project)

    with st.expander("Raw program structure (debug)"):
        st.json(
            {
                "title": program.get("title"),
                "key": program.get("key"),
                "semantic_type": program.get("semantic_type"),
                "parts/modules": program.get("parts") or program.get("modules") or program.get("lessons"),
            }
        )


if st.session_state.program:
    _render_main(st.session_state.program)
else:
    st.title("Udacity Rubric -> Task List")
    st.write(
        "Enter a cd/nd key in the sidebar and click **Load program**. "
        "The app fetches the program, its projects, and their rubrics from "
        "production classroom-content, then synthesizes a sequential task list "
        "per project using OpenAI (rubric-first, supporting content second)."
    )
