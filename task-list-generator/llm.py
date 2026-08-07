"""OpenAI LLM: synthesize a project task list from rubric + retrieved content,
then refine it in chat."""
from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI
from rubric import flatten_rubric, rubric_to_text

CHAT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a senior Udacity curriculum engineer. You convert a
project rubric into an ordered task list a student must complete to finish the
project and pass the review.

Rules:
- The RUBRIC is the primary source of truth. Every task must trace to one or
  more rubric criteria. Cite the criterion text in each task's `rubric_criteria`.
- The SUPPORTING CONTENT is secondary: use it to make tasks concrete and
  actionable, but never invent requirements that are not in the rubric.
- Tasks must be SEQUENTIAL and ordered the way a student would actually do them
  (setup -> implementation -> verification -> submission).
- Keep tasks student-facing and imperative. Each task needs a short title and a
  1-3 sentence description plus optional tips.
- If a rubric criterion is optional/exceedable, mark it but do not make it a
  required step; put exceedance in `tips`.

Output STRICT JSON only (no markdown fences), matching this schema:
{
  "tasks": [
    {
      "step": 1,
      "title": "...",
      "description": "...",
      "rubric_criteria": ["criterion text", "..."],
      "tips": "..."
    }
  ]
}
"""

REFINE_SYSTEM_PROMPT = """You are refining an existing ordered task list for a
Udacity project. Apply the user's feedback. Keep the rubric as the source of
truth: do not drop required criteria unless the user explicitly asks. Re-emit
the FULL updated task list as STRICT JSON with the same schema as before
({"tasks": [...]}). Preserve step numbering starting at 1. No markdown fences."""


def _client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def _extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of an LLM response, tolerating code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Find the outermost balanced object as a fallback.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def _format_context(retrieved: list[dict[str, Any]]) -> str:
    if not retrieved:
        return "(no supporting content retrieved)"
    lines = []
    for r in retrieved:
        tag = r.get("type", "content")
        lines.append(f"- [{tag}] {r.get('text','')[:1200]}")
    return "\n".join(lines)


def synthesize_tasklist(
    api_key: str,
    *,
    project: dict[str, Any],
    retrieved: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return {"tasks": [...]} parsed from the LLM."""
    rubric_text = rubric_to_text(project.get("rubric"), project_title=project.get("title", ""))
    criteria_count = len(flatten_rubric(project.get("rubric")))
    user_prompt = (
        f"PROJECT TITLE: {project.get('title','')}\n"
        f"RUBRIC ({criteria_count} criteria):\n{rubric_text or '(no rubric available)'}\n\n"
        f"SUPPORTING CLASSROOM CONTENT (secondary):\n{_format_context(retrieved)}\n\n"
        "Produce the ordered task list JSON now."
    )
    resp = _client(api_key).chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return _extract_json(resp.choices[0].message.content or "{}")


def refine_tasklist(
    api_key: str,
    *,
    project: dict[str, Any],
    current_tasks: list[dict[str, Any]],
    feedback: str,
    history: list[dict[str, str]],
) -> dict[str, Any]:
    rubric_text = rubric_to_text(project.get("rubric"), project_title=project.get("title", ""))
    messages = [
        {"role": "system", "content": REFINE_SYSTEM_PROMPT},
        {"role": "system", "content": f"RUBRIC:\n{rubric_text or '(no rubric)'}"},
        {"role": "user", "content": f"Current task list JSON:\n{json.dumps({'tasks': current_tasks})}"},
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": f"Feedback: {feedback}\nEmit the updated task list JSON."})
    resp = _client(api_key).chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
    )
    return _extract_json(resp.choices[0].message.content or "{}")
