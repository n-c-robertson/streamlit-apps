"""Parse and flatten classroom-content / reviews-api rubrics."""
from __future__ import annotations

from typing import Any


def flatten_rubric(rubric: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return a flat list of rubric items with section context.

    Each item: {section, position, criteria, passed_description,
    exceeded_description, optional, exceedable, reviewer_tips}.
    """
    if not rubric:
        return []
    items: list[dict[str, Any]] = []
    for section in rubric.get("sections") or []:
        section_name = section.get("name") or "(untitled section)"
        for item in section.get("rubric_items") or []:
            items.append(
                {
                    "section": section_name,
                    "position": item.get("position"),
                    "criteria": item.get("criteria") or "",
                    "passed_description": item.get("passed_description") or "",
                    "exceeded_description": item.get("exceeded_description") or "",
                    "optional": bool(item.get("optional")),
                    "exceedable": bool(item.get("exceedable")),
                    "reviewer_tips": item.get("reviewer_tips") or "",
                }
            )
    items.sort(key=lambda x: (x["section"], x["position"] or 0))
    return items


def rubric_to_text(rubric: dict[str, Any] | None, *, project_title: str = "") -> str:
    """Render a rubric as compact text for the LLM prompt."""
    items = flatten_rubric(rubric)
    if not items:
        return ""
    lines = []
    if project_title:
        lines.append(f"PROJECT: {project_title}")
    current_section = None
    for it in items:
        if it["section"] != current_section:
            current_section = it["section"]
            lines.append(f"\n## Section: {current_section}")
        flags = []
        if it["optional"]:
            flags.append("optional")
        if it["exceedable"]:
            flags.append("exceedable")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        lines.append(f"- CRITERIA: {it['criteria']}{flag_str}")
        if it["passed_description"]:
            lines.append(f"    PASS: {it['passed_description']}")
        if it["exceeded_description"]:
            lines.append(f"    EXCEED: {it['exceeded_description']}")
        if it["reviewer_tips"]:
            lines.append(f"    TIP: {it['reviewer_tips']}")
    return "\n".join(lines)
