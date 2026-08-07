"""Client for classroom-content GraphQL and reviews-api REST (production).

Branching mirrors the assessment-creator reference app exactly:

  ND keys (`nd*`):
    1. components(key:) -> pick locale (prefer en-us)   [nanodegree(key:) is locale-strict]
    2. component(key, locale) -> root_node_id            [construction-branch fallback]
    3. node(root_node_id) -> Nanodegree (parts[]) or Part (modules[])

  CD keys (`cd*` / other):
    1. component(key, "en-us") -> root_node_id            [fast path: most cds are en-us]
    2. components(key:) -> pick released locale -> component(key, locale)  [fallback]
    3. node(root_node_id) -> Part (modules[]) or Lesson
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import requests

# Production endpoints (per architecture.md and udacity-mcp endpoints.ts).
CLASSROOM_CONTENT_GRAPHQL = "https://api.udacity.com/api/classroom-content/v1/graphql"
REVIEWS_RUBRIC = "https://api.udacity.com/api/reviews/v1/rubrics/{rubric_id}"

_TIMEOUT = 60
DEFAULT_LOCALE = "en-us"

ND_KEY_PATTERN = re.compile(r"^nd", re.IGNORECASE)


class UdacityAPIError(RuntimeError):
    pass


def is_nd_key(key: str) -> bool:
    return bool(ND_KEY_PATTERN.match((key or "").strip()))


def _auth_headers(jwt: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {jwt}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _load_query() -> str:
    return (Path(__file__).parent / "queries.graphql").read_text()


def _gql(jwt: str, operation: str, variables: dict[str, Any]) -> dict[str, Any]:
    document = _load_query()
    resp = requests.post(
        CLASSROOM_CONTENT_GRAPHQL,
        headers=_auth_headers(jwt),
        json={"query": document, "operationName": operation, "variables": variables},
        timeout=_TIMEOUT,
    )
    if resp.status_code in (401, 403):
        raise UdacityAPIError(
            f"classroom-content HTTP {resp.status_code}: staff JWT invalid/expired/revoked. "
            f"Refresh UDACITY_JWT. Preview: {resp.text[:200]!r}"
        )
    if not resp.ok:
        raise UdacityAPIError(f"classroom-content HTTP {resp.status_code}: {resp.text[:500]}")
    body = resp.json()
    if body.get("errors"):
        raise UdacityAPIError(f"classroom-content GraphQL errors: {body['errors']}")
    return body.get("data") or {}


# ---- Component / construction resolution ---- #


def _components_by_key(jwt: str, key: str) -> list[dict[str, Any]]:
    return _gql(jwt, "ComponentsByKey", {"key": key}).get("components") or []


def _component_release(jwt: str, key: str, locale: str) -> dict[str, Any] | None:
    component = _gql(jwt, "ComponentByKey", {"key": key, "locale": locale}).get("component")
    if not component:
        return None
    release = component.get("latest_release")
    if not release or not release.get("root_node_id"):
        return None
    return release


def _construction_release(jwt: str, key: str, locale: str) -> dict[str, Any] | None:
    """Fallback for keys with no published RELEASE branch (unreleased/draft)."""
    data = _gql(jwt, "ConstructionByKey", {"key": key, "locale": locale})
    node = data.get("node")
    if not node or not node.get("id"):
        return None
    return {
        "root_node_id": node.get("id"),
        "root_node": {"id": node.get("id"), "title": node.get("title")},
        "_unreleased": True,
    }


# ---- Locale pickers (mirror _pick_nd_locale / _pick_released_locale) ---- #


def _pick_nd_locale(components: list[dict[str, Any]], requested: str = DEFAULT_LOCALE) -> str | None:
    """ND locale pick: prefer requested -> en-us -> first non-deprecated -> first."""
    if not components:
        return None
    available = [c.get("locale") for c in components if c.get("locale")]
    available_set = set(available)
    non_deprecated = [c.get("locale") for c in components if c.get("locale") and not c.get("deprecated")]
    if requested in available_set:
        return requested
    if DEFAULT_LOCALE in available_set:
        return DEFAULT_LOCALE
    if non_deprecated:
        return non_deprecated[0]
    return available[0]


def _pick_released_locale(components: list[dict[str, Any]]) -> str | None:
    """CD locale pick: prefer a locale that has a release, then en-us, then non-deprecated."""
    if not components:
        return None
    has_release = lambda c: bool((c.get("latest_release") or {}).get("root_node_id"))
    pool = [c for c in components if has_release()] or components
    chosen = (
        next((c for c in pool if c.get("locale") == DEFAULT_LOCALE), None)
        or next((c for c in pool if not c.get("deprecated")), None)
        or pool[0]
    )
    return chosen.get("locale")


def _root_id_from(release: dict[str, Any]) -> int | None:
    return release.get("root_node_id") or (release.get("root_node") or {}).get("id")


# ---- Public API ---- #


def _resolve_root_node_id(jwt: str, key: str) -> tuple[int, str]:
    """Return (root_node_id, locale), branching on nd vs cd exactly like the reference."""
    if is_nd_key(key):
        components = _components_by_key(jwt, key)
        if not components:
            raise UdacityAPIError(
                f"components(key:{key!r}) returned 0 rows - the key does not exist as a "
                "Component in classroom-content, or the JWT lacks visibility."
            )
        locale = _pick_nd_locale(components)
        if not locale:
            raise UdacityAPIError(f"No locale found for ND key {key!r}.")
        release = _component_release(jwt, key, locale)
        if not release:
            release = _construction_release(jwt, key, locale)
        if not release:
            raise UdacityAPIError(
                f"ND key {key!r} exists in locales "
                f"{sorted({c.get('locale') for c in components if c.get('locale')})} "
                f"(picked {locale!r}) but has no latest_release or CONSTRUCTION branch."
            )
        root_id = _root_id_from(release)
        if root_id is None:
            raise UdacityAPIError(f"ND key {key!r}: release has no root_node_id.")
        return int(root_id), locale

    # cd path: try en-us first (one round-trip), then enumerate.
    release = _component_release(jwt, key, DEFAULT_LOCALE)
    if release:
        return int(_root_id_from(release)), DEFAULT_LOCALE
    components = _components_by_key(jwt, key)
    locale = _pick_released_locale(components)
    if not locale or locale == DEFAULT_LOCALE:
        # No en-us release and no other released locale; try construction in en-us.
        release = _construction_release(jwt, key, DEFAULT_LOCALE)
        if release and _root_id_from(release) is not None:
            return int(_root_id_from(release)), DEFAULT_LOCALE
        raise UdacityAPIError(
            f"No published release found for cd key {key!r} in any locale. "
            "The key may be invalid, unreleased, or the JWT lacks visibility."
        )
    release = _component_release(jwt, key, locale)
    if not release or _root_id_from(release) is None:
        raise UdacityAPIError(f"cd key {key!r} resolved to locale {locale!r} but has no release.")
    return int(_root_id_from(release)), locale


def fetch_program(program_key: str, jwt: str) -> dict[str, Any]:
    """Resolve an nd/cd key to its root node and fetch the full content tree."""
    root_node_id, locale = _resolve_root_node_id(jwt, program_key)
    node = _gql(jwt, "NodeById", {"id": root_node_id}).get("node")
    if not node:
        raise UdacityAPIError(f"node(id:{root_node_id}) returned null for key {program_key!r}.")
    node["_kind"] = node.get("semantic_type") or "Unknown"
    node["_resolved_locale"] = locale
    return node


def fetch_rubric_rest(rubric_id: str, jwt: str) -> dict[str, Any]:
    """Fallback: fetch a rubric directly from reviews-api with full contents."""
    url = REVIEWS_RUBRIC.format(rubric_id=rubric_id)
    resp = requests.get(
        url,
        headers=_auth_headers(jwt),
        params={"projection": "with_project_and_contents"},
        timeout=_TIMEOUT,
    )
    if not resp.ok:
        raise UdacityAPIError(f"reviews-api HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.json()
