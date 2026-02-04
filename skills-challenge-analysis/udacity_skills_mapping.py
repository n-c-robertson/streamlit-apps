"""
Udacity Skills API → Taxonomy mapping.

Maps a free-text skill (e.g. from Workera) to Udacity skill name, then to
skill URI/ID, then walks the hierarchy to get subject (parent of skill) and
domain (ancestor below root). Used to build domain > subject > skill views
without a taxonomy CSV.
"""

from __future__ import annotations

import requests
from settings import (
    SKILLS_SEARCH_SCOPED_URL,
    UTAXONOMY_GRAPHQL_URL,
    SKILLS_NODES_RELATED_URL,
)

# Root node labels – stop climbing when we hit these so we don't use them as "domain"
ROOT_LABELS = frozenset({"udacity domain", "root"})
MAX_HIERARCHY_LEVELS = 5


def _headers(jwt_token: str) -> dict:
    return {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json",
    }


def convert_skill_to_udacity_skill(skill_description: str, jwt_token: str, scope: str | None = None) -> str | None:
    """
    Map a Workera/free-text skill to a canonical Udacity skill name using the Skills Search API.

    Args:
        skill_description: Raw skill text (e.g. "Identify the skewness of data").
        jwt_token: Udacity JWT for API auth.
        scope: Optional scope for the search.

    Returns:
        Udacity skill name (e.g. "Skewness") or None if no match.
    """
    payload = {
        "search": [skill_description],
        "searchField": "knowledge_component_names",
        "filter": {},
        "limit": 1,
        "deduplicate": False,
        "scopeLimit": 5,
    }
    if scope is not None:
        payload["scope"] = scope

    try:
        resp = requests.post(
            SKILLS_SEARCH_SCOPED_URL,
            headers=_headers(jwt_token),
            json=payload,
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    groups = data.get("groups") or {}
    if not groups:
        return None

    # Pick the group with the highest score (first match in response is typically best)
    best_name = None
    best_score = -1.0
    for name, matches in groups.items():
        if isinstance(matches, list) and matches:
            score = matches[0].get("score", 0) if isinstance(matches[0], dict) else 0
        else:
            score = 0
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def search_udacity_skills(query: str, jwt_token: str, limit: int = 20) -> list[dict]:
    """
    Search Udacity taxonomy for skills (Udaciskill) by name for type-ahead.
    Returns list of {"name": displayName, "uri": uri}.
    """
    if not (query or query.strip()):
        return []
    q = query.strip().replace('"', '\\"').replace("\n", " ")
    gql = (
        'query { topics(input: { typeUri: "model:Udaciskill", nameSearch: "%s", limit: %d, offset: 0 }) { displayName uri } }'
        % (q, limit)
    )
    try:
        resp = requests.post(
            UTAXONOMY_GRAPHQL_URL,
            headers=_headers(jwt_token),
            json={"query": gql},
            timeout=15,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
    except Exception:
        return []
    topics = (data.get("data") or {}).get("topics") or []
    return [{"name": t.get("displayName") or "", "uri": t.get("uri") or ""} for t in topics if t.get("displayName")]


def get_skill_hierarchy(skill_name: str, jwt_token: str) -> dict | None:
    """
    Resolve Udacity skill name to skill URI/ID, then get subject (parent) and domain (ancestor below root).

    Returns:
        Dict with keys: skill_id, skill_name, skill_uri, subject_id, subject_name, domain_id, domain_name.
        Or None if the skill cannot be resolved or has no hierarchy.
    """
    # Step 2: Skill name → skill URI/ID (Taxonomy GraphQL)
    query = (
        'query { topics(input: { typeUri: "model:Udaciskill", nameSearch: "%s", limit: 1, offset: 0 }) { displayName uri } }'
        % (skill_name.replace('"', '\\"').replace("\n", " "))
    )
    try:
        resp = requests.post(
            UTAXONOMY_GRAPHQL_URL,
            headers=_headers(jwt_token),
            json={"query": query},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    topics = (data.get("data") or {}).get("topics") or []
    if not topics:
        return None
    topic = topics[0]
    display_name = topic.get("displayName") or skill_name
    uri = topic.get("uri")
    if not uri:
        return None
    skill_id = uri.replace("taxonomy:", "") if isinstance(uri, str) else None
    if not skill_id:
        return None

    result = {
        "skill_id": skill_id,
        "skill_name": display_name,
        "skill_uri": uri,
        "subject_id": None,
        "subject_name": None,
        "domain_id": None,
        "domain_name": None,
    }

    # Step 3: Parent of skill = subject
    try:
        parent_resp = requests.get(
            f"{SKILLS_NODES_RELATED_URL}/{skill_id}/dir/in",
            headers=_headers(jwt_token),
            timeout=15,
        )
        if parent_resp.status_code != 200:
            return result
        parent_data = parent_resp.json()
    except Exception:
        return result

    nodes = parent_data.get("nodes") or []
    if not nodes:
        return result
    parent = nodes[0]
    parent_id = parent.get("id")
    parent_label = (parent.get("content") or {}).get("label") or ""
    result["subject_id"] = parent_id
    result["subject_name"] = parent_label

    # Step 4: Climb from subject to domain (stop at root)
    current_id = parent_id
    domain_id = parent_id
    domain_name = parent_label
    level = 0
    while current_id and level < MAX_HIERARCHY_LEVELS:
        try:
            in_resp = requests.get(
                f"{SKILLS_NODES_RELATED_URL}/{current_id}/dir/in",
                headers=_headers(jwt_token),
                timeout=15,
            )
            if in_resp.status_code != 200:
                break
            in_data = in_resp.json()
        except Exception:
            break
        in_nodes = in_data.get("nodes") or []
        if not in_nodes:
            break
        next_parent = in_nodes[0]
        next_id = next_parent.get("id")
        next_label = (next_parent.get("content") or {}).get("label") or ""
        if next_label.lower().strip() in ROOT_LABELS:
            break
        domain_id = next_id
        domain_name = next_label
        current_id = next_id
        level += 1

    result["domain_id"] = domain_id
    result["domain_name"] = domain_name
    return result


def batch_convert_skills_to_udacity(skill_descriptions: list[str], jwt_token: str) -> dict[str, str | None]:
    """
    Map each raw skill to Udacity skill name. Returns dict: raw_skill -> udacity_skill_name or None.
    """
    out = {}
    for raw in skill_descriptions:
        if raw in out:
            continue
        out[raw] = convert_skill_to_udacity_skill(raw, jwt_token)
    return out


def batch_get_skill_hierarchies(udacity_skill_names: list[str], jwt_token: str) -> dict[str, dict]:
    """
    For each unique Udacity skill name, get hierarchy (subject, domain). Returns dict: udacity_skill_name -> hierarchy record.
    """
    out = {}
    for name in udacity_skill_names:
        if not name or name in out:
            continue
        h = get_skill_hierarchy(name, jwt_token)
        if h is not None:
            out[name] = h
    return out


def build_taxonomy_from_hierarchies(hierarchies: dict[str, dict]) -> list[dict]:
    """
    Build taxonomy rows: Skill (Udacity), Subject, Domain. Workera skill is added per-row in exploded data.
    """
    rows = []
    for skill_name, h in hierarchies.items():
        subject = h.get("subject_name") or ""
        domain = h.get("domain_name") or ""
        if subject or domain:
            rows.append({
                "Skill": skill_name,
                "Subject": subject,
                "Domain": domain,
            })
    return rows
