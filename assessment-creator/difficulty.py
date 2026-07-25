"""Difficulty-level normalization helpers.

The Assessments API is the single source of truth for difficulty levels.
This module centralizes:

  * fetching the canonical level set (cached),
  * mapping between the three ways a level is referenced across the
    pipeline — the API `id` (UUID), the `externalId` (URI, e.g.
    `udacity://difficulty/beginner`), and the human-readable `label`
    ("Beginner"),
  * computing the "one step easier" level used by readiness assessments,
  * normalizing arbitrary LLM-emitted labels back to a canonical label.

The rest of the assessment-creator package should NEVER trust a
free-form `difficultyLevelId` string produced by the LLM. The URI is
authoritative; `difficulty_id_for_uri` is the only correct way to turn
a URI into a `difficultyLevelId`. Use `normalize_difficulty_label` to
clean up labels for display when only a label is available.
"""

import streamlit as st


# ---------------------------------------------------------------------------
# Canonical vocabulary
# ---------------------------------------------------------------------------
#
# The canonical level names, ordered easiest -> hardest. This mirrors the
# `label` field returned by the Assessments API `difficultyLevels` query.
# It is defined here as a constant so callers can reference a stable order
# (e.g. for UI selects) without an extra API round-trip, but the API remains
# the source of truth for ids/URIs.
CANONICAL_ORDER = ["Discovery", "Fluency", "Beginner", "Intermediate", "Advanced"]

# "One step easier" mapping used by readiness assessments. Readiness
# questions test prerequisite knowledge, so they sit one rung below the
# content's own difficulty. Note: the previous readiness prompt used the
# non-existent string "Discovery/Fluency" — that was a bug; Fluency is the
# single step below Beginner, and Discovery is the step below Fluency.
ONE_STEP_EASIER = {
    "Advanced": "Intermediate",
    "Intermediate": "Beginner",
    "Beginner": "Fluency",
    "Fluency": "Discovery",
    "Discovery": "Discovery",  # already the floor
}


# ---------------------------------------------------------------------------
# Fetch + lookup tables
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_difficulty_levels():
    """Fetch all difficulty levels from the Assessments API (cached).

    Returns a list of dicts with keys: id, externalId, label, labelValue,
    category, status. Mirrors the query used by the upload page and by
    `utils_assessment_generation.fetch_difficulty_levels`.
    """
    import requests
    import settings

    query = """
    query {
      difficultyLevels {
        id
        externalId
        label
        labelValue
        category
        status
      }
    }
    """
    r = requests.post(
        settings.ASSESSMENTS_API_URL,
        headers=settings.production_headers(),
        json={"query": query},
    )
    return r.json()["data"]["difficultyLevels"]


def _lookups():
    """Return (by_uri, by_label) dicts mapping identifier -> level dict.

    Labels are keyed by their lowercased, stripped form so lookups are
    case-insensitive. Both dicts are rebuilt on every call; callers should
    cache the result for the duration of a loop if making many lookups.
    """
    levels = fetch_difficulty_levels()
    by_uri = {dl["externalId"]: dl for dl in levels if dl.get("externalId")}
    by_label = {
        dl["label"].strip().lower(): dl for dl in levels if dl.get("label")
    }
    return by_uri, by_label


# ---------------------------------------------------------------------------
# Public mapping helpers
# ---------------------------------------------------------------------------
def difficulty_id_for_uri(uri):
    """The authoritative URI -> difficultyLevelId (API UUID) mapping.

    This is the only mapping that should be used to populate
    `difficultyLevelId` for persistence. Returns '' if the URI is empty
    or not found in the API (so callers can detect and reject the row).
    """
    if not uri:
        return ""
    by_uri, _ = _lookups()
    dl = by_uri.get(uri)
    return dl["id"] if dl else ""


def difficulty_uri_for_label(label):
    """Map a canonical label (e.g. 'Beginner') to its API externalId (URI).

    Used by the readiness flow, which must derive the *stepped-down* level's
    URI from a label. Returns '' if the label is unknown.
    """
    if not label:
        return ""
    _, by_label = _lookups()
    dl = by_label.get(label.strip().lower())
    return dl.get("externalId", "") if dl else ""


def difficulty_id_for_label(label):
    """Map a canonical label directly to its API id. Returns '' if unknown."""
    if not label:
        return ""
    _, by_label = _lookups()
    dl = by_label.get(label.strip().lower())
    return dl["id"] if dl else ""


def normalize_difficulty_label(label):
    """Map any junky/free-form label back to a canonical label.

    Returns the canonical label (e.g. 'Beginner') if the input matches a
    known level case-insensitively, otherwise ''. Use this for display
    so the Review UI never shows LLM-emitted noise like 'Discovery/Fluency'
    or 'Hard'.
    """
    if not label:
        return ""
    _, by_label = _lookups()
    dl = by_label.get(label.strip().lower())
    return dl["label"] if dl else ""


def canonical_labels():
    """Return canonical labels in easiest->hardest order.

    Falls back to the API's own label set (sorted) if the API is reachable
    at call time; otherwise returns the hardcoded CANONICAL_ORDER. Used to
    populate UI selects so they stay in sync with the API instead of
    drifting (the Analyze page previously hardcoded only 3 of the 5 levels).
    """
    try:
        levels = fetch_difficulty_levels()
        api_labels = [dl["label"] for dl in levels if dl.get("label")]
        if api_labels:
            # Order by CANONICAL_ORDER where known, then append any extras.
            ordered = [l for l in CANONICAL_ORDER if l in api_labels]
            ordered += [l for l in api_labels if l not in ordered]
            return ordered
    except Exception:
        pass
    return list(CANONICAL_ORDER)


def readiness_step_down(label):
    """Return the canonical label one step easier than `label`.

    Used by the readiness flow to compute the target difficulty for
    prerequisite-knowledge questions. Case-insensitive. Falls back to the
    input label (stripped) if it is not in the canonical ladder, so unknowns
    are passed through rather than silently dropped.
    """
    if not label:
        return ""
    key = label.strip()
    # Case-insensitive match against the canonical ladder.
    for canon, easier in ONE_STEP_EASIER.items():
        if key.lower() == canon.lower():
            return easier
    return key
