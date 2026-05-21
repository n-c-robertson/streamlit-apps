"""Standalone Streamlit app that walks the classroom-content GraphQL queries
the assessment-creator uses and shows the raw responses at every step.

Drop-in disposable. Delete this whole `gql-debug/` folder when you're done.

Run locally:
    cd ~/streamlit-apps/gql-debug
    pip install -r requirements.txt
    streamlit run streamlit_app.py

You will be prompted for the JWT in the UI. Nothing is persisted to disk.

What it does, in order, for an ND key + locale:

  Step 1.  components(key:)            -> enumerate every locale the key exists in
  Step 2.  nanodegree(key:, locale:)   -> old simple crosswalk shape
                                          (matches the previous query_nd_parts_by_key
                                          that worked before consolidation)
  Step 3.  nanodegree(key:, locale:)   -> NEW deep query that query_nd_full sends
                                          (modules > lessons > concepts > atoms)
                                          and ALSO the broken
                                          branch.component.metadata path
  Step 4.  For each part returned by step 3:
             component(key:, locale:)  -> per-part metadata fetch
                                          (skills + difficulty)
             node(id:)                 -> per-part content tree by node id
                                          (the cd-key flow's add_node_data call)

Every step prints request payload, HTTP status, GraphQL errors[], and the
raw data section. Designed so you can see at a glance exactly which step
returns null / errors / partial data.
"""

import hashlib
import json
from typing import Any, Optional

import requests
import streamlit as st


CLASSROOM_CONTENT_API_URL = (
    "https://api.udacity.com/api/classroom-content/v1/graphql"
)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

COMPONENTS_PROBE = """
query DebugComponentsProbe($key: String!) {
  components(key: $key, count: 50) {
    id
    key
    locale
    type
    deprecated
    latest_release {
      root_node_id
      major
      minor
      patch
    }
  }
}
"""

ND_SIMPLE = """
query DebugNDSimple($key: String!, $locale: String) {
  nanodegree(key: $key, locale: $locale) {
    id
    key
    locale
    version
    title
    semantic_type
    parts {
      id
      key
      locale
      version
      title
      semantic_type
    }
  }
}
"""

# Mirrors what query_nd_full currently asks for in production (commit efad01a),
# PLUS the parts.branch.component.metadata selection that was dropped because
# Part.branch has no field resolver. We include it here so you can see GraphQL's
# actual response (it should resolve as null, not error out — but if it ever
# does error, we want to see why).
ND_FULL = """
query DebugNDFull($key: String!, $locale: String) {
  nanodegree(key: $key, locale: $locale) {
    id
    key
    locale
    version
    title
    semantic_type
    parts {
      id
      key
      locale
      version
      title
      semantic_type
      branch {
        id
        component {
          id
          key
          locale
          metadata {
            difficulty_level { name uri }
            teaches_skills { name uri }
            prerequisite_skills { name uri }
          }
        }
      }
      modules {
        key
        locale
        version
        semantic_type
        title
        lessons {
          key
          locale
          version
          semantic_type
          title
          concepts {
            key
            locale
            version
            semantic_type
            title
            progress_key
            atoms {
              ... on TextAtom {
                key locale version semantic_type title text
              }
              ... on VideoAtom {
                key locale version semantic_type title
                video { vtt_url }
              }
              ... on RadioQuizAtom {
                semantic_type
                question {
                  prompt
                  answers { is_correct text }
                }
              }
              ... on CheckboxQuizAtom {
                semantic_type
                question {
                  prompt
                  correct_feedback
                  answers { is_correct text }
                }
              }
              ... on MatchingQuizAtom {
                semantic_type
                question {
                  answers_label
                  concepts_label
                  concepts {
                    text
                    correct_answer { text }
                  }
                  complex_prompt { text }
                  answers { text }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""

COMPONENT_FOR_KEY = """
query DebugComponentForKey($key: String!, $locale: String!) {
  component(key: $key, locale: $locale) {
    id
    key
    locale
    type
    latest_release {
      major
      minor
      patch
      root_node_id
      type
      component {
        metadata {
          difficulty_level { name uri }
          teaches_skills { name uri }
          prerequisite_skills { name uri }
        }
      }
      root_node {
        id
        key
        locale
        version
        title
      }
    }
  }
}
"""

NODE_BY_ID = """
query DebugNodeById($id: Int!) {
  node(id: $id) {
    ... on Part {
      id
      key
      locale
      version
      title
      semantic_type
      modules {
        key
        title
        lessons {
          key
          title
          concepts {
            key
            title
            atoms { __typename }
          }
        }
      }
    }
  }
}
"""


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def gql(jwt: str, query: str, variables: dict) -> dict:
    """Run a GraphQL query and return a structured result dict.

    Never raises; always returns a dict with at least `ok`, `status`,
    `body` keys so the UI can render uniformly.
    """
    headers = {
        "Authorization": f"Bearer {jwt}",
        "Content-Type": "application/json",
    }
    payload = {"query": query, "variables": variables}
    out: dict[str, Any] = {
        "payload": payload,
        "ok": False,
        "status": None,
        "body": None,
        "errors": None,
        "data": None,
        "raw_text": None,
        "exception": None,
    }
    try:
        resp = requests.post(
            CLASSROOM_CONTENT_API_URL,
            headers=headers,
            json=payload,
            timeout=60,
        )
        out["status"] = resp.status_code
        out["raw_text"] = resp.text
        try:
            body = resp.json()
            out["body"] = body
            out["data"] = body.get("data")
            out["errors"] = body.get("errors")
        except Exception as e:
            out["exception"] = f"non-JSON response: {type(e).__name__}: {e}"
        out["ok"] = resp.status_code == 200 and not out["errors"]
    except Exception as e:
        out["exception"] = f"{type(e).__name__}: {e}"
    return out


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def render_result(label: str, result: dict, query: str) -> None:
    """Render one GraphQL call's result in a uniform panel."""
    status = result.get("status")
    errors = result.get("errors")
    exception = result.get("exception")

    if exception:
        st.error(f"{label}: request failed - {exception}")
    elif status != 200:
        st.error(f"{label}: HTTP {status}")
    elif errors:
        st.error(f"{label}: GraphQL returned {len(errors)} error(s) (HTTP 200)")
    else:
        st.success(f"{label}: HTTP {status}, no GraphQL errors")

    with st.expander("Request"):
        st.code(query.strip(), language="graphql")
        st.json(result["payload"]["variables"])

    if errors:
        st.markdown("**GraphQL errors:**")
        st.json(errors)

    st.markdown("**Response `data`:**")
    if result.get("data") is None:
        st.warning("`data` is null")
    else:
        st.json(result["data"])

    if exception:
        with st.expander("Raw response text"):
            st.text(result.get("raw_text") or "<empty>")


def coerce_part_keys(nd_full_data: Optional[dict]) -> list[dict]:
    """Pull out parts[] from a (possibly partial) ND query response."""
    if not nd_full_data:
        return []
    nd = nd_full_data.get("nanodegree")
    if not nd:
        return []
    return [p for p in (nd.get("parts") or []) if p]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="classroom-content GraphQL debugger",
        layout="wide",
    )
    st.title("classroom-content GraphQL debugger")
    st.caption(
        "Walks every query the assessment-creator runs and shows raw "
        "responses at each step. Disposable, delete the gql-debug/ folder "
        "when done."
    )

    with st.sidebar:
        st.header("Inputs")
        jwt = st.text_input(
            "JWT (Bearer token)",
            type="password",
            help="Paste the same JWT the streamlit-cloud secret has under jwt_token.",
        )
        nd_key = st.text_input("ND or CD key", value="nd1827")
        locale = st.text_input("Locale", value="en-us")
        max_parts = st.number_input(
            "Max parts to probe individually",
            min_value=1,
            max_value=20,
            value=3,
            help=(
                "Step 4 calls component(key:, locale:) and node(id:) once per "
                "part. Cap it so a 10-part ND doesn't fire 20 round-trips."
            ),
        )
        run = st.button("Run all queries", type="primary", use_container_width=True)

    if not run:
        st.info(
            "Enter a JWT + key in the sidebar and click **Run all queries**. "
            "No JWT is persisted; it lives only in this Streamlit session."
        )
        return

    if not jwt:
        st.error("JWT is required.")
        return
    if not nd_key:
        st.error("ND or CD key is required.")
        return

    # JWT fingerprint - same algorithm the assessment-creator uses for its
    # deployed-environment banner. If these two fingerprints DO NOT match,
    # the hosted app is using a different JWT and the debug-vs-hosted
    # mismatch is fully explained by that secrets difference.
    jwt_fp = hashlib.sha256(jwt.encode('utf-8')).hexdigest()[:10]
    st.caption(
        f"JWT fingerprint (sha256[:10]): `{jwt_fp}` - compare to the "
        "fingerprint shown on the deployed assessment-creator's banner."
    )

    # ---- Step 1 ---------------------------------------------------------
    st.header("Step 1 - components(key:) cross-locale probe")
    st.caption(
        "Not locale-gated. Returns every Component row matching the key, "
        "so we can see which locales the program actually exists in."
    )
    step1 = gql(jwt, COMPONENTS_PROBE, {"key": nd_key})
    render_result("components(key:)", step1, COMPONENTS_PROBE)

    components = ((step1.get("data") or {}).get("components")) or []
    if components:
        st.markdown(
            f"**Found {len(components)} Component row(s) across "
            f"{len({c.get('locale') for c in components})} locale(s).**"
        )
    else:
        st.warning(
            "No Components found. Either the key doesn't exist, was deleted, "
            "or the JWT can't see it."
        )

    # ---- Step 1b --------------------------------------------------------
    st.header(
        "Step 1b - component(key:, locale:) on the INPUT key "
        "(ND-level metadata probe)"
    )
    st.caption(
        "Smoking-gun check: if this returns a Component with "
        "latest_release.component.metadata populated, then the ND itself "
        "carries the difficulty / skills metadata and we should use *this* "
        "result for every section derived from the ND - NOT try to call "
        "component(key: <part_uuid>) per part, which can never work because "
        "Part.key is a node UUID, not a Component cd* key."
    )
    step1b = gql(jwt, COMPONENT_FOR_KEY, {"key": nd_key, "locale": locale})
    render_result(
        f"component(key:{nd_key!r}, locale:{locale!r})",
        step1b,
        COMPONENT_FOR_KEY,
    )
    comp_data = ((step1b.get("data") or {}).get("component")) or {}
    nd_meta = (
        ((comp_data.get("latest_release") or {}).get("component") or {})
        .get("metadata")
    )
    if nd_meta:
        st.success(
            "ND-level metadata IS available on the input key's Component. "
            "Fix path: stop per-part component() calls; use this once and "
            "apply to all sections."
        )
    elif comp_data:
        st.warning(
            "Input key resolves to a Component, but latest_release."
            "component.metadata is null. Metadata may live somewhere else "
            "(e.g. Studio program metadata) for this program."
        )
    else:
        st.warning(
            "Input key did NOT resolve via component(key:, locale:). If "
            "Step 1 found components in other locales, retry Step 1b with "
            "one of those locales."
        )

    # ---- Step 2 ---------------------------------------------------------
    st.header("Step 2 - nanodegree(key:, locale:) shallow")
    st.caption(
        "Same selection set the old query_nd_parts_by_key used. If this "
        "returns parts but Step 3 doesn't, the issue is in the deeper "
        "selection (modules / atoms / branch.component.metadata)."
    )
    step2 = gql(jwt, ND_SIMPLE, {"key": nd_key, "locale": locale})
    render_result(
        f"nanodegree(key:{nd_key!r}, locale:{locale!r}) [shallow]",
        step2,
        ND_SIMPLE,
    )

    # ---- Step 3 ---------------------------------------------------------
    st.header("Step 3 - nanodegree(key:, locale:) FULL (production shape + branch path)")
    st.caption(
        "Mirrors what query_nd_full sends in production (commit efad01a) "
        "and ALSO requests parts.branch.component.metadata so you can see "
        "what that traversal really returns. Expected: branch resolves to "
        "null because Part.branch has no field resolver. If you see errors "
        "instead, this is the row that's breaking the ND path."
    )
    step3 = gql(jwt, ND_FULL, {"key": nd_key, "locale": locale})
    render_result(
        f"nanodegree(key:{nd_key!r}, locale:{locale!r}) [full]",
        step3,
        ND_FULL,
    )

    parts = coerce_part_keys(step3.get("data"))
    if not parts and step2.get("data"):
        # Fall back to the shallow query's parts list if step 3 fully failed.
        parts = coerce_part_keys(step2.get("data"))
        if parts:
            st.info(
                "Step 3 returned no parts; falling back to Step 2's parts list "
                "for the per-part probes in Step 4."
            )

    # ---- Step 4 ---------------------------------------------------------
    st.header("Step 4 - per-part component + node lookups")
    if not parts:
        st.warning(
            "No parts available from Step 2 or Step 3, skipping Step 4."
        )
        return

    st.caption(
        "For each part, run component(key:, locale:=part.locale) (this is "
        "what add_program_data calls for metadata) and node(id:=part.id) "
        "(what add_node_data used to call before the consolidated fetch). "
        f"Capped at {max_parts} part(s)."
    )

    for i, part in enumerate(parts[: int(max_parts)], start=1):
        part_key = part.get("key")
        part_locale = part.get("locale") or locale
        part_id = part.get("id")

        st.subheader(
            f"Part {i}/{min(int(max_parts), len(parts))} - "
            f"key={part_key!r}, locale={part_locale!r}, id={part_id!r}"
        )

        s4a = gql(
            jwt,
            COMPONENT_FOR_KEY,
            {"key": part_key, "locale": part_locale},
        )
        render_result(
            f"component(key:{part_key!r}, locale:{part_locale!r})",
            s4a,
            COMPONENT_FOR_KEY,
        )

        if part_id is None:
            st.warning(
                f"Part {part_key!r} has no id in the prefetched payload; "
                "skipping node(id:) lookup."
            )
            continue

        s4b = gql(jwt, NODE_BY_ID, {"id": int(part_id)})
        render_result(
            f"node(id:{part_id})",
            s4b,
            NODE_BY_ID,
        )

    st.success("All steps complete.")


if __name__ == "__main__":
    main()
