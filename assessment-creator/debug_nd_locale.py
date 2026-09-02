"""Standalone diagnostic for an ND key that returns no content.

Usage from the assessment-creator dir. The app no longer reads the JWT from
``st.secrets``; it is entered per session via the sidebar and stored in
``st.session_state['udacity_staff_jwt']``. To run via Streamlit:

    streamlit run debug_nd_locale.py nd029-ent-vfgermany

(Enter your JWT in the sidebar first, or export UDACITY_JWT before running.)

Or, if you have UDACITY_JWT exported:

    python debug_nd_locale.py nd029-ent-vfgermany

Probes `components(key:)` (which is NOT locale-gated) to enumerate every
locale the key exists in, then tries `nanodegree(key:, locale:)` against
each locale so we can see exactly which one returns a Nanodegree with
parts. This is the canonical way to find out why an ND lookup is silently
returning null.
"""

import json
import os
import sys

import requests

CLASSROOM_CONTENT_API_URL = (
    "https://api.udacity.com/api/classroom-content/v1/graphql"
)


def _headers(jwt):
    return {
        "Authorization": f"Bearer {jwt}",
        "Content-Type": "application/json",
    }


def _gql(jwt, query, variables):
    resp = requests.post(
        CLASSROOM_CONTENT_API_URL,
        headers=_headers(jwt),
        json={"query": query, "variables": variables},
        timeout=30,
    )
    try:
        body = resp.json()
    except Exception:
        body = {"_raw": resp.text}
    return resp.status_code, body


COMPONENTS_PROBE = """
query NDLocaleProbe_Components($key: String!) {
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

NANODEGREE_PROBE = """
query NDLocaleProbe_Nanodegree($key: String!, $locale: String) {
  nanodegree(key: $key, locale: $locale) {
    id
    key
    locale
    version
    title
    semantic_type
    parts {
      key
      title
      semantic_type
    }
  }
}
"""


def probe(nd_key, jwt):
    print(f"\n=== components(key:{nd_key!r}) ===")
    status, body = _gql(jwt, COMPONENTS_PROBE, {"key": nd_key})
    print(f"HTTP {status}")
    if body.get("errors"):
        print("GraphQL errors:", json.dumps(body["errors"], indent=2))
    components = ((body.get("data") or {}).get("components")) or []
    if not components:
        print(
            "No Components found with this key in ANY locale. Either the key "
            "is misspelled, the ND was deleted, or the JWT lacks visibility."
        )
        return

    print(f"Found {len(components)} component row(s):")
    for c in components:
        rel = c.get("latest_release") or {}
        version = (
            f"{rel.get('major')}.{rel.get('minor')}.{rel.get('patch')}"
            if rel
            else "(no release)"
        )
        print(
            f"  - id={c.get('id')} locale={c.get('locale')!r} "
            f"type={c.get('type')!r} deprecated={c.get('deprecated')} "
            f"latest_release={version} root_node_id={rel.get('root_node_id')}"
        )

    locales_to_try = sorted({c.get("locale") for c in components if c.get("locale")})

    print(f"\n=== nanodegree(key:{nd_key!r}, locale:...) per locale ===")
    for locale in locales_to_try:
        status, body = _gql(jwt, NANODEGREE_PROBE, {"key": nd_key, "locale": locale})
        nd = (body.get("data") or {}).get("nanodegree")
        if body.get("errors"):
            errs = "; ".join(e.get("message", "") for e in body["errors"])
            print(f"  locale={locale!r:10s}  HTTP {status}  errors: {errs}")
            continue
        if not nd:
            print(f"  locale={locale!r:10s}  HTTP {status}  nanodegree=null")
            continue
        parts = nd.get("parts") or []
        print(
            f"  locale={locale!r:10s}  HTTP {status}  "
            f"semantic_type={nd.get('semantic_type')!r}  "
            f"title={nd.get('title')!r}  version={nd.get('version')!r}  "
            f"parts={len(parts)}"
        )
        if parts:
            for p in parts:
                print(
                    f"      - part key={p.get('key')!r} "
                    f"semantic_type={p.get('semantic_type')!r} "
                    f"title={p.get('title')!r}"
                )


def main():
    nd_key = sys.argv[1] if len(sys.argv) > 1 else "nd029-ent-vfgermany"

    jwt = os.environ.get("UDACITY_JWT")
    if not jwt:
        try:
            import streamlit as st  # streamlit run path

            # The app no longer stores the JWT in st.secrets; it is entered per
            # session via the sidebar. Fall back to session state if present.
            jwt = st.session_state.get("udacity_staff_jwt") if hasattr(st, "session_state") else None
            if jwt:
                st.write(f"Probing ND key: `{nd_key}`")
            else:
                print(
                    "No JWT found. Set the UDACITY_JWT env var, or paste a "
                    "staff JWT into the app sidebar (which stores it in "
                    "st.session_state['udacity_staff_jwt']) and run via "
                    "`streamlit run debug_nd_locale.py`."
                )
                sys.exit(1)
        except Exception:
            print(
                "No JWT found. Set UDACITY_JWT env var or run via "
                "`streamlit run debug_nd_locale.py` after entering a JWT in "
                "the app sidebar."
            )
            sys.exit(1)

    probe(nd_key, jwt)


if __name__ == "__main__":
    main()
