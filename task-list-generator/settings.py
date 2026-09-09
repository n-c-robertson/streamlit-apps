"""Session-state staff JWT handling for the task-list-generator app.

The JWT is never read from Streamlit secrets. Each staff user must paste their
own Udacity staff JWT into the sidebar on first use; it is kept in
``st.session_state`` for the lifetime of the browser session and verified with
a read-only classroom-content query before it is saved. A module-level cache
mirrors it so worker threads (which cannot read ``st.session_state``) still
retrieve the token.
"""
from __future__ import annotations

import base64
import hashlib
import json

import requests
import streamlit as st

import udacity_client

SESSION_STATE_JWT_KEY = "udacity_staff_jwt"

# Module-level mirror of the session JWT. ``st.session_state`` is thread-local
# in Streamlit (bound to the script-run context), so worker threads spawned via
# ``concurrent.futures.ThreadPoolExecutor`` cannot read it. The main thread
# writes the JWT here whenever it (re)reads session state, and worker threads
# fall back to this cache. A single Python string reference is read/written
# atomically under the GIL, so no lock is needed for this one-writer/many-reader
# pattern. Cleared in lock-step with the session value.
_JWT_CACHE: str | None = None


def get_udacity_jwt() -> str | None:
    """Return the staff JWT for the current session, or ``None`` if unset.

    Reads from ``st.session_state`` (the authoritative store, only readable on
    the main Streamlit thread) and mirrors the value into ``_JWT_CACHE`` so
    worker threads — which have no script-run context and cannot access
    ``st.session_state`` — can still retrieve the token via the cache. Falls
    back to the thread-readable cache. There is no secrets fallback: a user
    must paste their own JWT.
    """
    global _JWT_CACHE
    jwt = None
    try:
        jwt = st.session_state.get(SESSION_STATE_JWT_KEY) or None
    except Exception:
        # ``st.session_state`` is only available inside a Streamlit run; module
        # imports outside Streamlit (e.g. unit tests) should not crash here.
        # Also reached inside worker threads that have no script-run context.
        pass
    if jwt:
        _JWT_CACHE = jwt
        return jwt
    return _JWT_CACHE


def clear_udacity_jwt() -> None:
    """Forget the session JWT from both session state and the thread cache.

    Called by ``render_jwt_sidebar`` when the user clears the token.
    """
    global _JWT_CACHE
    _JWT_CACHE = None
    try:
        st.session_state.pop(SESSION_STATE_JWT_KEY, None)
    except Exception:
        pass


def is_jwt_set() -> bool:
    """True if a staff JWT has been entered for this session."""
    return bool(get_udacity_jwt())


def _jwt_fingerprint(jwt_value: str | None) -> str:
    """Short non-reversible hash of a JWT so we can show which token is in use
    without ever displaying the token itself."""
    if not jwt_value:
        return "empty"
    return hashlib.sha256(jwt_value.encode("utf-8")).hexdigest()[:10]


# Minimal classroom-content GraphQL query used purely to validate that a
# candidate JWT is accepted. ``components(key:)`` is argument-light; a bogus key
# still returns HTTP 200 with an empty list when the JWT is valid, so we can
# distinguish auth rejection (401/403) from a valid token without coupling to a
# specific program existing.
_JWT_PROBE_QUERY = """
query ComponentsByKey($key: String!) {
  components(key: $key, count: 1) { id key locale }
}
"""


def verify_jwt(token: str, timeout: int = 15) -> tuple[bool, str]:
    """Validate a candidate staff JWT by making one authenticated read against
    classroom-content.

    Sends the tiny ``ComponentsByKey`` query with the candidate token as the
    Bearer header (NOT the session token, so we can validate before persisting).
    Returns ``(ok, message)`` where ``ok`` is True only when the API accepts the
    token (HTTP 200 with no GraphQL auth error). Used by ``render_jwt_sidebar``
    to reject expired/invalid tokens at save time.
    """
    token = (token or "").strip()
    if not token:
        return False, "JWT is empty."
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    try:
        resp = requests.post(
            udacity_client.CLASSROOM_CONTENT_GRAPHQL,
            headers=headers,
            json={
                "query": _JWT_PROBE_QUERY,
                "operationName": "ComponentsByKey",
                "variables": {"key": "nd900"},
            },
            timeout=timeout,
        )
    except requests.exceptions.RequestException as e:
        return False, f"Could not reach classroom-content to verify the JWT: {e}"
    if resp.status_code in (401, 403):
        return False, (
            f"classroom-content rejected this JWT (HTTP {resp.status_code}). "
            "It is likely expired or not a staff token — grab a fresh one and "
            "try again."
        )
    if resp.status_code != 200:
        preview = (resp.text or "")[:200]
        return False, (
            f"Unexpected HTTP {resp.status_code} from classroom-content while "
            f"verifying the JWT. Preview: {preview!r}"
        )
    try:
        body = resp.json()
    except Exception:
        return False, "classroom-content returned a non-JSON response while verifying the JWT."
    if body.get("errors"):
        return False, "Error: JWT token not accepted. It may be expired — grab a fresh one."
    return True, "JWT verified against classroom-content."


def _jwt_subject(token: str) -> str | None:
    """Best-effort, non-verifying decode of the JWT payload to surface a subject
    hint (uid/email) in the sidebar. Purely informational."""
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode("utf-8"))
        return payload.get("email") or payload.get("sub") or payload.get("uid")
    except Exception:
        return None


def render_jwt_sidebar() -> None:
    """Sidebar widget that lets a staff user paste their own Udacity JWT.

    Call once per page that needs authenticated API access. The token is stored
    in ``st.session_state`` so it persists for the rest of the browser session
    without re-entry. Shows a fingerprint + subject hint once set, and a Clear
    button to wipe it. The token is verified with a read-only classroom-content
    query before it is saved, so expired/invalid tokens are rejected upfront.
    There is no secrets fallback — a user must paste their own JWT.
    """
    with st.sidebar:
        st.markdown("### Udacity staff JWT")
        current = get_udacity_jwt()
        if current:
            st.success(f"JWT set (sha256[:10] `{_jwt_fingerprint(current)}`)")
            subject = _jwt_subject(current)
            if subject:
                st.caption(f"Token subject: `{subject}`")
            if st.button("Clear JWT", use_container_width=True, help="Forget the JWT for this session."):
                clear_udacity_jwt()
                st.rerun()
        else:
            st.info(
                "Paste your **Udacity staff JWT** below. It is stored only in "
                "this browser session (Streamlit session state) and is never "
                "written to disk or secrets."
            )
            new_jwt = st.text_input(
                "Udacity staff JWT",
                value="",
                key="udacity_staff_jwt_input",
                type="password",
                help=(
                    "Your personal Udacity staff JWT. Used as the Bearer token "
                    "for all GraphQL calls against classroom-content / "
                    "reviews-api. Expires with your staff session, so "
                    "re-paste when it expires."
                ),
            )
            if st.button("Save JWT", use_container_width=True):
                token = (new_jwt or "").strip()
                if not token:
                    st.error("Please paste a non-empty JWT.")
                else:
                    with st.spinner("Verifying JWT against classroom-content…"):
                        ok, message = verify_jwt(token)
                    if ok:
                        st.session_state[SESSION_STATE_JWT_KEY] = token
                        st.success("JWT verified and saved for this session.")
                        st.rerun()
                    else:
                        st.error(message)
