"""Shared logic for the nil-backfill Streamlit app.

Ported from the three nil_backfill_*.ipynb notebooks. All HTTP uses stdlib
urllib (no extra dependencies). Auth is via the `_jwt` cookie + a browser
User-Agent so Cloudflare doesn't block the request.
"""

from __future__ import annotations

import csv
import io
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Iterable

# ── Constants ─────────────────────────────────────────────────────────────────

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

SCORE_MIN = 0
SCORE_MAX = 300

USER_ID_BATCH_SIZE = 1000

ENV_URLS = {
    "prod": "https://api.udacity.com/api/workera",
    "staging": "https://api-staging.udacity.com/api/workera",
}

# Plans GraphQL endpoint (same gateway, same _jwt cookie auth as workera-api).
# Used by the step-ID lookup helper to resolve plan_id → [{step_id, step_type, label}].
PLANS_GRAPHQL_URL = "https://api.udacity.com/api/plans/graphql"

# GraphQL query: fetch every step in a learning plan with its id + label,
# including one level of nested child steps.
PLAN_STEPS_QUERY = """
query($planId: ID!) {
  learningPlan(id: $planId) {
    id
    title
    steps {
      stepId
      stepType
      label
      children {
        stepId
        stepType
        label
      }
    }
  }
}
"""

# Statuses returned by the batch endpoints.
STATUS_UPDATED = "updated"
STATUS_ALREADY_ATTACHED = "already_attached"
STATUS_NOT_FOUND = "not_found"
STATUS_INVALID_INPUT = "invalid_input"
STATUS_ERROR = "error"


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class ResultRow:
    """One normalized row of output, regardless of which operation produced it."""

    attempt_id: str
    status: str
    previous: str | None = None
    new: str | None = None
    message: str | None = None
    http_status: int | None = None

    def as_dict(self) -> dict:
        return {
            "attempt_id": self.attempt_id,
            "status": self.status,
            "previous": self.previous,
            "new": self.new,
            "message": self.message,
            "http_status": self.http_status,
        }


@dataclass
class RunSummary:
    """Aggregated totals across a run."""

    total: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0
    rows: list[ResultRow] = field(default_factory=list)

    def add(self, row: ResultRow) -> None:
        self.rows.append(row)
        self.total += 1
        if row.status == STATUS_UPDATED:
            self.updated += 1
        elif row.status in (STATUS_ALREADY_ATTACHED, STATUS_NOT_FOUND):
            self.skipped += 1
        else:
            self.errors += 1


# ── Validation ────────────────────────────────────────────────────────────────


def validate_uuid(value: str, label: str) -> str | None:
    """Return an error string if `value` is not a UUID, else None."""
    v = str(value).strip()
    if not UUID_RE.match(v):
        return f"{label}: not a valid UUID: {value!r}"
    return None


def validate_score(value, label: str) -> tuple[int | None, str | None]:
    """Return (int_score, error_str_or_None). Score must be an int in 0–300."""
    try:
        s = int(value)
    except (TypeError, ValueError):
        return None, f"{label}: score must be an integer, got {value!r}"
    if s < SCORE_MIN or s > SCORE_MAX:
        return None, f"{label}: score must be {SCORE_MIN}–{SCORE_MAX}, got {s}"
    return s, None


def parse_csv(uploaded_file, required_cols: set[str]) -> tuple[list[dict], list[str]]:
    """Parse a Streamlit UploadedFile (or file-like) as CSV.

    Returns (rows_as_dicts, errors). Each row dict has only the required columns,
    string-valued. `errors` is a list of human-readable strings; if non-empty,
    `rows` should be considered unusable.
    """
    errors: list[str] = []
    rows: list[dict] = []

    # Streamlit UploadedFile is a BytesIO-like object; read as text.
    raw = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file
    if isinstance(raw, bytes):
        text = raw.decode("utf-8-sig")
    else:
        text = str(raw)

    reader = csv.DictReader(io.StringIO(text))
    fieldnames = set(reader.fieldnames or [])
    missing = required_cols - fieldnames
    if missing:
        errors.append(f"CSV is missing required column(s): {sorted(missing)}")
        return rows, errors

    for i, row in enumerate(reader):
        clean = {col: (row.get(col) or "").strip() for col in required_cols}
        rows.append(clean)

    if not rows and not errors:
        errors.append("CSV has no data rows")

    return rows, errors


# ── HTTP client ───────────────────────────────────────────────────────────────


def make_headers(jwt: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Cookie": f"_jwt={jwt.strip()}",
        "User-Agent": BROWSER_UA,
    }


def verify_jwt(base_url: str, jwt: str) -> tuple[bool, str, int | None]:
    """Quick read-only auth check against the Workera API.

    Sends a GET to the API base. Returns (accepted, message, http_status).
    A 401/403 means the JWT was rejected; any other response (including 404
    or 405) means the auth layer accepted the JWT. No data is mutated.
    """
    headers = {
        "Cookie": f"_jwt={jwt.strip()}",
        "User-Agent": BROWSER_UA,
    }
    url = base_url.rstrip("/")
    req = urllib.request.Request(url, method="GET", headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            return True, "JWT accepted by Workera API", resp.status
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return False, f"JWT rejected (HTTP {exc.code})", exc.code
        return True, f"JWT accepted (probe returned HTTP {exc.code})", exc.code
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Network error — could not reach {url!r}. Reason: {exc.reason}. "
            "Check VPN / environment."
        ) from exc


def fetch_plan_steps(jwt: str, plan_id: str) -> tuple[dict, list[dict]]:
    """Look up all steps in a learning plan via the plans GraphQL API.

    Returns (plan, rows) where `plan` is {"id", "title"} and `rows` is a flat
    list of dicts: {"stepId", "stepType", "label", "parentLabel"}. Top-level
    steps have parentLabel = None; nested children carry their parent's label.
    Read-only — sends a single GraphQL POST, mutates nothing. Raises
    RuntimeError on network errors, non-200 responses, or GraphQL errors.
    """
    headers = {
        "Content-Type": "application/json",
        "Cookie": f"_jwt={jwt.strip()}",
        "User-Agent": BROWSER_UA,
    }
    payload = json.dumps(
        {"query": PLAN_STEPS_QUERY, "variables": {"planId": plan_id.strip()}}
    ).encode()
    req = urllib.request.Request(
        PLANS_GRAPHQL_URL, data=payload, method="POST", headers=headers
    )
    try:
        with urllib.request.urlopen(req) as resp:
            body = _parse_body(resp.read())
    except urllib.error.HTTPError as exc:
        detail = _parse_body(exc.read())
        detail = detail if isinstance(detail, str) else json.dumps(detail)
        raise RuntimeError(f"Plans API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Network error reaching plans API: {exc.reason}. Check VPN / environment."
        ) from exc

    if not isinstance(body, dict):
        raise RuntimeError(f"Unexpected response from plans API: {body!r}")
    if body.get("errors"):
        raise RuntimeError(f"GraphQL errors: {json.dumps(body['errors'])}")

    plan = (body.get("data") or {}).get("learningPlan") or {}
    steps = plan.get("steps") or []

    rows: list[dict] = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        rows.append(
            {
                "stepId": s.get("stepId"),
                "stepType": s.get("stepType"),
                "label": s.get("label"),
                "parentLabel": None,
            }
        )
        for child in s.get("children") or []:
            if not isinstance(child, dict):
                continue
            rows.append(
                {
                    "stepId": child.get("stepId"),
                    "stepType": child.get("stepType"),
                    "label": child.get("label"),
                    "parentLabel": s.get("label"),
                }
            )

    plan_info = {"id": plan.get("id"), "title": plan.get("title")}
    return plan_info, rows


def _request(
    url: str,
    payload: bytes,
    headers: dict,
    method: str = "PATCH",
) -> tuple[int, object]:
    """Send an HTTP request and return (status, parsed_body).

    `parsed_body` is a dict/list if the response is JSON, else a string.
    Raises SystemExit on network errors (caller wraps in try/except as needed).
    """
    req = urllib.request.Request(url, data=payload, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, _parse_body(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read()
        return exc.code, _parse_body(body)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Network error — could not reach {url!r}. Reason: {exc.reason}. "
            "Check VPN / environment."
        ) from exc


def _parse_body(raw: bytes):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw.decode(errors="replace")


# ── Operations ────────────────────────────────────────────────────────────────


def override_score(
    base_url: str,
    jwt: str,
    updates: Iterable[tuple[str, int]],
    progress=None,
) -> RunSummary:
    """Score override: one PATCH per attempt to /attempts/{id}/score.

    `progress` is an optional callable(completed, total) for UI progress bars.
    """
    headers = make_headers(jwt)
    updates = list(updates)
    summary = RunSummary()
    total = len(updates)

    for i, (attempt_id, score) in enumerate(updates, start=1):
        url = f"{base_url.rstrip('/')}/attempts/{attempt_id}/score"
        payload = json.dumps({"score": score}).encode()
        try:
            status, body = _request(url, payload, headers)
        except RuntimeError as exc:
            summary.add(
                ResultRow(
                    attempt_id=attempt_id,
                    status=STATUS_ERROR,
                    message=str(exc),
                    http_status=None,
                )
            )
            if progress:
                progress(i, total)
            continue

        if status == 200 and isinstance(body, dict):
            summary.add(
                ResultRow(
                    attempt_id=attempt_id,
                    status=STATUS_UPDATED,
                    previous=str(body.get("previous_score")) if body.get("previous_score") is not None else None,
                    new=str(body.get("new_score")),
                    http_status=status,
                )
            )
        else:
            detail = body if isinstance(body, str) else json.dumps(body)
            summary.add(
                ResultRow(
                    attempt_id=attempt_id,
                    status=STATUS_ERROR,
                    message=detail,
                    http_status=status,
                )
            )

        if progress:
            progress(i, total)

    return summary


def backfill_step_id(
    base_url: str,
    jwt: str,
    updates: Iterable[tuple[str, str]],
    progress=None,
) -> RunSummary:
    """Step-ID backfill: one PATCH to /attempts/step-id/batch with all updates.

    The endpoint only writes when step_id IS NULL.
    """
    headers = make_headers(jwt)
    updates = list(updates)
    url = f"{base_url.rstrip('/')}/attempts/step-id/batch"
    payload = json.dumps(
        {"updates": [{"id": a.strip(), "step_id": s.strip()} for a, s in updates]}
    ).encode()

    try:
        status, body = _request(url, payload, headers)
    except RuntimeError as exc:
        summary = RunSummary()
        for a, _ in updates:
            summary.add(ResultRow(attempt_id=a, status=STATUS_ERROR, message=str(exc)))
        return summary

    if progress:
        progress(1, 1)

    return _summarize_batch_response(updates, status, body)


def backfill_user_id(
    base_url: str,
    jwt: str,
    updates: Iterable[tuple[str, str]],
    batch_size: int = USER_ID_BATCH_SIZE,
    progress=None,
) -> RunSummary:
    """User-ID backfill: PATCH /attempts/user-id/batch, batched at `batch_size`.

    The endpoint only writes when user_id IS NULL.
    """
    headers = make_headers(jwt)
    updates = list(updates)
    url = f"{base_url.rstrip('/')}/attempts/user-id/batch"
    summary = RunSummary()

    batches = [updates[i : i + batch_size] for i in range(0, len(updates), batch_size)]
    completed = 0
    total = len(batches)

    for batch in batches:
        payload = json.dumps(
            {"updates": [{"id": a.strip(), "user_id": u.strip()} for a, u in batch]}
        ).encode()
        try:
            status, body = _request(url, payload, headers)
        except RuntimeError as exc:
            for a, _ in batch:
                summary.add(ResultRow(attempt_id=a, status=STATUS_ERROR, message=str(exc)))
            completed += 1
            if progress:
                progress(completed, total)
            continue

        _summarize_batch_response(batch, status, body, into=summary)
        completed += 1
        if progress:
            progress(completed, total)

    return summary


# ── Helpers ──────────────────────────────────────────────────────────────────


def _summarize_batch_response(
    updates: list[tuple],
    status: int,
    body,
    into: RunSummary | None = None,
) -> RunSummary:
    """Convert a batch endpoint response into ResultRows.

    Batch responses look like:
        {"total": N, "updated": N, "skipped": N, "errors": N,
         "results": [{"id": ..., "status": "updated", "previous_step_id": ...,
                       "new_step_id": ...}, ...]}
    """
    summary = into if into is not None else RunSummary()

    if status != 200 or not isinstance(body, dict):
        detail = body if isinstance(body, str) else json.dumps(body)
        for a, _ in updates:
            summary.add(
                ResultRow(
                    attempt_id=a,
                    status=STATUS_ERROR,
                    message=detail,
                    http_status=status,
                )
            )
        return summary

    results = body.get("results", [])
    # Index results by id for quick lookup; fall back to positional.
    by_id = {r.get("id"): r for r in results if isinstance(r, dict)}

    for a, _ in updates:
        r = by_id.get(a)
        if r is None:
            # No per-row result; infer a generic error.
            summary.add(
                ResultRow(
                    attempt_id=a,
                    status=STATUS_ERROR,
                    message="No per-row result returned by endpoint",
                    http_status=status,
                )
            )
            continue

        s = r.get("status", "unknown")
        row = ResultRow(attempt_id=a, status=s, http_status=status)

        # Step-id and user-id responses use different field names; handle both.
        if "previous_step_id" in r or "new_step_id" in r or "current_step_id" in r:
            row.previous = _to_str(r.get("previous_step_id"))
            row.new = _to_str(r.get("new_step_id"))
            if s == STATUS_ALREADY_ATTACHED and r.get("current_step_id") is not None:
                row.previous = _to_str(r.get("current_step_id"))
        elif "previous_user_id" in r or "new_user_id" in r or "current_user_id" in r:
            row.previous = _to_str(r.get("previous_user_id"))
            row.new = _to_str(r.get("new_user_id"))
            if s == STATUS_ALREADY_ATTACHED and r.get("current_user_id") is not None:
                row.previous = _to_str(r.get("current_user_id"))
        else:
            row.previous = _to_str(r.get("previous"))
            row.new = _to_str(r.get("new"))

        if r.get("message"):
            row.message = str(r["message"])

        summary.add(row)

    return summary


def _to_str(v) -> str | None:
    if v is None:
        return None
    return str(v)


def status_icon(status: str) -> str:
    """Human-readable label for a status, with an icon."""
    return {
        STATUS_UPDATED: "✅ updated",
        STATUS_ALREADY_ATTACHED: "⚠️ already_attached",
        STATUS_NOT_FOUND: "❌ not_found",
        STATUS_INVALID_INPUT: "❌ invalid_input",
        STATUS_ERROR: "❌ error",
    }.get(status, f"? {status}")
