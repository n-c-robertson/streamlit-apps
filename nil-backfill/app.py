"""nil-backfill Streamlit app.

Unified UI for the three Workera nil-backfill operations:
  - Score Override  (PATCH /attempts/{id}/score)
  - Step-ID Backfill (PATCH /attempts/step-id/batch)
  - User-ID Backfill (PATCH /attempts/user-id/batch)

Flow: pick operation + inputs → Preview (validates, no network)
→ Confirm & Run (sends PATCHes) → results table + CSV download.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

import streamlit as st

import lib

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="nil-backfill",
    page_icon="🛠️",
    layout="wide",
)

# ── Operations ───────────────────────────────────────────────────────────────

OP_SCORE = "Score Override"
OP_STEP_ID = "Step-ID Backfill"
OP_USER_ID = "User-ID Backfill"

OPERATIONS = [OP_SCORE, OP_STEP_ID, OP_USER_ID]


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("nil-backfill")

base_url = lib.ENV_URLS["prod"]

# Persist the JWT in session_state across reruns.
if "saved_jwt" not in st.session_state:
    st.session_state.saved_jwt = ""

jwt = st.sidebar.text_input(
    "Staff JWT",
    value=st.session_state.saved_jwt,
    type="password",
    help="Paste your staff JWT (from baton / config_secrets.json). "
    "Never committed; lives only in this browser session.",
)

if st.sidebar.button("Save & verify JWT", type="primary", use_container_width=True):
    if not jwt.strip():
        st.sidebar.error("Paste a JWT first.")
    else:
        st.session_state.saved_jwt = jwt.strip()
        with st.sidebar.status("Verifying JWT against prod…"):
            try:
                accepted, msg, _code = lib.verify_jwt(base_url, jwt.strip())
            except RuntimeError as exc:
                st.sidebar.error(f"Could not verify: {exc}")
            else:
                if accepted:
                    st.sidebar.success(f"✅ {msg}")
                else:
                    st.sidebar.error(f"❌ {msg}")

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("Workera nil-backfill")

# ── Instructions ─────────────────────────────────────────────────────────────

with st.expander("Instructions", expanded=False):
    st.markdown(
        "**Step 1:** run this query. Update the `workera_user_email` to the "
        "user with a reported issue. You can run this in PopSQL or DeepNote."
    )
    st.code(
        """select
    attempts.created_at,
    domains.title as workera_assessment_title,
    attempts.id as workera_attempt_id,
    attempts.user_id as workera_attempt_user_id,
    attempts.step_id as workera_attempt_learning_plan_step_id,
    attempts.score as workera_attempt_score,
    users.native_user_id as user_id
from crdb_exports.workera_api__public_attempts as attempts
left join crdb_exports.workera_api__public_assessments as domains
on attempts.domain_identifier = domains.identifier
left join factstores.user_fact_all_users as users
on attempts.workera_user_email = users.email
where workera_user_email = 'nathan.robertson@udacity.com'
and (
    -- looks for record with a fail state.
    attempts.user_id is null
    or 
    attempts.step_id is null
)
and 
    domains.title = 'ChatGPT'""",
        language="sql",
    )
    st.markdown("**Step 2:** assess the issue with the record:")
    st.markdown(
        "* **step_id:** if this is null, that means the score is not currently "
        "associated with any learning plan step. Learners can't use this score "
        "to get credit for completing a step.\n"
        "* **user_id:** if this is null, it causes a cosmetic issue where even "
        "if this is tied to a learning plan step, the step won't be marked as "
        "completed.\n"
        "* **score:** Sometimes, learners contest their score on Workera and "
        "get a higher score - but this doesn't sync back. So we have to "
        "manually patch their score to update it."
    )
    st.markdown("**Step 3:** make the fix:")
    st.markdown(
        "* **step_id:** Select Step ID on the operation dropdown. You can "
        "paste in the ID of the learning plan to quickly see all step IDs so "
        "you can find the right `step_id`. Make sure to confirm if this score "
        "needs to be attached to the pre-assessment or post-assessment step "
        "if the assessment is in the learning plan twice. Add the "
        "`workera_attempt_id` and the `step_id`, and submit.\n"
        "* **user_id:** Select User ID on the operation dropdown. The query "
        "you ran in step 1 has the user's `user_id`. Just paste that into the "
        "field along with the `attempt_id`, and submit.\n"
        "* **score:** Pick Score from the operation dropdown, put in the "
        "attempt ID and the desired score."
    )
    st.markdown(
        "The app will show you a preview. Make sure to click again to submit "
        "the operation."
    )
    st.markdown(
        "You can confirm your changes took place by impersonating the user "
        "and viewing the learning plan in question. The changes will take "
        "immediately."
    )

op = st.selectbox("Operation", OPERATIONS, help="Which backfill to run.")

st.session_state.setdefault("preview", None)  # list[tuple] | None
st.session_state.setdefault("preview_op", None)
st.session_state.setdefault("summary", None)


# ── Input collection ─────────────────────────────────────────────────────────

def _collect_score_inputs() -> tuple[list[tuple[str, int]] | None, list[str]]:
    """Return (updates, errors). updates is None if validation failed."""
    mode = st.radio("Input mode", ["Single", "CSV"], horizontal=True, key="score_mode")
    updates: list[tuple[str, int]] = []
    errors: list[str] = []

    if mode == "Single":
        c1, c2 = st.columns(2)
        attempt_id = c1.text_input("Attempt ID (UUID)", key="score_single_aid")
        score = c2.number_input(
            "Score (0–300)",
            min_value=0,
            max_value=300,
            value=0,
            step=1,
            key="score_single_score",
        )
        if st.button("Preview", type="primary", key="score_preview_btn"):
            err = lib.validate_uuid(attempt_id, "Attempt ID")
            if err:
                errors.append(err)
            s_val, s_err = lib.validate_score(score, "Score")
            if s_err:
                errors.append(s_err)
            if errors:
                return None, errors
            updates = [(attempt_id.strip(), int(s_val))]
            return updates, []
        return None, []

    # CSV
    uploaded = st.file_uploader(
        "CSV with columns: attempt_id, score",
        type=["csv"],
        key="score_csv_uploader",
    )
    if st.button("Preview", type="primary", key="score_csv_preview_btn"):
        if uploaded is None:
            return None, ["Upload a CSV file first."]
        rows, errs = lib.parse_csv(uploaded, {"attempt_id", "score"})
        if errs:
            return None, errs
        for i, row in enumerate(rows, start=2):
            err = lib.validate_uuid(row["attempt_id"], f"Row {i} attempt_id")
            if err:
                errors.append(err)
                continue
            s_val, s_err = lib.validate_score(row["score"], f"Row {i} score")
            if s_err:
                errors.append(s_err)
                continue
            updates.append((row["attempt_id"], int(s_val)))
        if errors:
            return None, errors
        if not updates:
            return None, ["No valid rows to send."]
        return updates, []
    return None, []

def _collect_step_id_inputs() -> tuple[list[tuple[str, str]] | None, list[str]]:
    """Step-ID is single-only."""
    # Helper: resolve plan_id → step labels/IDs so the operator can pick the
    # right step_id. Read-only GraphQL call; mutates nothing.
    with st.expander("Find step ID from a plan", expanded=False):
        plan_id = st.text_input("Plan ID (UUID)", key="step_lookup_plan_id")
        if st.button("Look up steps", key="step_lookup_btn"):
            err = lib.validate_uuid(plan_id, "Plan ID")
            if err:
                st.error(err)
            elif not jwt.strip():
                st.info("Enter & save your staff JWT in the sidebar first.")
            else:
                with st.spinner("Fetching plan steps…"):
                    try:
                        plan_info, steps = lib.fetch_plan_steps(jwt, plan_id)
                    except RuntimeError as exc:
                        st.error(str(exc))
                    else:
                        if not steps:
                            st.info("No steps found for that plan.")
                        else:
                            import pandas as pd
                            if plan_info.get("title"):
                                st.caption(
                                    f"**Plan:** {plan_info['title']} "
                                    f"(`{plan_info.get('id')}`)"
                                )
                            st.dataframe(
                                pd.DataFrame(steps),
                                use_container_width=True,
                                hide_index=True,
                            )
        st.caption("Copy the step_id you need into the Step ID field below.")

    c1, c2 = st.columns(2)
    attempt_id = c1.text_input("Attempt ID (UUID)", key="step_aid")
    step_id = c2.text_input("Step ID (UUID)", key="step_sid")

    if st.button("Preview", type="primary", key="step_preview_btn"):
        errors: list[str] = []
        e1 = lib.validate_uuid(attempt_id, "Attempt ID")
        if e1:
            errors.append(e1)
        e2 = lib.validate_uuid(step_id, "Step ID")
        if e2:
            errors.append(e2)
        if errors:
            return None, errors
        return [(attempt_id.strip(), step_id.strip())], []
    return None, []

def _collect_user_id_inputs() -> tuple[list[tuple[str, str]] | None, list[str]]:
    mode = st.radio("Input mode", ["Single", "CSV"], horizontal=True, key="user_mode")
    updates: list[tuple[str, str]] = []
    errors: list[str] = []

    if mode == "Single":
        c1, c2 = st.columns(2)
        attempt_id = c1.text_input("Attempt ID (UUID)", key="user_single_aid")
        user_id = c2.text_input("User ID (UUID)", key="user_single_uid")
        if st.button("Preview", type="primary", key="user_preview_btn"):
            e1 = lib.validate_uuid(attempt_id, "Attempt ID")
            if e1:
                errors.append(e1)
            e2 = lib.validate_uuid(user_id, "User ID")
            if e2:
                errors.append(e2)
            if errors:
                return None, errors
            return [(attempt_id.strip(), user_id.strip())], []
        return None, []

    uploaded = st.file_uploader(
        "CSV with columns: attempt_id, user_id",
        type=["csv"],
        key="user_csv_uploader",
    )
    if st.button("Preview", type="primary", key="user_csv_preview_btn"):
        if uploaded is None:
            return None, ["Upload a CSV file first."]
        rows, errs = lib.parse_csv(uploaded, {"attempt_id", "user_id"})
        if errs:
            return None, errs
        for i, row in enumerate(rows, start=2):
            e1 = lib.validate_uuid(row["attempt_id"], f"Row {i} attempt_id")
            if e1:
                errors.append(e1)
                continue
            e2 = lib.validate_uuid(row["user_id"], f"Row {i} user_id")
            if e2:
                errors.append(e2)
                continue
            updates.append((row["attempt_id"], row["user_id"]))
        if errors:
            return None, errors
        if not updates:
            return None, ["No valid rows to send."]
        return updates, []
    return None, []

# ── Run ──────────────────────────────────────────────────────────────────────


def _run(op: str, updates) -> lib.RunSummary:
    """Dispatch to the right operation and return a RunSummary."""
    progress_bar = st.progress(0.0, text="Sending…")
    status_text = st.empty()

    def progress_cb(done: int, total: int) -> None:
        frac = done / total if total else 1.0
        progress_bar.progress(min(frac, 1.0), text=f"{done}/{total} batch(es) sent")
        status_text.text(f"{done}/{total} complete")

    if op == OP_SCORE:
        return lib.override_score(base_url, jwt, updates, progress=progress_cb)
    if op == OP_STEP_ID:
        return lib.backfill_step_id(base_url, jwt, updates, progress=progress_cb)
    if op == OP_USER_ID:
        return lib.backfill_user_id(base_url, jwt, updates, progress=progress_cb)
    raise ValueError(f"Unknown operation: {op}")


# ── Render ────────────────────────────────────────────────────────────────────

# Collect inputs for the selected operation.
if op == OP_SCORE:
    updates, errors = _collect_score_inputs()
elif op == OP_STEP_ID:
    updates, errors = _collect_step_id_inputs()
else:
    updates, errors = _collect_user_id_inputs()

# Show validation errors from the preview attempt.
if errors:
    for e in errors:
        st.error(e)

# Store a successful preview.
if updates is not None:
    st.session_state.preview = updates
    st.session_state.preview_op = op

preview = st.session_state.preview
preview_op = st.session_state.preview_op

# ── Preview banner ───────────────────────────────────────────────────────────

if preview is not None and preview_op == op:
    st.divider()
    n = len(preview)

    # Build a preview dataframe.
    if op == OP_SCORE:
        rows_for_df = [
            {"attempt_id": a, "score": s, "endpoint": f"PATCH /attempts/{a}/score"}
            for a, s in preview
        ]
    elif op == OP_STEP_ID:
        rows_for_df = [
            {"attempt_id": a, "step_id": s, "endpoint": "PATCH /attempts/step-id/batch"}
            for a, s in preview
        ]
    else:
        rows_for_df = [
            {"attempt_id": a, "user_id": u, "endpoint": "PATCH /attempts/user-id/batch"}
            for a, u in preview
        ]

    import pandas as pd

    st.subheader("Preview")
    st.warning(
        f"**{n} PATCH(es)** will be sent to **prod** (`{base_url}`). "
        + (
            "⚠️ Score override **overwrites existing data** and sets "
            "`score_overridden` — this is **not reversible**."
            if op == OP_SCORE
            else "This operation only writes when the target column is NULL — "
            "existing attachments will be skipped, not overwritten."
        )
    )
    st.dataframe(pd.DataFrame(rows_for_df), use_container_width=True, hide_index=True)

    # ── Confirm & Run ─────────────────────────────────────────────────────────
    if not jwt.strip():
        st.info("Enter your staff JWT in the sidebar to enable Confirm & Run.")
        confirm_disabled = True
    else:
        confirm_disabled = False

    if st.button(
        "Confirm & Run",
        type="primary",
        disabled=confirm_disabled,
        key="confirm_run_btn",
    ):
        with st.spinner("Running…"):
            summary = _run(op, preview)
        st.session_state.summary = summary
        # Clear the preview so a second click doesn't re-run without re-preview.
        st.session_state.preview = None
        st.session_state.preview_op = None
        st.rerun()

# ── Results ──────────────────────────────────────────────────────────────────

summary: lib.RunSummary | None = st.session_state.summary
if summary is not None:
    st.divider()
    st.subheader("Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total", summary.total)
    m2.metric("Updated", summary.updated)
    m3.metric("Skipped", summary.skipped)
    m4.metric("Errors", summary.errors)

    import pandas as pd

    rows_df = [r.as_dict() for r in summary.rows]
    df = pd.DataFrame(rows_df)

    # Add a human-readable status label.
    if not df.empty:
        df["status_label"] = df["status"].map(lib.status_icon)

    st.dataframe(df, use_container_width=True, hide_index=True)

    # CSV download.
    if not df.empty:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        st.download_button(
            label="Download results as CSV",
            data=buf.getvalue(),
            file_name=f"nil-backfill-{ts}.csv",
            mime="text/csv",
        )

    # Clear button.
    if st.button("Clear results", key="clear_btn"):
        st.session_state.summary = None
        st.rerun()
