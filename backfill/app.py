"""
Streamlit reviewer for J&J assessment backfill: one user × domain at a time.

Configuration is read from **Streamlit secrets** only: local ``.streamlit/secrets.toml`` or
Streamlit Community Cloud **App settings → Secrets**.

**Copy-paste TOML (full template with multiline JSON):** see
``streamlit_secrets_dict.example.PASTE_READY_SECRETS_TOML`` — or use this minimal shape::

    [google_sheets]
    spreadsheet_id = "YOUR_SPREADSHEET_ID"
    source_sheet_gid = 0
    output_sheet_gid = 1198426749
    service_account_json = '''{ ... paste entire Google Cloud service account JSON ... }'''

    [google_form]
    form_id = "YOUR_FORM_ID"
    entry_payload = "entry.XXXXXXXX"

Omit the ``[google_form]`` section if you do not use Form backup. For a filled local dict
mirror (gitignored), copy ``streamlit_secrets_dict.example.py`` to ``streamlit_secrets_dict.py``.
"""

from __future__ import annotations

import hashlib
import io
import json
import traceback
from typing import Any

import altair as alt
import gspread
import pandas as pd
import requests
import streamlit as st
from google.oauth2.service_account import Credentials

# ---------------------------------------------------------------------------
# Google Sheet / Form defaults (GIDs overridden via Streamlit secrets)
# ---------------------------------------------------------------------------

_DEFAULT_SOURCE_SHEET_GID = 0
_DEFAULT_OUTPUT_SHEET_GID = 1198426749

_SCOPES = (
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
)


def _google_sheets_secret_block() -> dict[str, Any]:
    """``[google_sheets]`` section from ``st.secrets`` (may be empty)."""
    try:
        gs = st.secrets.get("google_sheets", {})
        if gs is None:
            return {}
        out = dict(gs)
        if out.get("service_account") is not None:
            out["service_account"] = dict(out["service_account"])
        return out
    except Exception:
        return {}


def spreadsheet_id() -> str:
    """Spreadsheet ID from ``st.secrets["google_sheets"]["spreadsheet_id"]``."""
    return str(_google_sheets_secret_block().get("spreadsheet_id", "")).strip()


def source_sheet_gid() -> int:
    v = _google_sheets_secret_block().get("source_sheet_gid", _DEFAULT_SOURCE_SHEET_GID)
    try:
        return int(v)
    except (TypeError, ValueError):
        return _DEFAULT_SOURCE_SHEET_GID


def output_sheet_gid() -> int:
    v = _google_sheets_secret_block().get("output_sheet_gid", _DEFAULT_OUTPUT_SHEET_GID)
    try:
        return int(v)
    except (TypeError, ValueError):
        return _DEFAULT_OUTPUT_SHEET_GID


# Google Form backup (optional): max JSON size for one answer field
_FORM_MAX_PAYLOAD_CHARS = 45_000

# Reviewer output columns (merged from output tab). assigned_step_id is filled from the source tab’s
# step_id by matching (domain_title, assigned_step_title) to (domain_title, step_title).
NEW_COLS = ("recommended_step_title", "assigned_step_title", "assigned_step_id")

# First tab must include this column next to step_title for ID lookup (same sheet as source data).
SOURCE_STEP_ID_COL = "step_id"


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def _valid_service_account_dict(d: Any) -> bool:
    return bool(
        isinstance(d, dict)
        and d.get("private_key")
        and str(d.get("private_key", "")).strip().startswith("-----BEGIN")
        and d.get("client_email")
    )


def _service_account_credentials() -> dict | None:
    """Sheets API credentials from ``st.secrets["google_sheets"]``.

    Supports either:

    - ``service_account_json`` — single string of the full Google Cloud JSON key (paste
      from the console), or
    - ``service_account`` — nested TOML table with the same keys as the JSON object.
    """
    try:
        gs = _google_sheets_secret_block()
        raw = gs.get("service_account_json")
        if raw is not None and str(raw).strip():
            d = json.loads(str(raw))
            if _valid_service_account_dict(d):
                return d
        sa = gs.get("service_account")
        if sa is not None:
            d = dict(sa)
            if _valid_service_account_dict(d):
                return d
    except (json.JSONDecodeError, TypeError, ValueError, Exception):
        pass
    return None


def _format_sheet_write_error(exc: BaseException) -> str:
    """gspread / HTTP errors sometimes have an empty str(exc); unwrap __cause__ (e.g. APIError behind PermissionError)."""
    parts: list[str] = [type(exc).__name__]
    msg = str(exc).strip()
    if msg:
        parts.append(msg)

    cause: BaseException | None = getattr(exc, "__cause__", None)
    depth = 0
    while cause is not None and depth < 5:
        cmsg = str(cause).strip()
        parts.append(f"Caused by ({type(cause).__name__}): {cmsg}" if cmsg else f"Caused by: {type(cause).__name__}")
        resp = getattr(cause, "response", None)
        if resp is not None:
            body = getattr(resp, "text", None) or ""
            if body.strip():
                parts.append(body.strip()[:1200])
        cause = getattr(cause, "__cause__", None)
        depth += 1

    if not any("Caused by" in p for p in parts):
        resp = getattr(exc, "response", None)
        if resp is not None:
            body = getattr(resp, "text", None) or ""
            if body.strip():
                parts.append(body.strip()[:1200])
    return "\n\n".join(parts) if len(parts) > 1 else (parts[0] if parts else repr(exc))


def get_gspread_client() -> gspread.Client:
    info = _service_account_credentials()
    if not info:
        raise FileNotFoundError(
            "Set ``google_sheets.service_account_json`` or ``google_sheets.service_account`` in "
            "Streamlit secrets (see streamlit_secrets_dict.example.py). "
            "Share the spreadsheet with that client_email (Editor)."
        )
    creds = Credentials.from_service_account_info(info, scopes=_SCOPES)
    return gspread.authorize(creds)


# ---------------------------------------------------------------------------
# Public CSV export (read source + output tabs)
# ---------------------------------------------------------------------------


def _export_url(spreadsheet_id: str, gid: int) -> str:
    return (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/export"
        f"?format=csv&gid={gid}"
    )


def fetch_public_sheet_csv(spreadsheet_id: str, gid: int, timeout: int = 120) -> pd.DataFrame:
    url = _export_url(spreadsheet_id, gid)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    text = r.content.decode("utf-8", errors="replace")
    head = text.lstrip()[:800].lower()
    if "<html" in head or "<!doctype" in head:
        raise ValueError(
            "Export returned HTML, not CSV. Set sharing to “Anyone with the link” can view, then try again."
        )
    return pd.read_csv(io.StringIO(text))


def _prev_assignments_mergeable(out: pd.DataFrame) -> bool:
    """Output tab can restore assignments if it has a row key and assigned_step_title."""
    if out.empty or "assigned_step_title" not in out.columns:
        return False
    return "_row_id" in out.columns or "id" in out.columns


def load_dataframe_from_public_sheet() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load source + optional assignment overlay from the spreadsheet (public export URLs)."""
    src = fetch_public_sheet_csv(spreadsheet_id(), source_sheet_gid())
    try:
        out = fetch_public_sheet_csv(spreadsheet_id(), output_sheet_gid())
    except Exception:
        return src, None

    if out.empty or not _prev_assignments_mergeable(out):
        return src, None
    return src, out


def _remote_data_fingerprint() -> str:
    """Stable hash of source + assignment overlay (same inputs as load_working_frame). Changes when the sheet is edited outside the app."""
    digest = hashlib.sha256()
    src, prev = load_dataframe_from_public_sheet()
    for label, part in (("src", src), ("prev", prev)):
        digest.update(label.encode())
        if part is None or part.empty:
            digest.update(b"empty")
            continue
        cols = sorted(part.columns.astype(str))
        sub = part[cols]
        buf = io.StringIO()
        sub.to_csv(buf, index=False)
        digest.update(buf.getvalue().encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Sheets API write (output tab)
# ---------------------------------------------------------------------------


def worksheet_by_gid(spreadsheet, gid: int):
    for ws in spreadsheet.worksheets():
        if ws.id == gid:
            return ws
    raise ValueError(f"No tab with sheet id {gid} in spreadsheet {spreadsheet_id()}")


def _cell_for_sheet(x) -> str | int | float | bool:
    if pd.isna(x):
        return ""
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if hasattr(x, "item"):
        try:
            return x.item()
        except (ValueError, AttributeError):
            pass
    return x


def fetch_current_output_tab() -> pd.DataFrame:
    """Read the output tab (public CSV). Empty if unavailable."""
    try:
        out = fetch_public_sheet_csv(spreadsheet_id(), output_sheet_gid())
    except Exception:
        return pd.DataFrame()
    for c in ("recommended_step_title", "_row_id"):
        if c in out.columns:
            out = out.drop(columns=[c])
    return out


def _merge_output_by_id(existing: pd.DataFrame, delta: pd.DataFrame) -> pd.DataFrame:
    """Keep prior rows; replace or append rows whose `id` appears in delta (delta wins)."""
    key = "id"
    if delta.empty:
        return existing
    d = delta.copy()
    d[key] = d[key].astype(str)
    if existing.empty:
        return d
    e = existing.copy()
    if key not in e.columns:
        return pd.concat([e, d], ignore_index=True)
    e[key] = e[key].astype(str)
    e = e.loc[~e[key].isin(set(d[key]))]
    return pd.concat([e, d], ignore_index=True)


def write_dataframe_to_worksheet(ws, df: pd.DataFrame) -> None:
    ws.clear()
    if df.empty:
        ws.update([list(df.columns)], value_input_option="USER_ENTERED")
        return

    out = df.copy()
    for c in out.columns:
        out[c] = out[c].map(_cell_for_sheet)
    headers = [str(c) for c in out.columns]
    data_rows = out.values.tolist()
    chunk = 2000
    first_block = [headers] + data_rows[:chunk]
    ws.update(first_block, value_input_option="USER_ENTERED")
    row_start = 2 + len(data_rows[:chunk])
    offset = chunk
    while offset < len(data_rows):
        part = data_rows[offset : offset + chunk]
        ws.update(part, range_name=f"A{row_start}", value_input_option="USER_ENTERED")
        row_start += len(part)
        offset += chunk


def save_dataframe_to_output_sheet(
    df: pd.DataFrame,
    *,
    pair_user: str,
    pair_domain: str,
) -> None:
    """Merge only this pair's backfill rows into the output tab (by `id`); keep all other rows already on the sheet."""
    delta = dataframe_backfill_rows_for_pair(df, pair_user, pair_domain)
    drop = [c for c in ("recommended_step_title", "_row_id") if c in delta.columns]
    if drop:
        delta = delta.drop(columns=drop)
    if delta.empty:
        return
    if "id" not in delta.columns:
        raise ValueError("Each row must have an `id` column to merge into the output sheet.")
    existing = fetch_current_output_tab()
    combined = _merge_output_by_id(existing, delta)
    gc = get_gspread_client()
    sh = gc.open_by_key(spreadsheet_id())
    out_ws = worksheet_by_gid(sh, output_sheet_gid())
    write_dataframe_to_worksheet(out_ws, combined)


# ---------------------------------------------------------------------------
# Optional Google Form POST
# ---------------------------------------------------------------------------


def _form_config_resolve() -> dict[str, str] | None:
    """Optional Google Form backup from ``st.secrets["google_form"]``."""
    try:
        gf = st.secrets.get("google_form", {})
        if not gf:
            return None
        gf = dict(gf)
        form_id = str(gf.get("form_id", "")).strip()
        entry = str(gf.get("entry_payload", "")).strip()
        if form_id and entry:
            return {"form_id": form_id, "entry_payload": entry}
    except Exception:
        pass
    return None


def _build_form_payload(
    df: pd.DataFrame,
    pair_user: str | None,
    pair_domain: str | None,
) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    full_csv = buf.getvalue()
    meta: dict[str, Any] = {
        "submitted_at": pd.Timestamp.utcnow().isoformat() + "Z",
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "pair_user": pair_user,
        "pair_domain": pair_domain,
    }
    payload: dict[str, Any] = {"meta": meta, "csv": full_csv, "truncated": False}

    if len(full_csv) > _FORM_MAX_PAYLOAD_CHARS and pair_user and pair_domain:
        mask = (df["user_email"] == pair_user) & (df["domain_title"] == pair_domain)
        sub = df.loc[mask]
        buf2 = io.StringIO()
        sub.to_csv(buf2, index=False)
        meta["truncated"] = True
        meta["truncation_reason"] = f"full_csv_length_{len(full_csv)}_exceeds_form_field_limit"
        meta["pair_rows"] = len(sub)
        payload["csv"] = buf2.getvalue()
        payload["truncated"] = True
    elif len(full_csv) > _FORM_MAX_PAYLOAD_CHARS:
        meta["truncated"] = True
        meta["truncation_reason"] = "no_pair_context_using_first_500_rows"
        buf3 = io.StringIO()
        df.head(500).to_csv(buf3, index=False)
        payload["csv"] = buf3.getvalue()

    text = json.dumps(payload, ensure_ascii=False)
    if len(text) > _FORM_MAX_PAYLOAD_CHARS:
        meta["truncated"] = True
        meta["truncation_reason"] = "hard_trim_json"
        payload["csv"] = payload["csv"][: max(0, _FORM_MAX_PAYLOAD_CHARS - 4000)]
        text = json.dumps(payload, ensure_ascii=False)
    return text


def _post_form_response(form_id: str, entry_key: str, text: str, timeout: int = 60) -> requests.Response:
    url = f"https://docs.google.com/forms/d/e/{form_id}/formResponse"
    data = {
        entry_key: text,
        "fvv": "1",
        "pageHistory": "0",
        "submit": "Submit",
    }
    return requests.post(url, data=data, timeout=timeout, headers={"User-Agent": "jnj-backfill-app/1.0"})


def submit_snapshot_to_google_form(
    df: pd.DataFrame,
    pair_user: str | None,
    pair_domain: str | None,
) -> tuple[bool, str]:
    cfg = _form_config_resolve()
    if not cfg:
        return True, "skipped_no_form_config"

    text = _build_form_payload(df, pair_user, pair_domain)
    try:
        r = _post_form_response(cfg["form_id"], cfg["entry_payload"], text)
        if r.status_code in (200, 302):
            return True, "ok"
        return False, f"http_{r.status_code}"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Domain logic (unchanged)
# ---------------------------------------------------------------------------


def parse_created_at(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series.astype(str).str.strip('"'), utc=True, errors="coerce")
    return dt.dt.tz_localize(None)


def format_created_at_display(val) -> str:
    ts = pd.to_datetime(val, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def series_has_step(df: pd.DataFrame) -> pd.Series:
    stitle = df["step_title"]
    return stitle.notna() & stitle.astype(str).str.strip().ne("")


def _norm_assigned_title_for_id(val) -> Any:
    """Normalize assigned_step_title for lookup; Null/empty means no id."""
    if pd.isna(val):
        return pd.NA
    s = str(val).strip()
    if s == "" or s == "Null":
        return pd.NA
    return s


def fill_assigned_step_ids(df: pd.DataFrame) -> None:
    """Set assigned_step_id by matching assigned_step_title to step_id on source rows (domain + step_title)."""
    if "assigned_step_id" not in df.columns:
        df["assigned_step_id"] = pd.NA

    if SOURCE_STEP_ID_COL not in df.columns:
        df["assigned_step_id"] = pd.NA
        return

    has_src = series_has_step(df)
    if not has_src.any():
        df["assigned_step_id"] = pd.NA
        return

    ref = (
        df.loc[has_src, ["domain_title", "step_title", SOURCE_STEP_ID_COL]]
        .dropna(subset=[SOURCE_STEP_ID_COL])
        .copy()
    )
    ref["_dom"] = ref["domain_title"].astype(str).str.strip()
    ref["_st"] = ref["step_title"].astype(str).str.strip()
    ref = ref.drop_duplicates(subset=["_dom", "_st"], keep="first")
    lut: dict[tuple[str, str], Any] = {
        (row["_dom"], row["_st"]): row[SOURCE_STEP_ID_COL] for _, row in ref.iterrows()
    }

    out: list[Any] = []
    for _, row in df.iterrows():
        at = _norm_assigned_title_for_id(row.get("assigned_step_title"))
        if pd.isna(at):
            out.append(pd.NA)
            continue
        dom = str(row["domain_title"]).strip() if pd.notna(row.get("domain_title")) else ""
        out.append(lut.get((dom, at), pd.NA))
    df["assigned_step_id"] = out


def series_has_assigned_step(df: pd.DataFrame) -> pd.Series:
    """Reviewer output (merged from the results tab) filled in assigned_step_title."""
    if "assigned_step_title" not in df.columns:
        return pd.Series(False, index=df.index)
    a = df["assigned_step_title"]
    return a.notna() & a.astype(str).str.strip().ne("")


def series_effectively_resolved(df: pd.DataFrame) -> pd.Series:
    """Source has a step, or the output tab already has an assignment for this row."""
    return series_has_step(df) | series_has_assigned_step(df)


def dataframe_backfill_rows_only(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where the source had no step_title but assigned_step_title is set (reviewer backfill only)."""
    orig_untied = ~series_has_step(df)
    a = df["assigned_step_title"]
    has_assigned = a.notna() & a.astype(str).str.strip().ne("")
    return df.loc[orig_untied & has_assigned].copy()


def dataframe_backfill_rows_for_pair(df: pd.DataFrame, pair_user: str, pair_domain: str) -> pd.DataFrame:
    """Backfill rows limited to one user × domain (typical single Submit)."""
    base = dataframe_backfill_rows_only(df)
    if base.empty:
        return base
    m = (base["user_email"] == pair_user) & (base["domain_title"] == pair_domain)
    return base.loc[m].copy()


def domain_step_options(df: pd.DataFrame, domain_title: str) -> list[str]:
    sub = df.loc[df["domain_title"] == domain_title, "step_title"]
    opts = sub.dropna().astype(str).str.strip()
    opts = opts[opts != ""].unique().tolist()
    return sorted(opts)


def domain_max_sequence(df: pd.DataFrame, domain_title: str) -> float | None:
    sub = df.loc[df["domain_title"] == domain_title, "step_sequence"]
    valid = pd.to_numeric(sub, errors="coerce")
    if valid.notna().any():
        return float(valid.max())
    return None


def pick_pre_post_titles(options: list[str]) -> tuple[str | None, str | None]:
    pres = [o for o in options if "Pre-Assessment" in o]
    posts = [o for o in options if "Post-Assessment" in o]
    pre = pres[0] if pres else None
    post = posts[0] if posts else None
    return pre, post


def recommend_for_untied(
    sub: pd.DataFrame,
    domain_max_seq: float | None,
    pre_title: str | None,
    post_title: str | None,
) -> dict[int, str | None]:
    out: dict[int, str | None] = {}
    sub = sub.sort_values("created_at").copy()
    sub["_has_step"] = series_has_step(sub)

    if pre_title is not None and post_title is not None:
        tied = sub.loc[sub["_has_step"]]
        stitles = tied["step_title"].astype(str).str.strip()
        has_post_attempt = stitles.eq(post_title).any()
        if not has_post_attempt:
            for _, r in sub.iterrows():
                if not r["_has_step"]:
                    out[int(r["_row_id"])] = pre_title
            return out

    if domain_max_seq is None or pre_title is None or post_title is None:
        for _, r in sub.iterrows():
            rid = int(r["_row_id"])
            if not r["_has_step"]:
                out[rid] = pre_title or post_title
        return out

    mask_latest = sub["_has_step"] & (pd.to_numeric(sub["step_sequence"], errors="coerce") == domain_max_seq)
    first_latest = sub.loc[mask_latest].head(1)

    if first_latest.empty:
        first_tied_any = sub.loc[sub["_has_step"]].head(1)
        if first_tied_any.empty:
            for _, r in sub.iterrows():
                rid = int(r["_row_id"])
                if not r["_has_step"]:
                    out[rid] = pre_title
            return out
        cutoff = first_tied_any["created_at"].iloc[0]
        for _, r in sub.iterrows():
            rid = int(r["_row_id"])
            if r["_has_step"]:
                continue
            out[rid] = pre_title if r["created_at"] < cutoff else post_title
        return out

    cutoff = first_latest["created_at"].iloc[0]
    for _, r in sub.iterrows():
        rid = int(r["_row_id"])
        if r["_has_step"]:
            continue
        out[rid] = pre_title if r["created_at"] < cutoff else post_title
    return out


def load_working_frame() -> pd.DataFrame:
    """Load source tab; merge assignments from the results (output) tab so reviewed rows are pre-filled."""
    src, prev = load_dataframe_from_public_sheet()
    if src.empty and len(src.columns) == 0:
        raise ValueError("Source sheet is empty or missing a header row.")
    df = src.copy()
    df["_row_id"] = range(len(df))
    df["created_at"] = parse_created_at(df["created_at"])
    for c in NEW_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    if prev is not None and not prev.empty:
        if "_row_id" in prev.columns:
            prev_idx = prev.set_index("_row_id")
            for c in NEW_COLS:
                if c in prev_idx.columns:
                    df[c] = df["_row_id"].map(prev_idx[c]).combine_first(df[c])
        elif "id" in prev.columns and "id" in df.columns:
            prev_idx = prev.set_index("id")
            for c in NEW_COLS:
                if c in prev_idx.columns:
                    df[c] = df["id"].map(prev_idx[c]).combine_first(df[c])

    fill_assigned_step_ids(df)
    return df


def save_working_frame(
    df: pd.DataFrame,
    *,
    pair_user: str | None = None,
    pair_domain: str | None = None,
) -> bool:
    fill_assigned_step_ids(df)
    try:
        ok, msg = submit_snapshot_to_google_form(df, pair_user, pair_domain)
        if msg != "skipped_no_form_config" and not ok:
            st.warning(
                f"Google Form backup POST did not succeed ({msg}). "
                "Check `[google_form]` in `.streamlit/secrets.toml` (keys `form_id`, `entry_payload`)."
            )
    except Exception:
        st.warning("Google Form backup failed:\n```\n" + traceback.format_exc() + "\n```")

    try:
        if pair_user is None or pair_domain is None:
            raise ValueError("Internal error: pair_user and pair_domain are required to update the output sheet.")
        save_dataframe_to_output_sheet(df, pair_user=pair_user, pair_domain=pair_domain)
        return True
    except Exception as e:
        st.error(
            "Could not write to the Google Sheet output tab.\n\n"
            f"**Summary:**\n```\n{_format_sheet_write_error(e)}\n```\n\n"
            f"**Full traceback:**\n```\n{traceback.format_exc()}\n```\n\n"
            "Your assignments exist only in this browser session until the output tab write succeeds. "
            "Fix the issue below and use **Submit and next** again.\n\n"
            "**If you see API disabled / 403:** In [Google Cloud Console](https://console.cloud.google.com/apis/library) "
            "for the **same project as your service account**, enable **Google Sheets API** (and usually **Google Drive API**). "
            "Wait a few minutes after enabling.\n\n"
            "**If you see permission denied on the file:** Share the spreadsheet with the service account **client_email** "
            "as **Editor**.\n\n"
            "Then **Submit and next** again."
        )
        return False


def sync_tied_assignments(df: pd.DataFrame, sub: pd.DataFrame) -> None:
    for _, r in sub.loc[series_has_step(sub)].iterrows():
        rid = int(r["_row_id"])
        df.loc[df["_row_id"] == rid, "recommended_step_title"] = pd.NA
        df.loc[df["_row_id"] == rid, "assigned_step_title"] = r["step_title"]
        if SOURCE_STEP_ID_COL in df.columns and SOURCE_STEP_ID_COL in sub.columns:
            df.loc[df["_row_id"] == rid, "assigned_step_id"] = r[SOURCE_STEP_ID_COL]


def unique_pairs_needing_backfill(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Pairs that still have at least one row needing assignment (no source step and no output assignment yet)."""
    still_open = ~series_effectively_resolved(df)
    needs = df.loc[still_open]
    if needs.empty:
        return []
    g = needs.groupby(["user_email", "domain_title"], sort=False).size().reset_index()[
        ["user_email", "domain_title"]
    ]
    g = g.sort_values(["user_email", "domain_title"])
    return list(zip(g["user_email"], g["domain_title"]))


def unique_pairs_with_source_gap(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Pairs that have at least one row with no source step_title (fixed total for n / x progress)."""
    gap = ~series_has_step(df)
    if not gap.any():
        return []
    g = df.loc[gap].groupby(["user_email", "domain_title"], sort=False).size().reset_index()[
        ["user_email", "domain_title"]
    ]
    g = g.sort_values(["user_email", "domain_title"])
    return list(zip(g["user_email"], g["domain_title"]))


def _require_streamlit_secrets() -> None:
    if not spreadsheet_id():
        st.error(
            "Missing **google_sheets.spreadsheet_id** in Streamlit secrets. "
            "Use the TOML template in `streamlit_secrets_dict.example.py` → `.streamlit/secrets.toml`."
        )
        st.stop()
    if not _service_account_credentials():
        st.error(
            "Missing or invalid **google_sheets.service_account_json** or **[google_sheets.service_account]**. "
            "See `streamlit_secrets_dict.example.py`."
        )
        st.stop()


def main() -> None:
    st.set_page_config(page_title="J&J backfill review", layout="wide")
    _require_streamlit_secrets()
    st.title("Assessment step backfill review")

    fp_now = _remote_data_fingerprint()
    if "data_fp" not in st.session_state:
        st.session_state.df = load_working_frame()
        st.session_state.data_fp = fp_now
    elif fp_now != st.session_state.data_fp:
        st.session_state.df = load_working_frame()
        st.session_state.data_fp = fp_now
        st.session_state.pair_idx = 0
        st.session_state.tada_shown = False
    if "pair_idx" not in st.session_state:
        st.session_state.pair_idx = 0
    if "tada_shown" not in st.session_state:
        st.session_state.tada_shown = False

    df: pd.DataFrame = st.session_state.df
    pairs = unique_pairs_needing_backfill(df)
    total_queue = len(pairs)
    scope_pairs = unique_pairs_with_source_gap(df)
    scope_total = len(scope_pairs)
    n_resolved = scope_total - total_queue
    idx = int(st.session_state.pair_idx)
    if idx > total_queue:
        st.session_state.pair_idx = total_queue
        idx = total_queue

    st.progress(min(n_resolved / max(scope_total, 1), 1.0))
    st.caption(
        f"Progress: {n_resolved} / {scope_total} user–domain pairs resolved "
        f"({total_queue} still in queue)"
    )

    if scope_total == 0:
        st.info("No user–domain pairs require backfill — every row already has a source step_title.")
        return

    if idx >= total_queue:
        if not st.session_state.tada_shown:
            st.balloons()
            st.session_state.tada_shown = True
        st.success("All pairs reviewed.")
        st.markdown("## Tada! You are done.")
        if st.button("Start over", type="primary"):
            st.session_state.pair_idx = 0
            st.session_state.tada_shown = False
            st.rerun()
        return

    user_email, domain_title = pairs[idx]
    st.subheader("Current pair")
    st.write(f"**User:** `{user_email}`")
    st.write(f"**Domain:** {domain_title}")

    sub = df.loc[(df["user_email"] == user_email) & (df["domain_title"] == domain_title)].copy()
    sub = sub.sort_values("created_at")
    sub["_has_step"] = series_effectively_resolved(sub)
    sub["status"] = sub["_has_step"].map(
        {True: "Resolved (source step or saved assignment)", False: "Needs assignment"}
    )
    sub["attempt_order"] = range(1, len(sub) + 1)

    options = domain_step_options(df, domain_title)
    dm = domain_max_sequence(df, domain_title)
    pre_t, post_t = pick_pre_post_titles(options)
    rec_map = recommend_for_untied(sub, dm, pre_t, post_t)

    chart_df = sub.assign(
        attempt_label=lambda d: "Attempt " + d["attempt_order"].astype(str),
        attachment=lambda d: d["_has_step"].map({True: "Attached", False: "Unattached"}),
    )[["attempt_label", "attempt_order", "score", "attachment"]].copy()
    chart_df["score"] = pd.to_numeric(chart_df["score"], errors="coerce").fillna(0)
    attempt_sort = [f"Attempt {i}" for i in range(1, len(sub) + 1)]
    bar_height = max(220, min(520, 80 + len(sub) * 28))
    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("attempt_label:N", sort=attempt_sort, title=None),
            y=alt.Y("score:Q", title="Score"),
            color=alt.Color(
                "attachment:N",
                scale=alt.Scale(
                    domain=["Attached", "Unattached"],
                    range=["#22c55e", "#ef4444"],
                ),
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip("attempt_label:N", title="Attempt"),
                alt.Tooltip("attachment:N", title="Status"),
                alt.Tooltip("score:Q", title="Score"),
            ],
        )
        .properties(height=bar_height)
    )
    st.altair_chart(chart, use_container_width=True)

    show_cols = [
        "created_at",
        "score",
        "plan_title",
        "step_title",
        "assigned_step_title",
        "assigned_step_id",
        "status",
    ]
    display_table = sub[[c for c in show_cols if c in sub.columns]].copy()
    display_table["created_at"] = sub["created_at"].map(format_created_at_display)
    st.dataframe(display_table, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Assign steps for attempts **not** tied to a step")

    untied = sub.loc[~sub["_has_step"]]
    if untied.empty:
        st.info("No untied attempts for this pair — nothing to assign.")
        if st.button("Submit and next", type="primary"):
            sync_tied_assignments(df, sub)
            st.session_state.df = df
            if not save_working_frame(df, pair_user=user_email, pair_domain=domain_title):
                st.stop()
            st.session_state.data_fp = _remote_data_fingerprint()
            st.session_state.pair_idx = idx + 1
            st.rerun()
        return

    if not options:
        st.error("No step titles found in the dataset for this domain — cannot assign.")
        return

    assign_placeholder = "— Select step —"
    null_assignment = "Null"
    select_options = [assign_placeholder, null_assignment, *options]

    def default_assignment_for_row(rid: int, rec) -> str:
        prior = df.loc[df["_row_id"] == rid, "assigned_step_title"].iloc[0]
        if pd.notna(prior) and str(prior).strip() != "":
            da = str(prior).strip()
        elif rec and rec in options:
            da = rec
        else:
            da = assign_placeholder
        if da not in select_options:
            da = assign_placeholder
        return da

    assignment_rows: list[tuple[int, str, str, str, str]] = []
    for _, r in untied.iterrows():
        rid = int(r["_row_id"])
        rec = rec_map.get(rid)
        default_assign = default_assignment_for_row(rid, rec)
        rec_display = rec if rec else ""
        score_val = r["score"]
        if pd.isna(score_val):
            score_display = ""
        else:
            score_display = str(score_val).strip()
        assignment_rows.append(
            (
                rid,
                format_created_at_display(r["created_at"]),
                score_display,
                rec_display,
                default_assign,
            )
        )

    st.caption(
        "Use the **Assignment** dropdown on each row to pick a step, or **Null** if you do not want an assignment. "
        "Submit stays disabled until each row is set to something other than “— Select step —”."
    )

    col_weights = (3, 1, 4, 4)
    h0, h1, h2, h3 = st.columns(col_weights)
    h0.markdown("**Date**")
    h1.markdown("**Score**")
    h2.markdown("**Recommendation**")
    h3.markdown("**Assignment**")

    for rid, date_s, score_s, rec_s, default_assign in assignment_rows:
        c0, c1, c2, c3 = st.columns(col_weights)
        c0.write(date_s)
        c1.write(score_s)
        c2.write(rec_s)
        pick_key = f"pick_{idx}_{rid}"
        idx_opt = select_options.index(default_assign) if default_assign in select_options else 0
        c3.selectbox(
            "Assignment",
            options=select_options,
            index=idx_opt,
            key=pick_key,
            label_visibility="collapsed",
        )

    incomplete = False
    if assignment_rows:
        incomplete = any(
            st.session_state.get(f"pick_{idx}_{rid}", default_assign) == assign_placeholder
            for rid, _, _, _, default_assign in assignment_rows
        )

    if st.button("Submit and next", type="primary", disabled=incomplete):
        for rid, _, _, _, default_assign in assignment_rows:
            pick_key = f"pick_{idx}_{rid}"
            choice = st.session_state.get(pick_key, default_assign)
            rec = rec_map.get(rid)
            df.loc[df["_row_id"] == rid, "recommended_step_title"] = rec
            df.loc[df["_row_id"] == rid, "assigned_step_title"] = (
                pd.NA if choice == assign_placeholder else choice
            )

        sync_tied_assignments(df, sub)

        st.session_state.df = df
        if not save_working_frame(df, pair_user=user_email, pair_domain=domain_title):
            st.stop()
        st.session_state.data_fp = _remote_data_fingerprint()
        st.session_state.pair_idx = idx + 1
        st.rerun()

    if incomplete and assignment_rows:
        st.caption("Set **Assignment** on every row (a step or **Null**) before submitting.")

    st.session_state.df = df


if __name__ == "__main__":
    main()
