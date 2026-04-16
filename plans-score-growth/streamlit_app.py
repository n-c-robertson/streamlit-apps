"""
Learning plan analytics — Streamlit UI.
Data: public Google Sheet (same source as notebook).
"""

from __future__ import annotations

import ast
import base64
import html
import io
import ssl
import urllib.error
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Google Sheet tab for CSV export (URL pattern: …/d/{G_SHEET_ID}/export?format=csv&gid=…)
SHEET_GID = "0"

# ── Style (match notebook) ───────────────────────────────────────────────────
C_PRE = "black"
C_UP = "#2ecc71"
C_DOWN = "#e74c3c"
C_FLAT = "#ADB5BD"
BG = "#FFFFFF"
GRID_CLR = "#E9ECEF"
# Score-delta charts: y=0 — grey darker than GRID_CLR (#E9ECEF) y-axis grid, slightly heavier than grid lw
DELTA_ZERO_LINE_CLR = "#868E96"
DELTA_ZERO_LINE_LW = 1.5
LBL_CLR = "#212529"
N_CLR = "#ADB5BD"

# Figure height vs original baselines (1.75 = 75% taller charts).
FIG_H_SCALE = 2

# Legend above axes, top-right, horizontal (wide ncol); bbox anchor at upper-right of axes; tight_layout top reserves margin.
LEGEND_ABOVE_TR_KW = dict(
    loc="lower right",
    bbox_to_anchor=(1.0, 1.10),
    borderaxespad=0,
    fontsize=10,
    framealpha=0.95,
    edgecolor=GRID_CLR,
    facecolor=BG,
)
# Figure coords: room for ticks + xlabel (see tight_layout rect).
XLABEL_PAD = 12
# Subplots fit below this (leave figure top for title / subtitle / legend above axes).
TIGHT_LAYOUT_RECT_TOP = 0.78

# User-facing definition: pre-assessment uses the earliest recorded score; post-assessment uses the latest.
Y_SCORE_DELTA_LABEL = "Score delta (latest post − earliest pre)"


def show_matplotlib_svg(fig: plt.Figure) -> None:
    """
    Display matplotlib output as SVG in Streamlit (vector graphics — sharp at any zoom/DPI).
    Uses an inline image so the figure scales with the app column width.
    """
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="svg",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    st.markdown(
        f'<img src="data:image/svg+xml;base64,{b64}" '
        f'style="width:100%;height:auto;display:block;" alt="" />',
        unsafe_allow_html=True,
    )


def chart_interpret_help(body: str, *, key: str) -> None:
    """Expandable “How do I interpret this?” copy for GTM readers."""
    with st.expander("How do I interpret this?", expanded=False):
        st.markdown(body)


# Plain-language chart guides (GTM-friendly).
_HELP_PRE_POST = """
**What you’re looking at**

Each **row** is one **assessment area** (domain). The horizontal axis is the **score scale** (0–300).

- **White dot** = **pre-assessment**: **earliest** recorded score we have for that stage.  
- **Green or red dot** = **post-assessment**: **latest** recorded score.  
- **Line** = movement from earliest pre to latest post (**up** = green, **down** = red, **flat** = gray).

The faint **bands** along the bottom are score ranges (novice → accomplished)—**context** on the scale, not a pass/fail rule.

**Tip:** Compare rows to see **which topics** moved most for this cohort.
"""

_HELP_SCATTER_DELTA = """
**What you’re looking at**

Each **dot** is **one learner** in **one assessment area**. **Vertical axis:** change from **earliest pre-assessment score** to **latest post-assessment score**. **Horizontal axis:** days between those two snapshots (from 0).

- **Above** the middle line = **gain**; **below** = **loss**; **on** the line = **no change**.  
- **Green / red / gray** = gain / loss / flat.

The **line** and **shaded band** are a rough “typical” pattern—**context**, not a quota.

**Tip:** Look for **clusters**, not single dots, when you tell a story.
"""

_HELP_SCATTER_LESSONS = """
**What you’re reading**

Same as the days chart, but across the bottom you see **how many lessons** were completed (not time).

You’re asking: does **more lesson activity** line up with **larger gains** (dots higher) for this plan?

**Tip:** If dots are scattered, look at the **overall cloud**, not one point.
"""

_HELP_BOX_PROJECTS = """
**What you’re reading**

Each **column** is a **project count**. For each count, you see how **score change** (latest post minus earliest pre) is spread out.

- **Box** = middle half of changes (25th–75th percentile).  
- **Middle line** = median.  
- **Whiskers** = spread to “normal” extremes (dots beyond that are uncommon).  
- **Dots** = individual learners (jittered so they don’t overlap).

**Tip:** Compare **columns** to see if gains look **stronger or steadier** when people complete more projects.
"""

_HELP_SCATTER_DELTA_DOM = """
Same chart as **section 1**, but only **{assessment}**—no other domains mixed in.

**Use it** for a **clean screenshot** or story focused on this topic.
"""

_HELP_SCATTER_LESSONS_DOM = """
Same as the **lessons** chart in section 1, filtered to **{assessment}**.

**Use it** to relate **lesson activity** to **score change** for this assessment only.
"""

_HELP_BOX_PROJECTS_DOM = """
Same **box + dots** chart as section 1, for **{assessment}** only.

**Use it** to compare **project counts** and **how gains vary** for a single topic.
"""

_HELP_SKILL = """
**What you’re looking at**

Each **row** is **one skill** (0–100% proficiency).

- **White dot** = **earliest pre-assessment** skill tags.  
- **Colored dot** = **latest post-assessment** tags when we have them.  
- **Line** = change on that skill.

**Tip:** Good for naming **specific skills** in customer stories.
"""


def _read_public_sheet_csv(url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(url)
    except (urllib.error.URLError, OSError) as e:
        msg = str(e).lower()
        if "certificate" not in msg and "ssl" not in msg:
            raise
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, context=ctx, timeout=120) as resp:
            return pd.read_csv(io.BytesIO(resp.read()))


def _dataframe_for_csv_download(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a CSV-safe copy: datetimes as YYYY-MM-DD HH:MM:SS strings (UTC-normalized),
    and pre/post attempt dates fall back to preserved sheet text when parsed value is NaT.
    """
    out = df.copy()

    def _fmt_dt_series(s: pd.Series) -> pd.Series:
        ser = pd.to_datetime(s, errors="coerce", utc=True)
        str_dt = ser.dt.strftime("%Y-%m-%d %H:%M:%S")
        return str_dt.where(ser.notna(), "")

    for c in ("pre_attempt_date", "post_attempt_date"):
        if c not in out.columns:
            continue
        sheet_col = f"{c}_sheet"
        dt = pd.to_datetime(out[c], errors="coerce", utc=True)
        str_dt = dt.dt.strftime("%Y-%m-%d %H:%M:%S")
        if sheet_col in out.columns:
            fb = (
                out[sheet_col]
                .astype(str)
                .replace({"nan": "", "NaT": "", "<NA>": "", "None": ""})
            )
            out[c] = str_dt.where(dt.notna(), fb)
            out = out.drop(columns=[sheet_col])
        else:
            out[c] = str_dt.where(dt.notna(), "")

    for col in list(out.columns):
        s = out[col]
        if not pd.api.types.is_datetime64_any_dtype(s):
            continue
        out[col] = _fmt_dt_series(s)

    return out


def _format_sheet_last_updated(value) -> str | None:
    """Normalize Google Sheet `last_updated` field for display."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().strip('"').strip("'")
    if not s or s.lower() == "nan":
        return None
    try:
        dt = pd.to_datetime(s, utc=True)
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
        return s


@st.cache_data(ttl=300)
def load_long_dataframe(sheet_id: str) -> tuple[pd.DataFrame, str | None]:
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={SHEET_GID}"
    raw = _read_public_sheet_csv(csv_url)
    last_updated_display = None
    if "last_updated" in raw.columns:
        lu = raw["last_updated"].dropna()
        if not lu.empty:
            last_updated_display = _format_sheet_last_updated(lu.iloc[0])

    # Preserve original sheet text so CSV export always has human-readable dates if parsing yields NaT
    for _dc in ("pre_attempt_date", "post_attempt_date"):
        if _dc in raw.columns:
            raw[f"{_dc}_sheet"] = raw[_dc].map(lambda x: "" if pd.isna(x) else str(x).strip())

    raw["pre_attempt_date"] = pd.to_datetime(raw["pre_attempt_date"], errors="coerce")
    raw["post_attempt_date"] = pd.to_datetime(raw["post_attempt_date"], errors="coerce")

    pre = raw.assign(
        course=raw["domain_title"],
        type="pre",
        score=raw["pre_score"],
        workera_created_at=raw["pre_attempt_date"],
        strong_skills=raw["pre_strong_skills"],
        needs_improvement_skills=raw["pre_needs_improvement_skills"],
    )
    post = raw.assign(
        course=raw["domain_title"],
        type="post",
        score=raw["post_score"],
        workera_created_at=raw["post_attempt_date"],
        strong_skills=raw["post_strong_skills"] if "post_strong_skills" in raw.columns else np.nan,
        needs_improvement_skills=raw["post_needs_improvement_skills"]
        if "post_needs_improvement_skills" in raw.columns
        else np.nan,
    )
    df_long = pd.concat([pre, post], ignore_index=True)
    return df_long.dropna(subset=["course"]), last_updated_display


def compute_pivot_all_users(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values("workera_created_at")
    pre_first = (
        df_sorted[df_sorted["type"] == "pre"]
        .groupby(["plan_title", "course", "workera_user_email"])["score"]
        .first()
        .reset_index()
    )
    pre = pre_first.groupby(["plan_title", "course"])["score"].mean().rename("pre")
    post = df_sorted[df_sorted["type"] == "post"].groupby(["plan_title", "course"])["score"].mean().rename("post")
    pre_n = pre_first.groupby(["plan_title", "course"])["workera_user_email"].nunique().rename("pre_n")
    post_n = (
        df_sorted[df_sorted["type"] == "post"]
        .groupby(["plan_title", "course"])["workera_user_email"]
        .nunique()
        .rename("post_n")
    )
    pivot = pd.concat([pre, post, pre_n, post_n], axis=1).reset_index()
    pivot.columns.name = None
    return pivot.sort_values("pre").reset_index(drop=True)


def plot_assessment_chart(pivot: pd.DataFrame, title: str, subtitle: str, figsize=(14.3, 6 * FIG_H_SCALE)) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "SF Pro Display", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
        }
    )
    n = len(pivot)
    if n == 0:
        fig, ax = plt.subplots(figsize=figsize, dpi=400)
        ax.text(0.5, 0.5, "No rows to plot", ha="center", va="center")
        return fig

    h = max(4.0, 0.45 * n + 2.0) * FIG_H_SCALE
    fig, ax = plt.subplots(figsize=(figsize[0], min(h, 24 * FIG_H_SCALE)), dpi=400)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    y_positions = np.arange(n)
    for y, (_, row) in enumerate(pivot.iterrows()):
        pre_val = row.get("pre", np.nan)
        post_val = row.get("post", np.nan)
        pren = int(row.get("pre_n", 1))
        postn = int(row.get("post_n", 1))

        pre_size = 40 + np.sqrt(pren) * 20
        post_size = 40 + np.sqrt(postn) * 20

        if pd.notna(pre_val) and pd.notna(post_val):
            lc = C_UP if post_val > pre_val else (C_DOWN if post_val < pre_val else C_FLAT)
            ax.plot([pre_val, post_val], [y, y], color=lc, linewidth=1.8, zorder=1, alpha=0.75, solid_capstyle="round")

        if pd.notna(pre_val):
            ax.scatter(pre_val, y, color="white", s=pre_size, zorder=3, edgecolors="black", linewidth=1.2)
            pre_lx, pre_ha = (pre_val + 5, "left") if pre_val <= 10 else (pre_val - 5, "right")
            ax.text(pre_lx, y + 0.22, f"{pre_val:.0f}", va="bottom", ha=pre_ha, fontsize=9, color=C_PRE, fontweight="bold")
            ax.text(pre_lx, y - 0.22, f"n={pren}", va="top", ha=pre_ha, fontsize=7.5, color=N_CLR)

        if pd.notna(post_val):
            post_fill = C_DOWN if (pd.notna(pre_val) and post_val < pre_val) else C_UP
            ax.scatter(post_val, y, color=post_fill, s=post_size, zorder=3, edgecolors="white", linewidth=1.2)
            post_lx, post_ha = (post_val + 5, "left")
            ax.text(post_lx, y + 0.22, f"{post_val:.0f}", va="bottom", ha=post_ha, fontsize=9, color=post_fill, fontweight="bold")
            ax.text(post_lx, y - 0.22, f"n={postn}", va="top", ha=post_ha, fontsize=7.5, color=N_CLR)

    ax.set_yticks(y_positions)
    # Only y-axis row labels smaller (−4 pt) to save vertical space
    ax.set_yticklabels(pivot["course"], fontsize=6, color=LBL_CLR)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Average Score (0 – 300)", fontsize=11, labelpad=XLABEL_PAD, color=LBL_CLR)
    ax.set_xlim(0, 300)
    ax.xaxis.grid(True, linestyle="-", alpha=1.0, color=GRID_CLR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID_CLR)
    ax.tick_params(axis="both", colors=LBL_CLR, length=0)

    _skill_bands = [
        (0, 10, "#868E96", "Novice"),
        (11, 100, "#E67700", "Beginner"),
        (101, 200, "#2F9E44", "Intermediate"),
        (201, 300, "#1971C2", "Accomplished"),
    ]
    for _x0, _x1, _color, _label in _skill_bands:
        ax.axvspan(_x0, _x1, alpha=0.06, color=_color, zorder=0.9, lw=0)
        ax.text(
            (_x0 + _x1) / 2,
            1.01,
            _label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color=_color,
            fontweight="bold",
            clip_on=False,
        )

    ax.text(0.0, 1.095, title, transform=ax.transAxes, va="bottom", ha="left", fontsize=14, fontweight="bold", clip_on=False, color=LBL_CLR)
    ax.text(0.0, 1.045, subtitle, transform=ax.transAxes, va="bottom", ha="left", fontsize=10.5, color="#6B7280", clip_on=False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.5,
            markersize=8,
            label="Pre (earliest pre-assessment)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=C_UP,
            markeredgecolor="white",
            markeredgewidth=1.0,
            markersize=8,
            label="Post — improved / flat (latest)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=C_DOWN,
            markeredgecolor="white",
            markeredgewidth=1.0,
            markersize=8,
            label="Post — declined (latest)",
        ),
    ]
    ax.legend(handles=legend_handles, ncol=len(legend_handles), **LEGEND_ABOVE_TR_KW)
    plt.tight_layout(rect=[0, 0, 1, TIGHT_LAYOUT_RECT_TOP])
    return fig


# Column names for score-delta scatter X axes (wide sheet fields, carried on long-format pre rows)
SCATTER_X_DAYS = "days_between_attempts"
SCATTER_X_LESSONS = "total_lessons_completed"
SCATTER_X_PROJECTS = "total_projects_passed"


def apply_max_pre_score_filter(df: pd.DataFrame, max_pre_allowed: float) -> pd.DataFrame:
    """
    Drop long-format rows where sheet `pre_score` is strictly above the cap (applies to both
    pre and post rows, which carry the same wide-row pre_score).
    Rows with missing pre_score are kept.
    """
    if df.empty or "pre_score" not in df.columns:
        return df
    ps = pd.to_numeric(df["pre_score"], errors="coerce")
    return df.loc[~(ps > float(max_pre_allowed))].copy()


def apply_min_pre_score_filter(df: pd.DataFrame, min_pre_allowed: float) -> pd.DataFrame:
    """
    Drop long-format rows where sheet `pre_score` is strictly below the floor (applies to both
    pre and post rows, which carry the same wide-row pre_score).
    Rows with missing pre_score are kept.
    """
    if df.empty or "pre_score" not in df.columns:
        return df
    ps = pd.to_numeric(df["pre_score"], errors="coerce")
    return df.loc[~(ps < float(min_pre_allowed))].copy()


def apply_advanced_pre_score_filters(df: pd.DataFrame, max_cap: float, min_floor: float) -> pd.DataFrame:
    """Apply max cap and min floor in one pass (same rules as the two helpers). Used for all charts + CSV."""
    out = apply_max_pre_score_filter(df, float(max_cap))
    return apply_min_pre_score_filter(out, float(min_floor))


def prepare_score_delta_scatter_df(df_plan: pd.DataFrame, x_col: str) -> pd.DataFrame:
    """One row per learner × domain (pre row: earliest snapshot) with numeric x_col and score_delta."""
    if not {x_col, "score_delta"}.issubset(df_plan.columns):
        return pd.DataFrame()
    pre = df_plan[df_plan["type"] == "pre"].sort_values("workera_created_at")
    pre = pre.groupby(["plan_title", "course", "workera_user_email"], as_index=False).first()
    pre[x_col] = pd.to_numeric(pre[x_col], errors="coerce")
    pre["score_delta"] = pd.to_numeric(pre["score_delta"], errors="coerce")
    return pre.dropna(subset=[x_col, "score_delta"])


def plot_score_delta_scatter(
    scatter_df: pd.DataFrame,
    x_col: str,
    x_label: str,
    title: str,
    subtitle: str,
) -> plt.Figure:
    """Scatter: X = chosen metric column, Y = score delta (same colors/grid/trend as days chart)."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "SF Pro Display", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
        }
    )
    if scatter_df.empty or x_col not in scatter_df.columns:
        fig, ax = plt.subplots(figsize=(14.3, 6 * FIG_H_SCALE), dpi=400)
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        ax.text(
            0.5,
            0.5,
            f'No rows with both "{x_label}" and score delta.',
            ha="center",
            va="center",
            color=LBL_CLR,
        )
        ax.set_axis_off()
        return fig

    fig, ax = plt.subplots(figsize=(14.3, 6 * FIG_H_SCALE), dpi=400)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    x = scatter_df[x_col].to_numpy(dtype=float)
    y = scatter_df["score_delta"].to_numpy(dtype=float)
    colors = np.where(y > 0, C_UP, np.where(y < 0, C_DOWN, C_FLAT))

    ax.axhline(0.0, color=DELTA_ZERO_LINE_CLR, linewidth=DELTA_ZERO_LINE_LW, zorder=1, linestyle="-")

    x_max_data = float(np.nanmax(x))
    x_right = max(x_max_data * 1.05, x_max_data + 1.0)

    # Linear trend + middle-50% band (25th–75th percentile of residuals around the trend)
    if len(x) >= 2 and np.nanstd(x) > 1e-9:
        coef = np.polyfit(x, y, 1)
        x_line = np.linspace(0.0, x_right, 200)
        y_line = np.polyval(coef, x_line)
        y_hat = np.polyval(coef, x)
        resid = y - y_hat
        r25, r75 = np.percentile(resid, [25, 75])
        y_low = y_line + r25
        y_high = y_line + r75
        ax.fill_between(
            x_line,
            y_low,
            y_high,
            color=LBL_CLR,
            alpha=0.3,
            zorder=1,
            linewidth=0,
        )
        ax.plot(
            x_line,
            y_line,
            color=LBL_CLR,
            linewidth=2.0,
            linestyle="-",
            zorder=2,
            alpha=0.95,
        )

    ax.scatter(
        x,
        y,
        c=colors,
        s=70,
        zorder=3,
        edgecolors="white",
        linewidths=1.0,
        alpha=0.92,
    )

    ax.set_xlim(0.0, x_right)
    ax.set_xlabel(x_label, fontsize=11, labelpad=XLABEL_PAD, color=LBL_CLR)
    ax.set_ylabel(Y_SCORE_DELTA_LABEL, fontsize=11, labelpad=8, color=LBL_CLR)
    ax.xaxis.grid(True, linestyle="-", alpha=1.0, color=GRID_CLR, linewidth=0.8)
    ax.yaxis.grid(True, linestyle="-", alpha=1.0, color=GRID_CLR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID_CLR)
    ax.tick_params(axis="both", colors=LBL_CLR, length=0)

    ax.text(0.0, 1.095, title, transform=ax.transAxes, va="bottom", ha="left", fontsize=14, fontweight="bold", clip_on=False, color=LBL_CLR)
    ax.text(0.0, 1.045, subtitle, transform=ax.transAxes, va="bottom", ha="left", fontsize=10.5, color="#6B7280", clip_on=False)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_UP, markeredgecolor="white", markeredgewidth=1.0, markersize=8, label="Gain (latest > earliest)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_DOWN, markeredgecolor="white", markeredgewidth=1.0, markersize=8, label="Loss (latest < earliest)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_FLAT, markeredgecolor="white", markeredgewidth=1.0, markersize=8, label="No change"),
    ]
    ax.legend(handles=legend_handles, ncol=len(legend_handles), **LEGEND_ABOVE_TR_KW)

    plt.tight_layout(rect=[0, 0, 1, TIGHT_LAYOUT_RECT_TOP])
    return fig


def plot_score_delta_box_jitter(
    scatter_df: pd.DataFrame,
    x_col: str,
    x_label: str,
    title: str,
    subtitle: str,
) -> plt.Figure:
    """
    Box-and-whisker of score delta by discrete X (e.g. projects completed), plus jittered points
    per learner × domain row. X groups are rounded integer counts.
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "SF Pro Display", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
        }
    )
    if scatter_df.empty or x_col not in scatter_df.columns:
        fig, ax = plt.subplots(figsize=(14.3, 6 * FIG_H_SCALE), dpi=400)
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        ax.text(
            0.5,
            0.5,
            f'No rows with both "{x_label}" and score delta.',
            ha="center",
            va="center",
            color=LBL_CLR,
        )
        ax.set_axis_off()
        return fig

    df = scatter_df[[x_col, "score_delta"]].copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df["score_delta"] = pd.to_numeric(df["score_delta"], errors="coerce")
    df = df.dropna(subset=[x_col, "score_delta"])
    df["_g"] = df[x_col].round().astype("Int64")
    df = df[df["_g"].notna()]
    if df.empty:
        fig, ax = plt.subplots(figsize=(14.3, 6 * FIG_H_SCALE), dpi=400)
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        ax.text(0.5, 0.5, "No valid rows after grouping.", ha="center", va="center", color=LBL_CLR)
        ax.set_axis_off()
        return fig

    groups = sorted(df["_g"].astype(int).unique())
    data = [df.loc[df["_g"] == g, "score_delta"].to_numpy(dtype=float) for g in groups]
    positions = [float(g) for g in groups]
    pos_arr = np.array(positions, dtype=float)
    if len(pos_arr) > 1:
        min_gap = float(np.min(np.diff(np.sort(pos_arr))))
        box_w = min(0.45, max(0.18, 0.32 * min_gap))
    else:
        box_w = 0.45

    fig, ax = plt.subplots(figsize=(14.3, 6 * FIG_H_SCALE), dpi=400)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.axhline(0.0, color=DELTA_ZERO_LINE_CLR, linewidth=DELTA_ZERO_LINE_LW, zorder=1, linestyle="-")

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=box_w,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": LBL_CLR, "linewidth": 1.6},
        boxprops={"facecolor": "#F1F3F5", "edgecolor": GRID_CLR, "linewidth": 1.0},
        whiskerprops={"color": LBL_CLR, "linewidth": 1.0},
        capprops={"color": LBL_CLR, "linewidth": 1.0},
        zorder=2,
    )

    rng = np.random.default_rng(42)
    jitter_hw = min(0.14, 0.22 * box_w)
    for pos, ys in zip(positions, data):
        if len(ys) == 0:
            continue
        jx = pos + rng.uniform(-jitter_hw, jitter_hw, size=len(ys))
        cols = np.where(ys > 0, C_UP, np.where(ys < 0, C_DOWN, C_FLAT))
        ax.scatter(
            jx,
            ys,
            c=cols,
            s=55,
            zorder=3,
            edgecolors="white",
            linewidths=0.9,
            alpha=0.88,
        )

    x_min = min(positions) - 0.55
    x_max = max(positions) + 0.55
    span = x_max - x_min or 1.0
    pad = 0.1 * span
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(int(p)) for p in positions], color=LBL_CLR)
    if len(groups) > 16:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right")

    ax.set_xlabel(x_label, fontsize=11, labelpad=XLABEL_PAD, color=LBL_CLR)
    ax.set_ylabel(Y_SCORE_DELTA_LABEL, fontsize=11, labelpad=8, color=LBL_CLR)
    ax.xaxis.grid(True, linestyle="-", alpha=1.0, color=GRID_CLR, linewidth=0.8)
    ax.yaxis.grid(True, linestyle="-", alpha=1.0, color=GRID_CLR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID_CLR)
    ax.tick_params(axis="both", colors=LBL_CLR, length=0)

    ax.text(0.0, 1.095, title, transform=ax.transAxes, va="bottom", ha="left", fontsize=14, fontweight="bold", clip_on=False, color=LBL_CLR)
    ax.text(0.0, 1.045, subtitle, transform=ax.transAxes, va="bottom", ha="left", fontsize=10.5, color="#6B7280", clip_on=False)

    legend_handles = [
        Patch(facecolor="#F1F3F5", edgecolor=GRID_CLR, linewidth=1.0, label="Box & whiskers (median, IQR, range)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_UP, markeredgecolor="white", markeredgewidth=1.0, markersize=8, label="Gain (latest > earliest)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_DOWN, markeredgecolor="white", markeredgewidth=1.0, markersize=8, label="Loss (latest < earliest)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_FLAT, markeredgecolor="white", markeredgewidth=1.0, markersize=8, label="No change"),
    ]
    ax.legend(handles=legend_handles, ncol=len(legend_handles), **LEGEND_ABOVE_TR_KW)

    plt.tight_layout(rect=[0, 0, 1, TIGHT_LAYOUT_RECT_TOP])
    return fig


def parse_skills(s):
    if pd.isna(s):
        return []
    try:
        return [html.unescape(x) for x in ast.literal_eval(s)]
    except Exception:
        return []


def _row_assessment_scores(row: pd.Series) -> tuple[float, float]:
    """
    Read domain pre/post scores for skill proxying.

    Long-format rows are built from wide sheet rows, so each `type == "pre"` row still carries
    `pre_score` and `post_score` from the export (the same values chart 1 uses via `score`).

    If a buggy merge created `pre_score_x` / `pre_score_y`, fall back to those names.
    """
    def _pick(*names: str) -> float:
        for name in names:
            if name not in row.index:
                continue
            v = row[name]
            if pd.isna(v):
                continue
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
        return np.nan

    ps = _pick("pre_score", "pre_score_x", "pre_score_y")
    qs = _pick("post_score", "post_score_x", "post_score_y")
    return ps, qs


def skill_pivot_pre_post_proxy(pre_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Pre skill rates from pre-assessment tags. Post rates use the same tags plus each user's
    (post_score − pre_score) / 300 shift so dots and lines match domain growth/decline when
    per-skill post tags are missing in the sheet (same idea as chart 1, on 0–1 skill scale).

    Uses `pre_score` / `post_score` from the sheet-wide columns on each pre row (see load_long_dataframe).
    """
    if pre_rows.empty:
        return pd.DataFrame(columns=["skill", "pre", "post", "pre_n", "post_n"])

    m = pre_rows.drop(columns=["score"], errors="ignore")

    rec: list[dict] = []
    for _, row in m.iterrows():
        ps, qs = _row_assessment_scores(row)
        delta = (qs - ps) / 300.0 if pd.notna(ps) and pd.notna(qs) else np.nan

        for sk in row["strong_list"]:
            pre_v = 1.0
            post_v = float(np.clip(pre_v + delta, 0.0, 1.0)) if pd.notna(delta) else np.nan
            rec.append({"skill": sk, "pre": pre_v, "post": post_v})

        for sk in row["weak_list"]:
            pre_v = 0.0
            post_v = float(np.clip(pre_v + delta, 0.0, 1.0)) if pd.notna(delta) else np.nan
            rec.append({"skill": sk, "pre": pre_v, "post": post_v})

    if not rec:
        return pd.DataFrame(columns=["skill", "pre", "post", "pre_n", "post_n"])

    t = pd.DataFrame(rec)
    rows = []
    for skill, x in t.groupby("skill"):
        rows.append(
            {
                "skill": skill,
                "pre": x["pre"].mean(),
                "post": x["post"].mean(),
                "pre_n": len(x),
                "post_n": int(x["post"].notna().sum()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("pre").reset_index(drop=True)


def plot_skill_chart(pivot: pd.DataFrame, title: str, subtitle: str) -> plt.Figure:
    SKILL_BANDS = [
        (0, 25, "#868E96", "Needs Improvement"),
        (25, 50, "#E67700", "Room for Growth"),
        (50, 75, "#2F9E44", "Proficient"),
        (75, 100, "#1971C2", "Expert"),
    ]
    n = len(pivot)
    if n == 0:
        fig, ax = plt.subplots(figsize=(14.3, 4 * FIG_H_SCALE), dpi=400)
        ax.text(0.5, 0.5, "No skill-level data for this assessment.", ha="center", va="center")
        return fig

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "SF Pro Display", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
        }
    )
    fig_h = max(6, 0.50 * n + 2) * FIG_H_SCALE
    fig, ax = plt.subplots(figsize=(16.9, min(fig_h, 28 * FIG_H_SCALE)), dpi=400)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(-4, 108)
    ax.set_yticks(range(n))
    ax.set_yticklabels(pivot["skill"], fontsize=10, color=LBL_CLR)
    fig.canvas.draw()
    _bbox = ax.get_window_extent()
    _x0, _x1 = ax.get_xlim()
    _dx_5px = (_x1 - _x0) * (5.0 / _bbox.width)

    for idx, (_, row) in enumerate(pivot.iterrows()):
        pre_v = row.get("pre")
        post_v = row.get("post")
        pren = int(row["pre_n"]) if pd.notna(row.get("pre_n")) else 0
        postn = int(row["post_n"]) if pd.notna(row.get("post_n")) else 0
        sz_pre = 40 + np.sqrt(pren) * 20
        sz_post = 40 + np.sqrt(postn) * 20
        pre_pct = pre_v * 100 if pd.notna(pre_v) else pre_v
        post_pct = post_v * 100 if pd.notna(post_v) else post_v

        pre_draw, post_draw = pre_pct, post_pct
        if pd.notna(pre_pct) and pd.notna(post_pct):
            diff = abs(float(pre_pct) - float(post_pct))
            if diff <= 5.0 and float(pre_pct) > float(post_pct):
                pre_draw = float(pre_pct) + _dx_5px
                post_draw = float(post_pct) - _dx_5px
            elif diff <= 5.0 and float(pre_pct) < float(post_pct):
                pre_draw = float(pre_pct) - _dx_5px
                post_draw = float(post_pct) + _dx_5px

        if pd.notna(pre_pct) and pd.notna(post_pct):
            lc = C_UP if post_pct > pre_pct else (C_DOWN if post_pct < pre_pct else C_FLAT)
            ax.plot(
                [pre_draw, post_draw],
                [idx, idx],
                color=lc,
                linewidth=1.8,
                zorder=1,
                alpha=0.75,
                solid_capstyle="round",
            )

        post_fill = C_DOWN if (pd.notna(post_pct) and pd.notna(pre_pct) and post_pct < pre_pct) else C_UP
        off = 2.2
        if pd.notna(pre_pct):
            ax.scatter(pre_draw, idx, color="white", s=sz_pre, zorder=3, edgecolors="black", linewidth=1.2)
            lx, ha = (pre_draw + off, "left") if pre_draw <= 4 else (pre_draw - off, "right")
            ax.text(lx, idx + 0.22, f"{pre_pct:.0f}%", va="bottom", ha=ha, fontsize=9, fontweight="bold", color=C_PRE)
            ax.text(lx, idx - 0.22, f"n={pren}", va="top", ha=ha, fontsize=7.5, color=N_CLR)

        if pd.notna(post_pct):
            ax.scatter(post_draw, idx, color=post_fill, s=sz_post, zorder=3, edgecolors="white", linewidth=1.2)
            lx, ha = (post_draw - off, "right") if post_draw >= 96 else (post_draw + off, "left")
            ax.text(lx, idx + 0.22, f"{post_pct:.0f}%", va="bottom", ha=ha, fontsize=9, fontweight="bold", color=post_fill)
            ax.text(lx, idx - 0.22, f"n={postn}", va="top", ha=ha, fontsize=7.5, color=N_CLR)

    ax.set_xlabel("Skill Proficiency Rate (%)", fontsize=11, labelpad=XLABEL_PAD, color=LBL_CLR)
    ax.set_xlim(-4, 108)
    ax.xaxis.grid(True, linestyle="-", alpha=1.0, color=GRID_CLR, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID_CLR)
    ax.tick_params(axis="both", colors=LBL_CLR, length=0)

    for x0, x1, col, lbl in SKILL_BANDS:
        ax.axvspan(x0, x1, alpha=0.06, color=col, zorder=0.9, lw=0)
        ax.text(
            (x0 + x1) / 2,
            1.01,
            lbl,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color=col,
            fontweight="bold",
            clip_on=False,
        )

    ax.text(0.0, 1.095, title, transform=ax.transAxes, va="bottom", ha="left", fontsize=14, fontweight="bold", clip_on=False, color=LBL_CLR)
    ax.text(0.0, 1.065, subtitle, transform=ax.transAxes, va="bottom", ha="left", fontsize=10.5, color="#6B7280", clip_on=False)

    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.5, markersize=8, label="Pre (earliest pre-assessment)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=C_UP, markeredgecolor="white", markeredgewidth=1.0, markersize=8, label="Post — improved / flat (latest)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=C_DOWN, markeredgecolor="white", markeredgewidth=1.0, markersize=8, label="Post — declined (latest)"),
        ],
        ncol=3,
        **LEGEND_ABOVE_TR_KW,
    )
    plt.tight_layout(rect=[0, 0, 1, TIGHT_LAYOUT_RECT_TOP])
    return fig


def main():
    st.set_page_config(page_title="Learning plan analytics", layout="wide")
    st.title("Learning plan analytics")

    try:
        app_password = st.secrets["APP_PASSWORD"]
    except KeyError:
        st.error("Missing Streamlit secret `APP_PASSWORD`. Set it in app settings or `.streamlit/secrets.toml`.")
        st.stop()

    if "auth" not in st.session_state:
        st.session_state.auth = False

    if not st.session_state.auth:
        st.subheader("Sign in")
        pw = st.text_input("Password", type="password", autocomplete="current-password")
        if st.button("Continue"):
            if pw == app_password:
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        return

    try:
        sheet_id = st.secrets["G_SHEET_ID"]
    except KeyError:
        st.error("Missing Streamlit secret `G_SHEET_ID`. Set it in app settings or `.streamlit/secrets.toml`.")
        st.stop()

    try:
        df_long, sheet_last_updated = load_long_dataframe(sheet_id)
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return

    if sheet_last_updated:
        st.caption(f"Data last updated (from source sheet): **{sheet_last_updated}**")

    with st.expander("Advanced filters", expanded=False):
        max_pre_cap = st.number_input(
            "Exclude rows with earliest pre-assessment score above",
            min_value=0,
            max_value=300,
            value=300,
            step=1,
            key="filter_max_pre_score",
            help=(
                "Drops learner × domain rows whose **earliest pre-assessment score** is strictly above this value "
                "(applies to both row types for that assessment). Use 300 to include everyone on a 0–300 scale. "
                "Applies to **every chart** and the CSV download."
            ),
        )
        min_pre_floor = st.number_input(
            "Exclude rows with earliest pre-assessment score below",
            min_value=0,
            max_value=300,
            value=0,
            step=1,
            key="filter_min_pre_score",
            help=(
                "Drops learner × domain rows whose **earliest pre-assessment score** is strictly below this value. "
                "Use 0 for no lower bound. Applies to **every chart** and the CSV download."
            ),
        )

    df_filtered = apply_advanced_pre_score_filters(df_long, float(max_pre_cap), float(min_pre_floor))

    plans = sorted(df_filtered["plan_title"].dropna().unique())
    if not plans:
        st.warning("No learning plans left after applying the advanced filters. Loosen the score range and try again.")
        return

    # If the current plan vanished from the list (filters tightened), reset selection
    if st.session_state.get("plan_select") not in plans:
        st.session_state.plan_select = plans[0]

    plan = st.selectbox("Learning plan", plans, key="plan_select")

    df_plan = df_filtered[df_filtered["plan_title"] == plan].copy()

    safe_name = "".join(
        c if c.isalnum() or c in (" ", "-", "_") else "_" for c in str(plan)
    ).strip().replace(" ", "_")[:120] or "data"
    csv_str = _dataframe_for_csv_download(df_plan).to_csv(index=False)
    csv_bytes = csv_str.encode("utf-8-sig")
    st.download_button(
        label="Download filtered raw data (CSV)",
        data=csv_bytes,
        file_name=f"learning_plan_{safe_name}.csv",
        mime="text/csv",
        help=(
            "Filtered table for this plan: pre rows (earliest pre-assessment snapshot) and post rows "
            "(latest post-assessment snapshot), per learner × domain."
        ),
    )

    st.divider()
    st.subheader("1 · Pre vs post (all assessments in this plan)")
    chart_interpret_help(_HELP_PRE_POST, key="interpret_pre_post")
    pivot = compute_pivot_all_users(df_plan)
    fig1 = plot_assessment_chart(
        pivot,
        "Assessment scores by domain",
        f"Pre vs. post · {plan} · Pre = earliest pre-assessment score · Post = latest post-assessment score",
    )
    show_matplotlib_svg(fig1)

    st.subheader("Score delta vs. days between attempts")
    chart_interpret_help(_HELP_SCATTER_DELTA, key="interpret_scatter_days")
    st.caption(
        "Each point: one learner × domain. **Y:** latest post-assessment score minus earliest pre-assessment score. "
        "**X:** days between those snapshots (from 0). Line + shaded band = typical pattern in the data."
    )
    scatter_dd = prepare_score_delta_scatter_df(df_plan, SCATTER_X_DAYS)
    fig_delta = plot_score_delta_scatter(
        scatter_dd,
        SCATTER_X_DAYS,
        "Days between attempts",
        "Score delta vs. days between attempts",
        f"{plan} · Δ = latest post − earliest pre · one row per learner × domain",
    )
    show_matplotlib_svg(fig_delta)

    st.subheader("Score delta vs. lessons completed")
    chart_interpret_help(_HELP_SCATTER_LESSONS, key="interpret_scatter_lessons")
    st.caption(
        "Each point: one learner × domain. **Y:** latest post minus earliest pre. **X:** lessons completed. "
        "Line + shaded band = typical pattern in the data."
    )
    scatter_lessons = prepare_score_delta_scatter_df(df_plan, SCATTER_X_LESSONS)
    fig_lessons = plot_score_delta_scatter(
        scatter_lessons,
        SCATTER_X_LESSONS,
        "Lessons completed",
        "Score delta vs. lessons completed",
        f"{plan} · Δ = latest post − earliest pre · one row per learner × domain",
    )
    show_matplotlib_svg(fig_lessons)

    st.subheader("Score delta by projects completed (distribution)")
    chart_interpret_help(_HELP_BOX_PROJECTS, key="interpret_box_projects")
    st.caption(
        "**Y:** latest post minus earliest pre per row. **X:** project count (`total_projects_passed`). "
        "Box = median & IQR; whiskers ≈ 1.5×IQR; dots = learners (jittered)."
    )
    scatter_proj = prepare_score_delta_scatter_df(df_plan, SCATTER_X_PROJECTS)
    fig_proj = plot_score_delta_box_jitter(
        scatter_proj,
        SCATTER_X_PROJECTS,
        "Projects completed (passed)",
        "Score delta distribution by project count",
        f"{plan} · Δ = latest post − earliest pre · one row per learner × domain",
    )
    show_matplotlib_svg(fig_proj)

    domains = sorted(df_plan["course"].dropna().unique())
    st.divider()
    st.subheader("2 · Selected assessment")
    assessment = st.selectbox(
        "Individual assessment (domain)",
        domains,
        index=0,
        key="assessment_select",
        help="Skill chart for this domain.",
    )

    df_dom = df_plan[df_plan["course"] == assessment].copy()

    st.subheader("Score delta vs. days — selected domain")
    chart_interpret_help(_HELP_SCATTER_DELTA_DOM.replace("{assessment}", str(assessment)), key="interpret_scatter_days_dom")
    st.caption(
        f"**{assessment}** only. Same definitions as section 1 (latest post − earliest pre vs. days between snapshots)."
    )
    scatter_dom = prepare_score_delta_scatter_df(df_dom, SCATTER_X_DAYS)
    fig_delta_dom = plot_score_delta_scatter(
        scatter_dom,
        SCATTER_X_DAYS,
        "Days between attempts",
        "Score delta vs. days between attempts",
        f"{plan} · {assessment} · Δ = latest post − earliest pre",
    )
    show_matplotlib_svg(fig_delta_dom)

    st.subheader("Score delta vs. lessons — selected domain")
    chart_interpret_help(_HELP_SCATTER_LESSONS_DOM.replace("{assessment}", str(assessment)), key="interpret_scatter_lessons_dom")
    st.caption(
        f"**{assessment}** only. Same definitions as section 1 (latest post − earliest pre vs. lessons completed)."
    )
    scatter_lessons_dom = prepare_score_delta_scatter_df(df_dom, SCATTER_X_LESSONS)
    fig_lessons_dom = plot_score_delta_scatter(
        scatter_lessons_dom,
        SCATTER_X_LESSONS,
        "Lessons completed",
        "Score delta vs. lessons completed",
        f"{plan} · {assessment} · Δ = latest post − earliest pre",
    )
    show_matplotlib_svg(fig_lessons_dom)

    st.subheader("Score delta by projects — selected domain")
    chart_interpret_help(_HELP_BOX_PROJECTS_DOM.replace("{assessment}", str(assessment)), key="interpret_box_projects_dom")
    st.caption(
        f"**{assessment}** only. Distribution of latest post − earliest pre by project count."
    )
    scatter_proj_dom = prepare_score_delta_scatter_df(df_dom, SCATTER_X_PROJECTS)
    fig_proj_dom = plot_score_delta_box_jitter(
        scatter_proj_dom,
        SCATTER_X_PROJECTS,
        "Projects completed (passed)",
        "Score delta distribution by project count",
        f"{plan} · {assessment} · Δ = latest post − earliest pre",
    )
    show_matplotlib_svg(fig_proj_dom)

    df_c = df_dom.copy()
    df_c["strong_list"] = df_c["strong_skills"].apply(parse_skills)
    df_c["weak_list"] = df_c["needs_improvement_skills"].apply(parse_skills)

    pre_all = (
        df_c[df_c["type"] == "pre"].sort_values("workera_created_at").groupby("workera_user_email", as_index=False).first()
    )
    post_all = df_c[df_c["type"] == "post"].copy()
    matched_u = set(pre_all["workera_user_email"]) & set(post_all["workera_user_email"])
    pre_mp = pre_all[pre_all["workera_user_email"].isin(matched_u)]

    pivot_mp = skill_pivot_pre_post_proxy(pre_mp)

    st.subheader("Skill breakdown")
    chart_interpret_help(_HELP_SKILL, key="interpret_skill")
    st.caption(
        "White = pre-assessment (earliest) skill tags; colored = latest post-assessment tags when available."
    )

    fig_skill = plot_skill_chart(
        pivot_mp,
        f"Skill proficiency — {plan} — {assessment}",
        "Learners with pre & post · Latest post-assessment",
    )
    show_matplotlib_svg(fig_skill)

    if st.button("Sign out"):
        st.session_state.auth = False
        st.rerun()


if __name__ == "__main__":
    main()
