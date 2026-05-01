# Assessment performance analytics across a fixed set of assessments (staff only).

from collections import Counter

import pandas as pd
import streamlit as st
import utils_assessment_analysis


@st.cache_data(ttl=7200, show_spinner=False)
def _cached_assessment_titles(ids_tuple):
    return utils_assessment_analysis.fetch_assessment_titles_map(list(ids_tuple))


ADMIN_ASSESSMENT_IDS = (
    "e40cedb5-6658-49d5-9de8-fe18c08fa0e9",
    "2ca0d560-0a85-4525-bf3e-ca17b7b384ee",
    "3dffaee0-f987-409e-8dc9-86a5e8e34c88",
    "98d0e7f1-f288-4b82-8092-09b860a0e062",
    "d662a9b5-c52d-4d8b-9f3f-2a0fc5e9c823",
    "b8a665ee-8098-4d90-b2dd-4b1fbf9ddaa7",
    "df89501b-bf2a-46ac-a991-d8e2942906c2",
    "a2cccf2c-f650-40b3-af23-f8780550c55c",
    "7a6ea823-704a-422d-b482-2209cceb1192",
    "d3a8269f-32ab-4795-aab1-c7610132279e",
    "da24aa70-721a-429d-bb15-48f0d931b1da",
    "ba879186-7ec6-48b0-950b-6b848db57073",
    "92b81dee-0980-4f42-adec-85757f02e5a1",
    "2600fa44-752d-4d8d-afb8-9bbcbf596a92",
    "af4f0000-a705-432d-88b5-991998ac4a6b",
    "963a85a4-9927-463e-b7c4-882806a4de05",
    "df72b3ae-58a5-47e9-b9a7-542ea9094590",
    "66d36ee7-1b16-4c07-a9a7-9c61f567310d",
    "901160fa-815f-4d5c-9b77-5da1709a3796",
    "08ddc21a-b70c-4270-9067-08cbb23e4f1e",
    "6bdd3732-e5b3-4bf2-befd-b6589f57547d",
    "6677bc06-2b81-457c-8d6b-1fb8c16b15d9",
    "d9293201-f36e-4865-ae4c-64557d1c4715",
    "3765905f-1293-45e0-82cd-26c2d0bd4083",
    "eb3ecd84-e0f4-4e94-9fdf-3c972abaf4c1",
    "f3f78593-151a-4702-a1e5-f6fa09860041",
    "2fcbfefc-06fa-4c56-824c-886297e9c845",
    "30b11a7d-51e7-498d-9ddd-bfab10e28734",
    "79e66277-6e43-4cad-bc34-b8ecb1d76041",
    "fd674195-cc62-4fbe-8dcb-4b1fec7722b7",
    "e9d43249-f2f8-426a-aece-a35c2ac91b34",
    "4a1a70a2-26af-4340-aa90-56ad49b0edce",
    "bac415b4-ae06-406e-8125-23108c8e7597",
    "91b8d450-1194-4f75-afc1-7dd0ce97ca10",
    "aa4adf51-4735-4c93-b17f-3aafaa8c2827",
    "73144685-dce7-4592-857f-dd17567308d3",
    "aaba1da4-a743-4731-9456-cbc83ff92fd8",
    "37286ce7-e3fa-4252-9af2-3f3892df6b17",
    "351a34e3-80d1-4e9c-853e-f1fc33dbe5ef",
    "bc59bd61-a985-409e-bc3a-d8b2bb3153bb",
    "dc7673d9-081b-4c38-aefe-7bdeab77d328",
    "d945c1ba-d740-42ff-bb7b-b3c69105dbab",
    "004b1220-4393-4267-992f-2ea59a449165",
    "e280cdb8-96e3-483b-9d68-e0185553a472",
    "99335d2c-6763-4c82-b7c8-4ef40ea89489",
    "bfde5145-8527-42f5-9f84-d77eaa7fe2eb",
    "c1d83f31-53bd-4db6-8539-36ce5797eef4",
    "06eb9728-d08a-4c3c-95f8-6bfaded2528e",
    "c61a1eb7-1f8e-4c3a-84f0-3e4c082f93e2",
    "3488534e-f599-40c7-b5b3-5fc486250c15",
    "7297d30b-eee1-4416-990e-18e557e208ac",
    "9daab139-fbca-4c1f-bd9d-fb7ef885add1",
    "259396bb-1b30-49b9-b729-a9a5470b975a",
    "f50c0b35-6383-41b6-aeac-45b9553048d8",
    "a5299f92-4e85-44cc-8cd6-ce67a5114f27",
    "873956ef-4ce4-4518-b5a8-33b4e6e1f90d",
    "3bc80325-3026-4a5a-839b-c7efa00cf6ba",
)

st.set_page_config(
    page_title="[admin]",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("[admin]")
st.caption("Aggregated assessment performance analytics for the configured assessment IDs.")

with st.form("admin_load"):
    st.markdown("#### Staff password")
    password = st.text_input(
        "Staff password",
        type="password",
        help="Same staff password as other secured tabs (`password` in Streamlit secrets).",
        key="admin_password_input",
    )
    submitted = st.form_submit_button("Load analytics for all assessments")

if submitted:
    if password != utils_assessment_analysis.settings.PASSWORD:
        st.error("Incorrect password.")
    else:
        prog = st.progress(0.0, text="Loading assessments…")
        try:
            df = utils_assessment_analysis.get_results_multi(
                ADMIN_ASSESSMENT_IDS,
                progress_bar=prog,
            )
        finally:
            prog.empty()
        if df.empty:
            st.warning("No attempt data returned for any configured assessment ID.")
            st.session_state.pop("admin_perf_df", None)
        else:
            st.session_state["admin_perf_df"] = df
            loaded_ids = sorted(df["assessmentId"].astype(str).unique())
            st.success(
                f"Loaded {len(df):,} rows across {len(loaded_ids)} assessment(s) with data."
            )

if "admin_perf_df" not in st.session_state:
    st.stop()

full_df = st.session_state["admin_perf_df"]
assessment_options = sorted(full_df["assessmentId"].astype(str).unique())

_ids_tuple = tuple(sorted(assessment_options))
sidebar_title_map = _cached_assessment_titles(_ids_tuple)
_base_labels = {
    aid: ((sidebar_title_map.get(aid) or "").strip() or "(Untitled)")
    for aid in assessment_options
}
_title_counts = Counter(_base_labels.values())
_assessment_display_labels = {
    aid: (
        f"{_base_labels[aid]} ({aid[:8]}…)"
        if _title_counts[_base_labels[aid]] > 1
        else _base_labels[aid]
    )
    for aid in assessment_options
}

st.sidebar.markdown("### Filters")
scope = st.sidebar.radio(
    "Assessments",
    ["All assessments", "Choose assessments"],
    horizontal=False,
)
if scope == "All assessments":
    selected_assessments = assessment_options
else:
    selected_assessments = st.sidebar.multiselect(
        "Assessments",
        options=assessment_options,
        default=assessment_options,
        format_func=lambda aid: _assessment_display_labels.get(str(aid), str(aid)),
    )

assessment_filtered = full_df[full_df["assessmentId"].astype(str).isin(selected_assessments)]
view_df = assessment_filtered

if not selected_assessments:
    st.warning("Select at least one assessment.")
    st.stop()

if view_df.empty:
    st.warning("No rows match the current filters.")
    st.stop()

single_assessment = len(selected_assessments) == 1

st.info(
    "**Assessment Performance Analysis** — question quality and total-score distributions "
    "(same as the Analyzing Assessments tab). **Section performance** charts appear only when "
    "exactly **one** assessment is selected. Use **Assessment-level summary** below to compare "
    "assessments using aggregated question metrics."
)

st.subheader("Assessment-level summary")
st.caption(
    "Question-level stats are aggregated per assessment (weighted by response count). Only "
    "questions with at least three attempts and valid discrimination are included—matching "
    "the question performance chart. **Attempts** counts distinct assessment attempts in the "
    "current filter sample."
)
summary_df = utils_assessment_analysis.assessment_level_summary_table(view_df)
if summary_df.empty:
    st.warning("No assessments in the current filter.")
else:
    summary_df = summary_df.copy()
    summary_df["assessment_title"] = summary_df["assessment_id"].astype(str).map(
        lambda x: (sidebar_title_map.get(x) or "").strip()
    )

    utils_assessment_analysis.plot_assessment_level_summary_scatter(summary_df)

    display_summary = summary_df.copy()
    display_summary["Assessment title"] = display_summary["assessment_title"].replace("", "—")
    display_summary["Avg success rate"] = display_summary["avg_success_rate"].map(
        lambda x: f"{x:.3f}" if pd.notna(x) else "—"
    )
    display_summary["Avg discrimination"] = display_summary["avg_discrimination"].map(
        lambda x: f"{x:.3f}" if pd.notna(x) else "—"
    )
    display_summary = display_summary.rename(
        columns={
            "assessment_id": "Assessment ID",
            "questions_in_summary": "Questions in summary",
            "n_attempts": "Attempts (distinct)",
        }
    )
    display_summary = display_summary[
        [
            "Assessment title",
            "Assessment ID",
            "Questions in summary",
            "Avg success rate",
            "Avg discrimination",
            "Attempts (distinct)",
        ]
    ]
    st.dataframe(display_summary, use_container_width=True, hide_index=True)

utils_assessment_analysis.plot_question_analysis(view_df)
utils_assessment_analysis.plot_total_score_histogram(view_df)

if single_assessment:
    utils_assessment_analysis.plot_section_scores(view_df)
else:
    st.subheader("Section performance")
    st.info(
        "Select **exactly one** assessment in the sidebar to load section-level charts and tables."
    )
