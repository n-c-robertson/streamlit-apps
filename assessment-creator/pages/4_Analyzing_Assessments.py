#========================================
#IMPORT PACKAGES
#========================================

import streamlit as st
import utils_assessment_analysis

#========================================
# PAGE CONFIGURATION
#========================================

st.set_page_config(
    page_title="Assessment Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

#========================================
# SESSION STATE INIT
#========================================

for _key in ["results_df", "user_skills_df", "assessment_id_loaded", "reco_filter_state", "question_details_df"]:
    if _key not in st.session_state:
        st.session_state[_key] = None

#========================================
# UI
#========================================

with st.form("Analyze Assessments"):
    assessment_id = st.text_input("Assessment ID", value='c84dd4d7-0fa0-47e7-9757-ac5b2ceb85d6')
    
    with st.expander("Recommendation Filters", expanded=False):
        st.info("**Recommendation Filters**: Configure these settings to customize the content recommendations that will be generated from the Skills API.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            difficulty_options = ['Beginner', 'Intermediate', 'Advanced']
            selected_difficulties = st.multiselect(
                "Difficulty Levels",
                options=difficulty_options,
                default=difficulty_options,
                help="Select difficulty levels to include in recommendations"
            )
            
            program_type_options = ['Part', 'Course', 'Degree']
            selected_program_types = st.multiselect(
                "Program Types",
                options=program_type_options,
                default=program_type_options,
                help="Select program types to include in recommendations"
            )
        
        with col2:
            duration_options = ['Minutes', 'Hours', 'Days', 'Weeks', 'Months']
            selected_durations = st.multiselect(
                "Duration",
                options=duration_options,
                default=duration_options,
                help="Select duration ranges to include in recommendations"
            )
            
            program_keys_input = st.text_input(
                "Specific Program Keys (optional)",
                placeholder="e.g., ud1110, cd2841",
                help="Enter specific program keys separated by commas (leave empty to include all)"
            )
    
    st.markdown('#### Staff Password')
    password = st.text_input("Staff Password", type="password", help="Enter the required staff password")
    submitted = st.form_submit_button("Submit")

if submitted:
    if password != utils_assessment_analysis.settings.PASSWORD:
        st.error("❌ Incorrect password. Please try again.")
    else:
        fetch_progress = st.progress(0)
        fetch_status = st.empty()

        results_df, question_details_df = utils_assessment_analysis.get_results(
            assessment_id,
            progress_bar=fetch_progress,
            status_text=fetch_status,
        )

        user_skills_df = utils_assessment_analysis.user_skills(results_df)

        st.session_state.results_df = results_df
        st.session_state.user_skills_df = user_skills_df
        st.session_state.question_details_df = question_details_df
        st.session_state.assessment_id_loaded = assessment_id
        st.session_state.reco_filter_state = {
            "difficulties": selected_difficulties,
            "program_types": selected_program_types,
            "durations": selected_durations,
            "program_keys": program_keys_input.strip() if program_keys_input.strip() else None,
        }

        fetch_progress.empty()
        fetch_status.empty()

# Render analysis sections whenever results are available in session state
if st.session_state.results_df is not None and not st.session_state.results_df.empty:
    results_df = st.session_state.results_df
    user_skills_df = st.session_state.user_skills_df
    question_details_df = st.session_state.question_details_df

    # Skills Analysis Expander
    with st.expander("Skills Analysis", expanded=False):
        st.info("**Skills Analysis**: These charts show how learners perform across different skills. The heatmap displays individual skill performance, while the correlation chart shows which skills learners tend to score similarly on.")
        utils_assessment_analysis.plot_net_skills_heatmap(user_skills_df, results_df)
        utils_assessment_analysis.plot_skill_correlation_heatmap(user_skills_df)
    
    # Assessment Performance Expander
    with st.expander("Assessment Performance Analysis", expanded=False):
        st.info("**Assessment Performance Analysis**: These charts analyze question quality, score distributions, and section performance. The question analysis shows which questions are most effective, while the histogram and section charts show overall performance patterns.")
        utils_assessment_analysis.plot_question_analysis(results_df, question_details=question_details_df)
        utils_assessment_analysis.plot_total_score_histogram(results_df)
        utils_assessment_analysis.plot_section_scores(results_df)

    # Recommendations Expander — manual trigger only
    with st.expander("Learner Recommendations", expanded=False):
        st.subheader("Content Recommendations Based on Skill Gaps")
        st.info("**Content Recommendations**: Based on learners' skill gaps, these tables show personalized content recommendations. The program table summarizes top program recommendations, the lesson table shows individual lesson suggestions, and the learner table provides detailed recommendations for each user.")

        if st.button("Load Recommendations", type="primary"):
            filters = st.session_state.reco_filter_state or {}
            with st.spinner("Fetching recommendations from Skills API..."):
                recommendations_df = utils_assessment_analysis.get_skills_recommendations(
                    user_skills_df,
                    results_df,
                    difficulty_filter=filters.get("difficulties"),
                    program_type_filter=filters.get("program_types"),
                    duration_filter=filters.get("durations"),
                    program_keys_filter=filters.get("program_keys"),
                )

            if not recommendations_df.empty:
                st.markdown("### Program Recommendations Summary Table")
                st.caption("**Program Summary**: Shows which programs are most commonly recommended as top choices, along with the skills they address for users who have them as their #1 recommendation.")
                program_table_df = utils_assessment_analysis.create_program_recommendations_table(recommendations_df)
                if program_table_df is not None:
                    st.dataframe(program_table_df, use_container_width=True)
                else:
                    st.info("Dataframe is empty.")

                st.markdown("### Lesson Recommendations Summary Table")
                st.caption("**Lesson Summary**: Displays individual lesson recommendations for each user, showing their top 5 lessons and associated programs in a user-friendly format.")
                lesson_table_df = utils_assessment_analysis.create_lesson_recommendations_table(recommendations_df)
                if lesson_table_df is not None:
                    st.dataframe(lesson_table_df, use_container_width=True)
                else:
                    st.info("Dataframe is empty.")

                st.markdown("### Learner Recommendations Table")
                st.caption("**Learner Details**: Comprehensive view of all recommendations for each learner, including their weak skills, top lessons, and program recommendations.")
                learner_table_df = utils_assessment_analysis.create_learner_recommendations_table(recommendations_df)
                if learner_table_df is not None:
                    st.dataframe(learner_table_df, use_container_width=True)
                else:
                    st.info("Dataframe is empty.")
            else:
                st.warning("No recommendations available. This could be due to API issues or no weak skills identified.")
