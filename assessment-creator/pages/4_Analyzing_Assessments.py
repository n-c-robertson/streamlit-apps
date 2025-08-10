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
# UI
#========================================

with st.form("Analyze Assessments"):
    assessment_id = st.text_input("Assessment ID", value='c84dd4d7-0fa0-47e7-9757-ac5b2ceb85d6')
    
    # Add filtering controls under an expander within the form
    with st.expander("Recommendation Filters", expanded=False):
        st.info("**Recommendation Filters**: Configure these settings to customize the content recommendations that will be generated from the Skills API.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Difficulty filter
            difficulty_options = ['Beginner', 'Intermediate', 'Advanced']
            selected_difficulties = st.multiselect(
                "Difficulty Levels",
                options=difficulty_options,
                default=difficulty_options,
                help="Select difficulty levels to include in recommendations"
            )
            
            # Program Type filter
            program_type_options = ['Part', 'Course', 'Degree']
            selected_program_types = st.multiselect(
                "Program Types",
                options=program_type_options,
                default=program_type_options,
                help="Select program types to include in recommendations"
            )
        
        with col2:
            # Duration filter
            duration_options = ['Minutes', 'Hours', 'Days', 'Weeks', 'Months']
            selected_durations = st.multiselect(
                "Duration",
                options=duration_options,
                default=duration_options,
                help="Select duration ranges to include in recommendations"
            )
            
            # Program Key filter (optional text input for specific keys)
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
        results_df = utils_assessment_analysis.get_results(assessment_id)
        user_skills_df = utils_assessment_analysis.user_skills(results_df)
        
        # Skills Analysis Expander
        with st.expander("Skills Analysis", expanded=False):
            st.info("**Skills Analysis**: These charts show how learners perform across different skills. The heatmap displays individual skill performance, while the correlation chart shows which skills learners tend to score similarly on.")
            utils_assessment_analysis.plot_net_skills_heatmap(user_skills_df, results_df)
            utils_assessment_analysis.plot_skill_correlation_heatmap(user_skills_df)
        
        # Assessment Performance Expander
        with st.expander("Assessment Performance Analysis", expanded=False):
            st.info("**Assessment Performance Analysis**: These charts analyze question quality, score distributions, and section performance. The question analysis shows which questions are most effective, while the histogram and section charts show overall performance patterns.")
            utils_assessment_analysis.plot_question_analysis(results_df)
            utils_assessment_analysis.plot_total_score_histogram(results_df)
            utils_assessment_analysis.plot_section_scores(results_df)

        # Recommendations Expander
        with st.expander("Learner Recommendations", expanded=False):
            st.subheader("Content Recommendations Based on Skill Gaps")
            st.info("**Content Recommendations**: Based on learners' skill gaps, these tables show personalized content recommendations. The program table summarizes top program recommendations, the lesson table shows individual lesson suggestions, and the learner table provides detailed recommendations for each user.")
            
            # Get recommendations from Skills API using the filters from the form
            with st.spinner("Fetching recommendations from Skills API..."):
                recommendations_df = utils_assessment_analysis.get_skills_recommendations(
                    user_skills_df, 
                    results_df,
                    difficulty_filter=selected_difficulties,
                    program_type_filter=selected_program_types,
                    duration_filter=selected_durations,
                    program_keys_filter=program_keys_input.strip() if program_keys_input.strip() else None
                )
            
            if not recommendations_df.empty:
                # Add the program recommendations table
                st.markdown("### Program Recommendations Summary Table")
                st.caption("**Program Summary**: Shows which programs are most commonly recommended as top choices, along with the skills they address for users who have them as their #1 recommendation.")
                program_table_df = utils_assessment_analysis.create_program_recommendations_table(recommendations_df)
                
                if program_table_df is not None:
                    # Display the table
                    st.dataframe(program_table_df, use_container_width=True)
                
                # Add the lesson recommendations table
                st.markdown("### Lesson Recommendations Summary Table")
                st.caption("**Lesson Summary**: Displays individual lesson recommendations for each user, showing their top 5 lessons and associated programs in a user-friendly format.")
                lesson_table_df = utils_assessment_analysis.create_lesson_recommendations_table(recommendations_df)
                
                if lesson_table_df is not None:
                    # Display the table
                    st.dataframe(lesson_table_df, use_container_width=True)
                
                # Add the learner recommendations table
                st.markdown("### Learner Recommendations Table")
                st.caption("**Learner Details**: Comprehensive view of all recommendations for each learner, including their weak skills, top lessons, and program recommendations.")
                learner_table_df = utils_assessment_analysis.create_learner_recommendations_table(recommendations_df)
                
                if learner_table_df is not None:
                    # Display the table
                    st.dataframe(learner_table_df, use_container_width=True)
                    

            else:
                st.warning("No recommendations available. This could be due to API issues or no weak skills identified.")
        
