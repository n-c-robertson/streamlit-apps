#========================================
#IMPORT PACKAGES
#========================================

import streamlit as st
import utils_assessment_generation

#========================================
#UI
#========================================

def main():
    st.title("Generating Assessments")
    st.markdown("Create AI-generated assessment questions for one or more Udacity programs.")
    
    # Initialize session state for storing results
    if 'generated_questions_df' not in st.session_state:
        st.session_state.generated_questions_df = None
    
    if 'progress_data' not in st.session_state:
        st.session_state.progress_data = {}
    
    # Processing time information
    st.info("**Processing Time**: Expect this to take 3-5~ minutes per program.")

    with st.form('Generate Assessments'):
        st.markdown('#### Required Parameters')
        PROGRAM_KEYS = st.text_input(
            'Program Keys (comma separated)', 
            value='cd13303,cd13318,cd13267,cd1827', 
            placeholder='Enter program keys (comma separated)',
            help="Enter the program keys for the content you want to generate questions for (cd101, cd102, etc.)"
        )
        
        with st.expander("Advanced Settings"):
            ASSESSMENT_TYPE = st.selectbox(
                'Assessment Type',
                ['Placement', 'Readiness'],
                help="Placement: Test skills taught by the content. Readiness: Test prerequisite skills needed to understand the content."
            )

            QUESTION_TYPES = st.multiselect(
            'Question Types', 
            ['MULTIPLE_CHOICE', 'SINGLE_CHOICE'], 
            default=['MULTIPLE_CHOICE', 'SINGLE_CHOICE'],
            help="Select the types of questions you want to generate. Only SINGLE CHOICE and MULTIPLE CHOICE problems are supported."
        )

            # Add question limit slider
            QUESTION_LIMIT = st.select_slider(
                'Question Limit', 
                options=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 'No Limit'], 
                value='No Limit',
                help="Set a maximum number of questions in the final assessment. If the limit is higher than available questions, an AI agent will select the best questions for each skill."
            )

            CUSTOMIZED_DIFFICULTY = st.select_slider('Custom Difficulty', options=['Much Easier', 'Easier', 'A Little Easier', 'No Change', 'A Little Harder', 'Harder', 'Much Harder'], value='No Change', help="Adjust the base difficulty of the questions.")

            TEMPERATURE = st.slider(
            'Temperature', 
            value=0.2, 
            min_value=0.0, 
            max_value=1.0, 
            step=0.1,
            help="Controls creativity vs consistency. Lower values (0.1-0.3) produce more consistent questions. Higher values (0.7-0.9) produce more creative questions, with some risk of hallucinations."
        )

            CUSTOMIZED_PROMPT_INSTRUCTIONS = st.text_area('Custom Instructions', value='', help="Enter additional instructions for the prompt. This will be appended to the base prompt.")


        st.markdown('#### Staff Password')
        password = st.text_input("Staff Password", type="password", help="Enter the required staff password")
        
        submitted = st.form_submit_button("Generate Assessments", use_container_width=True)
        
        if submitted:
            if password != utils_assessment_generation.settings.PASSWORD:
                st.error("❌ Incorrect password. Please try again.")
            elif not PROGRAM_KEYS.strip():
                st.error("❌ Please enter at least one program key.")
            elif not QUESTION_TYPES:
                st.error("❌ Please select at least one question type.")
            else:
                # Create progress bar and text elements
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                try:
                    with st.spinner("Generating assessments..."):
                        questions_choices_df, progress_data = utils_assessment_generation.generate_assessments(
                            PROGRAM_KEYS, 
                            QUESTION_TYPES, 
                            QUESTION_LIMIT, 
                            CUSTOMIZED_DIFFICULTY, 
                            CUSTOMIZED_PROMPT_INSTRUCTIONS, 
                            TEMPERATURE, 
                            ASSESSMENT_TYPE, # Pass the selected assessment type
                            progress_bar, 
                            progress_text
                        )
                    
                    # Check for missing prerequisite skills warning
                    if ASSESSMENT_TYPE == "Readiness" and 'missing_prerequisite_skills' in progress_data:
                        missing_skills = progress_data['missing_prerequisite_skills']
                        if missing_skills:
                            st.warning("⚠️ **No prerequisite skills found on some programs. Falling back to skills tagged on the program, this may affect the results.**")
                            with st.expander("📋 Programs with missing prerequisite skills"):
                                for item in missing_skills:
                                    st.write(f"**{item['title']}** (Key: {item['key']})")
                                    st.write(f"Teaches skills: {', '.join(item['teaches_skills'])}")
                    
                    # Store results in session state
                    st.session_state.generated_questions_df = questions_choices_df
                    st.session_state.progress_data = progress_data
                    st.rerun()
                        
                except ValueError as e:
                    # Handle specific ValueError for missing prerequisite skills
                    error_message = str(e)
                    if "prerequisite skills" in error_message.lower():
                        st.error(f"❌ **Readiness Assessment Error**: {error_message}")
                        st.info("💡 **Solution**: Go to Studio and add prerequisite skills to the program to generate a readiness assessment.")
                    else:
                        st.error(f"❌ An error occurred during generation: {error_message}")
                        st.error(utils_assessment_generation.format_exception_details(e))
                except Exception as e:
                    st.error(f"❌ An error occurred during generation: {str(e)}")
                    st.error(utils_assessment_generation.format_exception_details(e))
                    st.error("Please check your program keys and try again.")

    # Display results outside the form
    if st.session_state.generated_questions_df is not None and not st.session_state.generated_questions_df.empty:
        questions_choices_df = st.session_state.generated_questions_df
        progress_data = st.session_state.get('progress_data', {})
        
        st.success(f"✅ Successfully generated {len(questions_choices_df)} assessment items!")
        
        # Show summary statistics
        st.markdown("### 📊 Generation Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            unique_questions = questions_choices_df['question_content'].nunique()
            st.metric("Unique Questions", unique_questions)
        with col2:
            total_choices = len(questions_choices_df)
            st.metric("Total Answer Choices", total_choices)
        with col3:
            correct_choices = questions_choices_df['choice_isCorrect'].sum()
            st.metric("Correct Answers", correct_choices)
        with col4:
            avg_choices_per_question = total_choices / unique_questions if unique_questions > 0 else 0
            st.metric("Avg Choices/Question", f"{avg_choices_per_question:.1f}")
        
        # Display question count progression
        if progress_data and 'question_counts' in progress_data:
            
            question_counts = progress_data['question_counts']
            if len(question_counts) >= 2:
                initial_questions = question_counts[0][1]
                final_questions = question_counts[-1][1]
                total_reduction = initial_questions - final_questions
                reduction_percentage = (total_reduction / initial_questions * 100) if initial_questions > 0 else 0
                generation_percentage = (final_questions / initial_questions * 100) if initial_questions > 0 else 0
                
                # Create a simple horizontal progress bar
                st.write(f"**Questions Pruned from Original Generation: {reduction_percentage:.0f}%**")
                
                # Create the progress bar using HTML/CSS for a native look
                progress_html = f"""
                <div style="margin: 15px 0;">
                    <div style="
                        background-color: #262730; 
                        height: 28px; 
                        border-radius: 14px; 
                        overflow: hidden; 
                        position: relative;
                        border: 1px solid #404040;
                        box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);
                    ">
                        <div style="
                            background: linear-gradient(90deg, #ff4b4b 0%, #ff6b6b 100%);
                            height: 100%; 
                            width: {generation_percentage}%; 
                            transition: width 0.8s ease-out;
                            border-radius: 14px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
                        "></div>
                        <div style="
                            position: absolute;
                            top: 50%;
                            left: 50%;
                            transform: translate(-50%, -50%);
                            color: white;
                            font-weight: 600;
                            font-size: 13px;
                            white-space: nowrap;
                        ">
                            {initial_questions:,} → {final_questions:,} questions
                        </div>
                    </div>
                </div>
                """
                st.markdown(progress_html, unsafe_allow_html=True)
                
                # Add a subtle breakdown
                with st.expander("Debug: Detailed breakdown of pruning", expanded=False):                    
                    # Get the original number of questions for consistent percentage calculations
                    original_questions = question_counts[0][1] if len(question_counts) > 0 else 0
                    
                    st.markdown("#### 🔽 **Funnel Steps - Questions Pruned**")
                    
                    # Add evaluation filtering breakdown
                    if 'evaluation_stats' in progress_data:
                        evaluation_stats = progress_data['evaluation_stats']
                        questions_failed = evaluation_stats['failed_questions']
                        remaining_questions = evaluation_stats['after_evaluation_unique_questions']
                        
                        st.write(f"- **Evaluation filtering**: {questions_failed:,} questions removed → {remaining_questions:,} remaining")
                    
                    # Add detailed deduplication breakdown
                    if 'intermediate_counts' in progress_data:
                        intermediate_counts = progress_data['intermediate_counts']
                        
                        # TF-IDF deduplication
                        tfidf_removed = intermediate_counts['initial_unique_questions'] - intermediate_counts['after_tfidf_unique_questions']
                        remaining_after_tfidf = intermediate_counts['after_tfidf_unique_questions']
                        st.write(f"- **TF-IDF deduplication**: {tfidf_removed:,} questions removed → {remaining_after_tfidf:,} remaining")
                        
                        # Semantic deduplication
                        semantic_removed = intermediate_counts['after_tfidf_unique_questions'] - intermediate_counts['after_semantic_unique_questions']
                        remaining_after_semantic = intermediate_counts['after_semantic_unique_questions']
                        st.write(f"- **Semantic deduplication**: {semantic_removed:,} questions removed → {remaining_after_semantic:,} remaining")
                    
                    # Add content specificity filtering breakdown
                    if 'filtering_stats' in progress_data:
                        filtering_stats = progress_data['filtering_stats']
                        questions_filtered = filtering_stats['filtered_out_questions']
                        remaining_after_filtering = filtering_stats['after_filtering_unique_questions']
                        
                        st.write(f"- **Content specificity filtering**: {questions_filtered:,} questions removed → {remaining_after_filtering:,} remaining")
                    
                    # Add intelligent question selection breakdown
                    if 'selection_stats' in progress_data:
                        selection_stats = progress_data['selection_stats']
                        if selection_stats['selection_needed']:
                            questions_before_selection = selection_stats['total_questions']
                            questions_selected = selection_stats['selected_questions']
                            questions_removed = questions_before_selection - questions_selected
                            
                            st.write(f"- **Question limit selection**: {questions_removed:,} questions removed → {questions_selected:,} remaining")
                        else:
                            st.write(f"- **Question limit selection**: No selection needed ({selection_stats['reason']})")
                    
                    st.markdown("#### 🔄 **Transform Steps - Questions Touched**")
                    
                    # Add case study conversion breakdown
                    if 'conversion_stats' in progress_data:
                        conversion_stats = progress_data['conversion_stats']
                        questions_converted = conversion_stats['successfully_converted']
                        questions_selected = conversion_stats['questions_selected_for_conversion']
                        
                        st.write(f"- **Case study conversion**: {questions_converted:,} questions converted (from {questions_selected:,} selected)")
                    
                    # Add code format conversion breakdown
                    if 'code_conversion_stats' in progress_data:
                        code_conversion_stats = progress_data['code_conversion_stats']
                        questions_converted = code_conversion_stats['converted_questions']
                        coding_questions_available = code_conversion_stats['total_coding_questions']
                        
                        st.write(f"- **Code format conversion**: {questions_converted:,} questions converted (from {coding_questions_available:,} with coding content)")
                    
                    # Add distractor tuning breakdown
                    if 'tuning_stats' in progress_data:
                        tuning_stats = progress_data['tuning_stats']
                        questions_tuned = tuning_stats['tuned_questions']
                        questions_selected = tuning_stats['questions_selected_for_tuning']
                        
                        st.write(f"- **Distractor tuning**: {questions_tuned:,} questions improved (from {questions_selected:,} selected)")

        # Display results
        st.markdown("### 📋 Generated Questions")
        st.dataframe(questions_choices_df, use_container_width=True)
        
        # Download section
        st.markdown("### 💾 Download Results")
        csv_data = questions_choices_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download {len(questions_choices_df)} Assessment Items",
            data=csv_data,
            file_name="generated_assessments.csv",
            mime="text/csv",
            use_container_width=True
        )

        
        # Next steps
        st.markdown("### Next Steps")
        st.markdown("""
        1. **Review the generated questions** in the table above
        2. **Download the CSV file** to save your results
        3. **Go to the 'Reviewing Assessments' tab** to review and edit questions one by one
        4. **Accept or reject questions** based on your quality standards
        5. **Download your final approved questions** from the review interface
        """)
        
    elif st.session_state.generated_questions_df is not None and st.session_state.generated_questions_df.empty:
        st.warning("No questions were generated. Please check your program keys and try again.")
        # Clear the empty result from session state
        st.session_state.generated_questions_df = None

if __name__ == "__main__":
    main()