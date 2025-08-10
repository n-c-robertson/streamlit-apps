#========================================
#IMPORT PACKAGES
#========================================

import streamlit as st
import ast
import concurrent.futures
import hashlib
import json
import os
import pickle
import random
import re
import requests
import time
import traceback
from collections import Counter
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import graphql_queries
import settings

#========================================
# FUNCTIONS
#========================================

def initialize_session_state():
    """Initialize session state variables for the review interface"""
    if 'questions_data' not in st.session_state:
        st.session_state.questions_data = None
    if 'current_question_index' not in st.session_state:
        st.session_state.current_question_index = 0
    if 'accepted_questions' not in st.session_state:
        st.session_state.accepted_questions = []
    if 'rejected_questions' not in st.session_state:
        st.session_state.rejected_questions = []
    if 'question_groups' not in st.session_state:
        st.session_state.question_groups = []

def process_csv_data(df):
    """Process CSV data and group by question_content"""
    # Group by question_content to get unique questions
    question_groups = []
    
    for question_content, group in df.groupby('question_content'):
        question_data = {
            'question_content': question_content,
            'records': group.to_dict('records'),
            'status': 'pending'  # pending, accepted, rejected
        }
        question_groups.append(question_data)
    
    return question_groups

def has_changes(original_question, edited_question, original_records, edited_choices):
    """Check if any changes were made to the question or choices"""
    if original_question != edited_question:
        return True
    
    for i, (original_record, edited_choice) in enumerate(zip(original_records, edited_choices)):
        if original_record['choice_content'] != edited_choice['choice_content']:
            return True
    
    return False

def display_question(question_data, question_index, total_questions):
    """Display a single question with its answer choices in a form"""
    st.markdown("---")
    
    # Header with progress (outside form)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### Question {question_index + 1} of {total_questions}")
    with col2:
        status = question_data.get('status', 'pending')
        if status == 'accepted':
            st.success("✅ Accepted")
        elif status == 'rejected':
            st.error("❌ Rejected")
        else:
            st.info("⏳ Pending")
    
    # Extract metadata from the first record (all records for a question have the same metadata)
    if question_data['records']:
        first_record = question_data['records'][0]
        
        # Create metadata display
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            # Skill display
            skill_id = first_record.get('skillId', 'Unknown Skill')
            st.markdown(f"**Skill:**")
            st.write(skill_id)
        
        with col2:
            # Difficulty display
            difficulty = first_record.get('difficultyLevelId', 'Unknown Difficulty')
            st.markdown(f"**Difficulty:**")
            st.write(difficulty)
        
        with col3:
            # Content URI with expandable details
            source = first_record.get('source', {})
            if isinstance(source, str):
                try:
                    source = ast.literal_eval(source)
                except:
                    source = {}
            
            uri = source.get('uri', 'No URI available')
            concept_title = source.get('conceptTitle', 'Unknown Concept')
            lesson_title = source.get('lessonTitle', 'Unknown Lesson')
            
            st.markdown(f"**Content Source:**")
            with st.expander(f"📚 {concept_title}", expanded=False):
                st.markdown(f"**Lesson:** {lesson_title}")
                st.markdown(f"**Concept:** {concept_title}")
                if uri != 'No URI available':
                    st.markdown(f"**URI:** [{uri}]({uri})")
                else:
                    st.markdown("**URI:** Not available")
    
    # Question form
    with st.form(f"question_form_{question_index}"):

        # Decision buttons in form
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            accept_submitted = st.form_submit_button("✅ Accept", type="secondary", use_container_width=True)
        with col2:
            reject_submitted = st.form_submit_button("❌ Reject", type="secondary", use_container_width=True)
        with col3:
            skip_submitted = st.form_submit_button("⏭️ Skip", use_container_width=True)

        # Editable question content
        edited_question = st.text_area(
            "Question",
            value=question_data['question_content'],
            key=f"question_edit_{question_index}",
            height=68,
            help="Edit the question text as needed"
        )
        
        # Editable answer choices
        records = question_data['records']
        edited_choices = []
        
        for i, record in enumerate(records):
            # Create label with correct answer indicator
            choice_label = f"Choice {chr(65 + i)}:"
            if record['choice_isCorrect']:
                choice_label += " ✅ (Correct Answer)"
            
            # Editable choice content
            edited_choice = st.text_area(
                choice_label,
                value=record['choice_content'],
                key=f"choice_edit_{question_index}_{i}",
                height=68,
                help=f"Edit choice {chr(65 + i)} content" + (" (This is the correct answer)" if record['choice_isCorrect'] else "")
            )
            edited_choices.append({
                'choice_content': edited_choice,
                'choice_isCorrect': record['choice_isCorrect'],
                'original_record': record
            })
        
        # Show changes indicator
        if has_changes(question_data['question_content'], edited_question, records, edited_choices):
            st.info("📝 **Changes detected** - Your edits will be saved when you accept this question.")
        
        
        # Return the form submission result with edited data
        if accept_submitted:
            return {
                'action': 'accept',
                'edited_question': edited_question,
                'edited_choices': edited_choices
            }
        elif reject_submitted:
            return {
                'action': 'reject',
                'edited_question': edited_question,
                'edited_choices': edited_choices
            }
        elif skip_submitted:
            return {
                'action': 'skip',
                'edited_question': edited_question,
                'edited_choices': edited_choices
            }
        else:
            return None

def create_download_data():
    """Create data for download from accepted questions"""
    if not st.session_state.accepted_questions:
        return None
    
    # Combine all accepted question records
    download_records = []
    for question_data in st.session_state.accepted_questions:
        download_records.extend(question_data['records'])
    
    return pd.DataFrame(download_records)

#========================================
# MAIN INTERFACE
#========================================

def main():
    st.title("Reviewing Assessments")
    st.markdown("Upload a CSV file and review questions one by one. Mark questions as accepted or rejected, then download your approved questions.")
    
    # Initialize session state
    initialize_session_state()
    
    # File upload section
    st.markdown("### Upload Assessment Data")
    
    with st.form("upload_form"):
        csv = st.file_uploader("Upload a CSV generated from the 'Generating Assessments' tab", type="csv", key="csv_uploader")
        review_submitted = st.form_submit_button("Review Questions", type="secondary", use_container_width=True)
    
    # Process the form submission
    if review_submitted and csv is not None:
        # Load and process data
        with st.spinner("Processing CSV data..."):
            df = pd.read_csv(csv)
            st.session_state.questions_data = df
            st.session_state.question_groups = process_csv_data(df)
            st.session_state.current_question_index = 0
            st.session_state.accepted_questions = []
            st.session_state.rejected_questions = []
    
    # Show review interface if we have data loaded
    if st.session_state.questions_data is not None and st.session_state.question_groups:
        # Display summary statistics
        total_questions = len(st.session_state.question_groups)
        accepted_count = len(st.session_state.accepted_questions)
        rejected_count = len(st.session_state.rejected_questions)
        pending_count = total_questions - accepted_count - rejected_count
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Questions", total_questions)
        with col2:
            st.metric("Accepted", accepted_count)
        with col3:
            st.metric("Rejected", rejected_count)
        with col4:
            st.metric("Pending", pending_count)
        
        # Progress bar
        progress = (accepted_count + rejected_count) / total_questions if total_questions > 0 else 0
        st.progress(progress)
        
        # Question review interface
        if total_questions > 0:
            current_question = st.session_state.question_groups[st.session_state.current_question_index]
            result = display_question(current_question, st.session_state.current_question_index, total_questions)
            
            # Handle form submission results
            if result is not None:
                if result['action'] == 'accept':
                    # Update the question with edited content
                    current_question['question_content'] = result['edited_question']
                    
                    # Update the records with edited choices and question content
                    for i, edited_choice in enumerate(result['edited_choices']):
                        if i < len(current_question['records']):
                            current_question['records'][i]['choice_content'] = edited_choice['choice_content']
                            # Also update the question_content in each record
                            current_question['records'][i]['question_content'] = result['edited_question']
                    
                    current_question['status'] = 'accepted'
                    if current_question not in st.session_state.accepted_questions:
                        st.session_state.accepted_questions.append(current_question)
                    if current_question in st.session_state.rejected_questions:
                        st.session_state.rejected_questions.remove(current_question)
                    
                    # Auto-advance to next question if not at the end
                    if st.session_state.current_question_index < total_questions - 1:
                        st.session_state.current_question_index += 1
                    st.rerun()
                elif result['action'] == 'reject':
                    current_question['status'] = 'rejected'
                    if current_question not in st.session_state.rejected_questions:
                        st.session_state.rejected_questions.append(current_question)
                    if current_question in st.session_state.accepted_questions:
                        st.session_state.accepted_questions.remove(current_question)
                    
                    # Auto-advance to next question if not at the end
                    if st.session_state.current_question_index < total_questions - 1:
                        st.session_state.current_question_index += 1
                    st.rerun()
                elif result['action'] == 'skip':
                    # Auto-advance to next question if not at the end
                    if st.session_state.current_question_index < total_questions - 1:
                        st.session_state.current_question_index += 1
                    st.rerun()
            
            # Navigation and action buttons
            col1, col2, col3 = st.columns([2, 2, 8])
            
            with col1:
                if st.button("⬅️ Previous", disabled=st.session_state.current_question_index == 0):
                    st.session_state.current_question_index = max(0, st.session_state.current_question_index - 1)
                    st.rerun()
            
            with col2:
                if st.button("➡️ Next", disabled=st.session_state.current_question_index == total_questions - 1):
                    st.session_state.current_question_index = min(total_questions - 1, st.session_state.current_question_index + 1)
                    st.rerun()
            
            with col3:
                pass

            # Create a selectbox for quick navigation
            question_options = [f"Question {i+1}: {q['question_content'][:50]}..." 
                              for i, q in enumerate(st.session_state.question_groups)]
            
            selected_question = st.selectbox(
                "Jump to a specific question:",
                options=question_options,
                index=st.session_state.current_question_index
            )
            
            if selected_question:
                new_index = question_options.index(selected_question)
                if new_index != st.session_state.current_question_index:
                    st.session_state.current_question_index = new_index
                    st.rerun()
            
            # Download section
            st.markdown("### Download Results")
            
            # Recalculate counts to ensure they're current
            current_accepted_count = len(st.session_state.accepted_questions)
            
            if current_accepted_count > 0:
                download_data = create_download_data()
                if download_data is not None:
                    csv_data = download_data.to_csv(index=False)
                    st.download_button(
                        label=f"Download {current_accepted_count} Accepted Questions",
                        data=csv_data,
                        file_name="accepted_questions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Show preview of accepted questions
                    with st.expander("👀 Preview Accepted Questions"):
                        st.dataframe(download_data, use_container_width=True)
            else:
                st.info("No questions have been accepted yet. Accept some questions to enable download.")
    
    elif review_submitted and csv is None:
        st.error("Please upload a CSV file before clicking 'Review Questions'.")
    
    else:
        pass
if __name__ == "__main__":
    main()
