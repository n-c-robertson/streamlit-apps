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

def generate_slug(ASSESSMENT_TITLE):
    # Remove text inside square brackets (including the brackets)
    title_no_brackets = re.sub(r'\[.*?\]|/|,', '', ASSESSMENT_TITLE)
    # Strip whitespace, convert to lowercase, and replace spaces (or any whitespace) with hyphens.
    slug = re.sub(r'\s+', '-', title_no_brackets.strip().lower())
    return slug

def fetch_difficulty_levels():
    query = """
    query {
      difficultyLevels {
        id
        externalId
        label
        labelValue
        category
        status
      }
    }
    """
    r = requests.post(settings.ASSESSMENTS_API_URL, headers=settings.production_headers(), json={"query": query})
    response_json = r.json()
    return response_json['data']['difficultyLevels']
    
def fetch_skills():
    query = """
    query {
      skills {
        id
        externalId
        title
        category
        status
      }
    }
    """
    
    r = requests.post(settings.ASSESSMENTS_API_URL, headers=settings.production_headers(), json={"query": query})    
    response_json = r.json()
    return response_json['data']['skills']


def add_id_fields(df):
    """Add difficulty level and skill IDs to the DataFrame"""
    # Fetch data from API
    difficulty_levels = fetch_difficulty_levels()
    skills = fetch_skills()

    print('DIFFICULTY LEVELS: ', difficulty_levels)
    print('SKILLS: ', skills)
    
    # Create mappings using externalId
    difficulty_mapping = {dl['externalId']: dl['id'] for dl in difficulty_levels if dl.get('externalId')}
    skill_mapping = {skill['externalId']: skill['id'] for skill in skills if skill.get('externalId')}
    
    print('DIFFICULTY MAPPING: ', difficulty_mapping)
    print('SKILL MAPPING: ', skill_mapping)

    # Add ID columns using cleaned values
    df['difficultyLevelId'] = df['difficultyLevelUri'].map(difficulty_mapping).fillna('')
    df['skillId'] = df['skillUri'].map(skill_mapping).fillna('')
    
    return df

def create_assessment(assessment_title, description=""):
    """Create an assessment using the GraphQL mutation"""
    create_assessment_mutation = """
    mutation createAssessment($input: CreateAssessmentInput!) {
      createAssessment(input: $input) {
        id
        title
        description
        category
        status
        passingScore
        attemptRoles
      }
    }
    """
    
    assessment_variables = {
        "input": {
            "title": assessment_title,
            "description": description,
            "category": "PASS_FAIL",
            "passingScore": 0.8,
            "status": "OPEN",
            "opensAt": None,
            "closesAt": None,
            "attemptRoles": ["SERVICE","STAFF","USER"],
            "isTimed": False,
            "timeLimitMinutes": None,
            "maxTotalAttempts": None,
            "maxAttemptsWithinWindow": None,
            "maxAttemptsWindowHours": None,
            "startMessage": "Good luck!",
            "endMessage": "Thank you for completing the assessment.",
            "passMessage": "Congratulations, you passed!",
            "failMessage": "Unfortunately, you did not pass.",
            "termsAgreement": ""
        }
    }
    
    assessment_payload = {
        "query": create_assessment_mutation,
        "variables": assessment_variables
    }
    
    try:
        response = requests.post(
            settings.ASSESSMENTS_API_URL,
            headers=settings.production_headers(),
            json=assessment_payload
        )
        if response.status_code != 200:
            raise Exception(f"GraphQL mutation failed with status code {response.status_code}: {response.text}")
        
        response_json = response.json()
        assessment_id = response_json['data']['createAssessment']['id']
        return assessment_id, response_json
        
    except Exception as e:
        st.error(f"Error creating assessment: {e}")
        return None, None

def create_section(assessment_id, section_title, num_questions_to_ask=10):
    """Create a section for an assessment"""
    create_section_mutation = """
    mutation createSection($input: CreateSectionInput!) {
      createSection(input: $input) {
        id
        title
        description
        status
        isAdaptive
        isRandom
        numberOfQuestionsToAsk
        isTimed
        timeLimitMinutes
        orderIndex
        assessmentId
      }
    }
    """
    
    section_variables = {
        "input": {
            "assessmentId": assessment_id,
            "title": section_title,
            "description": "",
            "status": "ACTIVE",
            "isAdaptive": False,
            "isRandom": True,
            "numberOfQuestionsToAsk": num_questions_to_ask,
            "isTimed": False,
            "timeLimitMinutes": None,
            "orderIndex": 0,
        }
    }
    
    section_payload = {
        "query": create_section_mutation,
        "variables": section_variables
    }
    
    try:
        response = requests.post(
            settings.ASSESSMENTS_API_URL,
            headers=settings.production_headers(),
            json=section_payload
        )
        if response.status_code != 200:
            raise Exception(f"GraphQL mutation failed with status code {response.status_code}: {response.text}")
        
        response_json = response.json()
        section_id = response_json['data']['createSection']['id']
        return section_id, response_json
        
    except Exception as e:
        st.error(f"Error creating section {section_title}: {e}")
        return None, None

def create_question(section_id, question_data):
    """Create a question with its choices"""
    create_question_mutation = """
    mutation createQuestion($input: CreateQuestionInput!) {
      createQuestion(input: $input) {
        id
        content
        category
        status
      }
    }
    """
    
    # Handle source data properly - convert to CreateQuestionSourceInput format
    source_data = question_data.get('source', {})
    if isinstance(source_data, str):
        try:
            source_data = ast.literal_eval(source_data)
        except:
            source_data = {}
    
    # Format source as CreateQuestionSourceInput - only include the most basic valid fields
    formatted_source = {
        "uri": source_data.get('uri', ''),
        "key": source_data.get('partKey', '')
    }
    
    # Remove any None or empty values to avoid validation issues
    formatted_source = {k: v for k, v in formatted_source.items() if v is not None and v != ''}
    
    question_variables = {
        "input": {
            'sectionId': section_id,
            'difficultyLevelId': question_data['difficultyLevelId'],
            'skillId': question_data['skillId'],
            'category': question_data['category'],
            'status': question_data['question_status'],
            'content': question_data['question_content'],
            'source': formatted_source,
            'sourceCategory': 'UDACITY'
        }
    }
    
    # Add logging to debug the issue
    st.write(f"Creating question with data: {question_variables}")
    
    question_payload = {
        "query": create_question_mutation,
        "variables": question_variables
    }
    
    try:
        response = requests.post(
            settings.ASSESSMENTS_API_URL,
            headers=settings.production_headers(),
            json=question_payload
        )
        if response.status_code != 200:
            st.error(f"Question mutation failed with status code {response.status_code}: {response.text}")
            st.write(f"Request payload: {question_payload}")
            raise Exception(f"Question mutation failed with status code {response.status_code}: {response.text}")
        
        response_json = response.json()
        question_id = response_json['data']['createQuestion']['id']
        return question_id, response_json
        
    except Exception as e:
        st.error(f"Error creating question: {e}")
        return None, None

def create_choice(question_id, choice_data):
    """Create a choice for a question"""
    create_choice_mutation = """
    mutation createChoice($input: CreateChoiceInput!) {
      createChoice(input: $input) {
        id
        content
        isCorrect
      }
    }
    """
    
    choice_variables = {
        "input": {
            'questionId': question_id,
            'content': choice_data['choice_content'],
            'isCorrect': choice_data['choice_isCorrect'],
            'orderIndex': choice_data['choice_orderIndex'],
            'status': choice_data['choice_status']
        }
    }
    
    choice_payload = {
        "query": create_choice_mutation,
        "variables": choice_variables
    }
    
    try:
        response = requests.post(
            settings.ASSESSMENTS_API_URL,
            headers=settings.production_headers(),
            json=choice_payload
        )
        if response.status_code != 200:
            raise Exception(f"Choice mutation failed with status code {response.status_code}: {response.text}")
        
        response_json = response.json()
        return response_json
        
    except Exception as e:
        st.error(f"Error creating choice: {e}")
        return None

def process_choice(question_id, row):
    """
    Process one choice by sending the createChoice mutation.
    Returns a tuple: (choice_payload, response_json, exception)
    """
    create_choice_mutation = """
    mutation createChoice($input: CreateChoiceInput!) {
      createChoice(input: $input) {
        id
        content
        isCorrect
      }
    }
    """
    choice_variables = {
        "input": {
            'questionId': question_id,
            'content': row['choice_content'],
            'isCorrect': row['choice_isCorrect'],
            'orderIndex': row['choice_orderIndex'],
            'status': row['choice_status']
        }
    }
    choice_payload = {
        "query": create_choice_mutation,
        "variables": choice_variables
    }
    try:
        r = requests.post(settings.ASSESSMENTS_API_URL, headers=settings.production_headers(), json=choice_payload)
        if r.status_code != 200:
            raise Exception(f"Choice mutation failed with status code {r.status_code}: {r.text}")
        response_json = r.json()
        return (choice_payload, response_json, None)
    except Exception as e:
        return (choice_payload, None, e)

def process_question_group(question_tuple, group):
    """
    For a given grouped question (question_tuple with associated rows in group),
    send the createQuestion mutation first, and then concurrently process each choice.
    Returns a tuple: (question_payload, question_id, question_exception, list_of_choice_results)
    Each element in list_of_choice_results is (choice_payload, response_json, exception)
    """
    create_question_mutation = """
    mutation createQuestion($input: CreateQuestionInput!) {
      createQuestion(input: $input) {
        id
        content
        category
        status
      }
    }
    """
    # question_tuple is a tuple of the columns in question_columns:
    # [sectionId, difficultyLevelId, skillId, category, question_status, question_content, source]
    question_variables = {
        "input": {
            'sectionId': question_tuple[0],
            'difficultyLevelId': question_tuple[1],
            'skillId': question_tuple[2],
            'category': question_tuple[3],
            'status': question_tuple[4],
            'content': question_tuple[5],
            'source': ast.literal_eval(question_tuple[6]) if question_tuple[6] else {},
            'sourceCategory': 'UDACITY'
        }
    }
    question_payload = {
        "query": create_question_mutation,
        "variables": question_variables
    }
    
    # First, send the question mutation
    try:
        r = requests.post(settings.ASSESSMENTS_API_URL, headers=settings.production_headers(), json=question_payload)
        if r.status_code != 200:
            raise Exception(f"Question mutation failed with status code {r.status_code}: {r.text}")
        response_json = r.json()
        question_id = response_json['data']['createQuestion']['id']
        q_exception = None
    except Exception as e:
        question_id = None
        q_exception = e
    
    # Now, process choices for this question.
    choice_results = []
    if question_id is not None:
        with concurrent.futures.ThreadPoolExecutor() as choice_executor:
            futures = {
                choice_executor.submit(process_choice, question_id, row): idx 
                for idx, row in group.iterrows()
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                choice_results.append(result)
    return (question_payload, question_id, q_exception, choice_results)

def upload_assessment_to_api(df, assessment_title):
    """Upload the complete assessment to the API using concurrent processing"""
    try:
        # Step 1: Create assessment
        with st.spinner("Creating assessment..."):
            assessment_id, assessment_response = create_assessment(assessment_title)
            if not assessment_id:
                return None
        
        # Step 2: Create sections
        section_ids = {}
        unique_sections = df['sectionId'].unique()
        
        with st.spinner(f"Creating {len(unique_sections)} sections..."):
            for section in unique_sections:
                section_id, section_response = create_section(assessment_id, section)
                if section_id:
                    section_ids[section] = section_id
        
        # Step 3: Update sectionId in DataFrame to use actual section IDs
        df['sectionId'] = df['sectionId'].map(section_ids)
        
        # Step 4: Main processing: Group questions and process them concurrently.
        question_problems = []
        choices_problems = []
        
        # Define question columns for grouping
        question_columns = ['sectionId', 'difficultyLevelId', 'skillId', 'category', 'question_status', 'question_content', 'source']
        
        MAX_WORKERS = 5
        
        results = []
        with st.spinner("Creating questions and choices..."):
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_group = {
                    executor.submit(process_question_group, question, group): question 
                    for question, group in df.groupby(question_columns)
                }
                for future in concurrent.futures.as_completed(future_to_group):
                    try:
                        res = future.result(timeout=60)
                        results.append(res)
                    except Exception as e:
                        st.error(f"Error processing a question group: {e}")
        
        # Now, iterate over results to collect problems
        for question_payload, question_id, q_exception, choice_results in results:
            if q_exception is not None:
                question_problems.append((q_exception, question_payload))
            for choice_payload, response_json, c_exception in choice_results:
                if c_exception is not None:
                    choices_problems.append((c_exception, choice_payload))
        
        # Calculate success metrics
        total_questions = len(results)
        successful_questions = total_questions - len(question_problems)
        total_choices = sum(len(choice_results) for _, _, _, choice_results in results)
        successful_choices = total_choices - len(choices_problems)
        
        return {
            'assessment_id': assessment_id,
            'sections_created': len(section_ids),
            'questions_created': successful_questions,
            'choices_created': successful_choices,
            'assessment_preview_url': f'https://learn.udacity.com/assessment-preview?assessmentId={assessment_id}',
            'question_problems': question_problems,
            'choice_problems': choices_problems
        }
        
    except Exception as e:
        st.error(f"Error uploading assessment: {e}")
        return None

#========================================
# UI
#========================================

def main():
    st.title("Uploading Assessments")
    st.markdown("Upload your reviewed assessment CSV file to create your assessment.")
    
    # Initialize session state
    if 'upload_result' not in st.session_state:
        st.session_state.upload_result = None
    
    # Form for complete upload process
    with st.form('complete_upload_assessments'):
        st.markdown('#### Upload and Create Assessment')
        
        # File upload
        csv = st.file_uploader(
            "Upload a CSV generated from the 'Generating Assessments' tab", 
            type="csv",
            help="Select a CSV file with assessment questions and answers"
        )
        
        # Title input
        assessment_title = st.text_input(
            'Assessment Title',
            placeholder='Enter a title for this assessment',
            help="Provide a descriptive title for your assessment"
        )

        st.markdown('#### Staff Password')
        password = st.text_input("Staff Password", type="password", help="Enter the required staff password")
        
        # Submit button
        submitted = st.form_submit_button("Create Assessment", use_container_width=True)
        
        if submitted:
            if password != st.secrets['password']:
                st.error("❌ Incorrect password. Please try again.") 
            elif csv is None:
                st.error("Please upload a CSV file.")
            elif not assessment_title.strip():
                st.error("Please enter an assessment title.")
            else:
                # Load the CSV
                df = pd.read_csv(csv)
                updated_df = add_id_fields(df.copy())
                
                # Upload to API
                with st.spinner("Creating assessment in API..."):
                    result = upload_assessment_to_api(updated_df, assessment_title)
                    
                    if result:
                        st.success("Assessment created successfully!")
                        
                        # Display upload summary
                        col1, col2, col3= st.columns(3)
                        with col1:
                            st.metric("Sections Created", result['sections_created'])
                        with col2:
                            st.metric("Questions Created", result['questions_created'])
                        with col3:
                            st.metric("Choices Created", result['choices_created'])
                        
                        # Display URLs
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Preview URL:**")
                            st.info(f"{result['assessment_preview_url']}")
                        with col2:
                            st.markdown("**Staff View URL:**")
                            staff_url = f"https://manage.udacity.com/admin/assessments/{result['assessment_id']}"
                            st.info(f"{staff_url}")

                    else:
                        st.error("Failed to create assessment. Please check the console for errors.")
    
if __name__ == "__main__":
    main()
