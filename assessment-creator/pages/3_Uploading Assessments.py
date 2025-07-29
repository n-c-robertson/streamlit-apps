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
    
    try:
        print(f"\n{'='*80}")
        print("FETCHING SKILLS FROM API")
        print(f"{'='*80}")
        
        r = requests.post(settings.ASSESSMENTS_API_URL, headers=settings.production_headers(), json={"query": query})
        
        print(f"Skills API Response Status: {r.status_code}")
        print(f"Skills API Response Headers: {dict(r.headers)}")
        
        if r.status_code != 200:
            print(f"ERROR: Skills API returned non-200 status: {r.status_code}")
            print(f"ERROR: Skills API Response Text: {r.text}")
            raise Exception(f"Skills API failed with status {r.status_code}: {r.text}")
        
        response_json = r.json()
        print(f"Skills API Response JSON Keys: {list(response_json.keys())}")
        
        if 'data' not in response_json:
            print(f"ERROR: No 'data' key in skills response: {response_json}")
            raise Exception("No 'data' key in skills response")
        
        if 'skills' not in response_json['data']:
            print(f"ERROR: No 'skills' key in skills data: {response_json['data']}")
            raise Exception("No 'skills' key in skills data")
        
        skills = response_json['data']['skills']
        print(f"Successfully fetched {len(skills)} skills from API")
        
        # Log each skill for debugging
        for i, skill in enumerate(skills):
            print(f"Skill {i+1}: ID={skill.get('id', 'MISSING')}, ExternalID={skill.get('externalId', 'MISSING')}, Title={skill.get('title', 'MISSING')}, Category={skill.get('category', 'MISSING')}, Status={skill.get('status', 'MISSING')}")
        
        # Check for skills with missing externalId
        skills_without_external_id = [s for s in skills if not s.get('externalId')]
        if skills_without_external_id:
            print(f"WARNING: {len(skills_without_external_id)} skills missing externalId:")
            for skill in skills_without_external_id:
                print(f"  - ID: {skill.get('id')}, Title: {skill.get('title')}")
        
        # Check for skills with missing id
        skills_without_id = [s for s in skills if not s.get('id')]
        if skills_without_id:
            print(f"WARNING: {len(skills_without_id)} skills missing id:")
            for skill in skills_without_id:
                print(f"  - ExternalID: {skill.get('externalId')}, Title: {skill.get('title')}")
        
        print(f"{'='*80}")
        return skills
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"CRITICAL ERROR FETCHING SKILLS: {type(e).__name__}: {str(e)}")
        print(f"Full traceback:")
        traceback.print_exc()
        print(f"{'='*80}")
        raise e


def add_id_fields(df):
    """Add difficulty level and skill IDs to the DataFrame"""
    try:
        print(f"\n{'='*80}")
        print("ADDING ID FIELDS TO DATAFRAME")
        print(f"{'='*80}")
        
        print(f"Input DataFrame shape: {df.shape}")
        print(f"Input DataFrame columns: {list(df.columns)}")
        
        # Check for required columns
        required_columns = ['difficultyLevelUri', 'skillUri']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"ERROR: Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            raise Exception(f"Missing required columns: {missing_columns}")
        
        # Fetch data from API
        print("\nFetching difficulty levels...")
        difficulty_levels = fetch_difficulty_levels()
        print(f"Fetched {len(difficulty_levels)} difficulty levels")
        
        print("\nFetching skills...")
        skills = fetch_skills()
        print(f"Fetched {len(skills)} skills")
        
        # Create mappings using externalId
        difficulty_mapping = {dl['externalId']: dl['id'] for dl in difficulty_levels if dl.get('externalId')}
        skill_mapping = {skill['externalId']: skill['id'] for skill in skills if skill.get('externalId')}
        
        print(f"\nDifficulty mapping created: {len(difficulty_mapping)} mappings")
        print(f"Skill mapping created: {len(skill_mapping)} mappings")
        
        # Log some sample mappings
        print("\nSample difficulty mappings:")
        for i, (ext_id, int_id) in enumerate(list(difficulty_mapping.items())[:5]):
            print(f"  {ext_id} -> {int_id}")
        
        print("\nSample skill mappings:")
        for i, (ext_id, int_id) in enumerate(list(skill_mapping.items())[:5]):
            print(f"  {ext_id} -> {int_id}")
        
        # Analyze DataFrame data before mapping
        print(f"\nDataFrame analysis before mapping:")
        print(f"Unique difficultyLevelUri values: {df['difficultyLevelUri'].nunique()}")
        print(f"Unique skillUri values: {df['skillUri'].nunique()}")
        
        # Check for missing values
        missing_difficulty_uris = df['difficultyLevelUri'].isna().sum()
        missing_skill_uris = df['skillUri'].isna().sum()
        print(f"Missing difficultyLevelUri values: {missing_difficulty_uris}")
        print(f"Missing skillUri values: {missing_skill_uris}")
        
        # Show sample values
        print(f"\nSample difficultyLevelUri values:")
        sample_difficulties = df['difficultyLevelUri'].dropna().unique()[:10]
        for diff in sample_difficulties:
            print(f"  '{diff}'")
        
        print(f"\nSample skillUri values:")
        sample_skills = df['skillUri'].dropna().unique()[:10]
        for skill in sample_skills:
            print(f"  '{skill}'")
        
        # Add ID columns using cleaned values
        print(f"\nMapping difficulty levels...")
        df['difficultyLevelId'] = df['difficultyLevelUri'].map(difficulty_mapping).fillna('')
        
        print(f"Mapping skills...")
        df['skillId'] = df['skillUri'].map(skill_mapping).fillna('')
        
        # Analyze mapping results
        print(f"\nMapping results:")
        mapped_difficulties = (df['difficultyLevelId'] != '').sum()
        mapped_skills = (df['skillId'] != '').sum()
        total_rows = len(df)
        
        print(f"Difficulty levels mapped: {mapped_difficulties}/{total_rows} ({mapped_difficulties/total_rows*100:.1f}%)")
        print(f"Skills mapped: {mapped_skills}/{total_rows} ({mapped_skills/total_rows*100:.1f}%)")
        
        # Find unmapped values
        unmapped_difficulties = df[df['difficultyLevelId'] == '']['difficultyLevelUri'].unique()
        unmapped_skills = df[df['skillId'] == '']['skillUri'].unique()
        
        if len(unmapped_difficulties) > 0:
            print(f"\nWARNING: {len(unmapped_difficulties)} unmapped difficulty URIs:")
            for diff in unmapped_difficulties[:10]:  # Show first 10
                print(f"  '{diff}'")
            if len(unmapped_difficulties) > 10:
                print(f"  ... and {len(unmapped_difficulties) - 10} more")
        
        if len(unmapped_skills) > 0:
            print(f"\nWARNING: {len(unmapped_skills)} unmapped skill URIs:")
            for skill in unmapped_skills[:10]:  # Show first 10
                print(f"  '{skill}'")
            if len(unmapped_skills) > 10:
                print(f"  ... and {len(unmapped_skills) - 10} more")
        
        # Show final DataFrame info
        print(f"\nFinal DataFrame shape: {df.shape}")
        print(f"Final DataFrame columns: {list(df.columns)}")
        
        print(f"{'='*80}")
        return df
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"CRITICAL ERROR IN ADD_ID_FIELDS: {type(e).__name__}: {str(e)}")
        print(f"Full traceback:")
        traceback.print_exc()
        print(f"{'='*80}")
        raise e

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
    
    try:
        print(f"\n{'='*80}")
        print("CREATING QUESTION")
        print(f"{'='*80}")
        
        # Log input data
        print(f"Section ID: {section_id}")
        print(f"Question data keys: {list(question_data.keys())}")
        
        # Validate required fields
        required_fields = ['difficultyLevelId', 'skillId', 'category', 'question_status', 'question_content']
        missing_fields = [field for field in required_fields if field not in question_data]
        if missing_fields:
            print(f"ERROR: Missing required fields: {missing_fields}")
            print(f"Available fields: {list(question_data.keys())}")
            raise Exception(f"Missing required fields: {missing_fields}")
        
        # Log skill and difficulty information
        print(f"Skill ID: '{question_data['skillId']}' (type: {type(question_data['skillId'])})")
        print(f"Difficulty Level ID: '{question_data['difficultyLevelId']}' (type: {type(question_data['difficultyLevelId'])})")
        print(f"Category: '{question_data['category']}'")
        print(f"Status: '{question_data['question_status']}'")
        print(f"Content length: {len(question_data['question_content'])} characters")
        
        # Check for empty skill or difficulty IDs
        if not question_data['skillId'] or question_data['skillId'] == '':
            print(f"ERROR: Empty skillId - this will cause the question creation to fail")
            print(f"Original skillUri: {question_data.get('skillUri', 'NOT_FOUND')}")
            raise Exception("Empty skillId - skill mapping failed")
        
        if not question_data['difficultyLevelId'] or question_data['difficultyLevelId'] == '':
            print(f"ERROR: Empty difficultyLevelId - this will cause the question creation to fail")
            print(f"Original difficultyLevelUri: {question_data.get('difficultyLevelUri', 'NOT_FOUND')}")
            raise Exception("Empty difficultyLevelId - difficulty mapping failed")
        
        # Handle source data properly - convert to CreateQuestionSourceInput format
        source_data = question_data.get('source', {})
        if isinstance(source_data, str):
            try:
                source_data = ast.literal_eval(source_data)
                print(f"Successfully parsed source data from string")
            except Exception as e:
                print(f"WARNING: Failed to parse source data string: {e}")
                source_data = {}
        
        # Format source as CreateQuestionSourceInput - only include the most basic valid fields
        formatted_source = {
            "uri": source_data.get('uri', ''),
            "key": source_data.get('partKey', '')
        }
        
        # Remove any None or empty values to avoid validation issues
        formatted_source = {k: v for k, v in formatted_source.items() if v is not None and v != ''}
        print(f"Formatted source: {formatted_source}")
        
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
        
        print(f"Question variables prepared successfully")
        
        question_payload = {
            "query": create_question_mutation,
            "variables": question_variables
        }
        
        print(f"Sending question creation request...")
        response = requests.post(
            settings.ASSESSMENTS_API_URL,
            headers=settings.production_headers(),
            json=question_payload
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code != 200:
            print(f"ERROR: Question mutation failed with status code {response.status_code}")
            print(f"ERROR: Response text: {response.text}")
            print(f"ERROR: Request payload: {question_payload}")
            raise Exception(f"Question mutation failed with status code {response.status_code}: {response.text}")
        
        response_json = response.json()
        print(f"Response JSON keys: {list(response_json.keys())}")
        
        if 'data' not in response_json:
            print(f"ERROR: No 'data' key in response: {response_json}")
            raise Exception("No 'data' key in response")
        
        if 'createQuestion' not in response_json['data']:
            print(f"ERROR: No 'createQuestion' key in response data: {response_json['data']}")
            raise Exception("No 'createQuestion' key in response data")
        
        question_id = response_json['data']['createQuestion']['id']
        print(f"Successfully created question with ID: {question_id}")
        
        print(f"{'='*80}")
        return question_id, response_json
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"CRITICAL ERROR CREATING QUESTION: {type(e).__name__}: {str(e)}")
        print(f"Question data that failed:")
        for key, value in question_data.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: '{value[:100]}...' (truncated)")
            else:
                print(f"  {key}: '{value}'")
        print(f"Full traceback:")
        traceback.print_exc()
        print(f"{'='*80}")
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
    try:
        print(f"\n{'='*80}")
        print("PROCESSING QUESTION GROUP")
        print(f"{'='*80}")
        
        # Log question tuple information
        print(f"Question tuple length: {len(question_tuple)}")
        print(f"Question tuple values: {question_tuple}")
        print(f"Group size: {len(group)} rows")
        
        # Validate question tuple structure
        if len(question_tuple) < 7:
            print(f"ERROR: Question tuple too short, expected 7 elements, got {len(question_tuple)}")
            raise Exception(f"Invalid question tuple length: {len(question_tuple)}")
        
        # Extract and validate individual fields
        section_id = question_tuple[0]
        difficulty_level_id = question_tuple[1]
        skill_id = question_tuple[2]
        category = question_tuple[3]
        status = question_tuple[4]
        content = question_tuple[5]
        source_str = question_tuple[6]
        
        print(f"Extracted fields:")
        print(f"  Section ID: '{section_id}' (type: {type(section_id)})")
        print(f"  Difficulty Level ID: '{difficulty_level_id}' (type: {type(difficulty_level_id)})")
        print(f"  Skill ID: '{skill_id}' (type: {type(skill_id)})")
        print(f"  Category: '{category}' (type: {type(category)})")
        print(f"  Status: '{status}' (type: {type(status)})")
        print(f"  Content length: {len(content) if content else 0} characters")
        print(f"  Source string: '{source_str}' (type: {type(source_str)})")
        
        # Check for critical failures
        if not skill_id or skill_id == '':
            print(f"ERROR: Empty skillId in question tuple - this will cause failure")
            print(f"Original skillUri from group: {group['skillUri'].iloc[0] if 'skillUri' in group.columns else 'NOT_FOUND'}")
            raise Exception("Empty skillId in question tuple")
        
        if not difficulty_level_id or difficulty_level_id == '':
            print(f"ERROR: Empty difficultyLevelId in question tuple - this will cause failure")
            print(f"Original difficultyLevelUri from group: {group['difficultyLevelUri'].iloc[0] if 'difficultyLevelUri' in group.columns else 'NOT_FOUND'}")
            raise Exception("Empty difficultyLevelId in question tuple")
        
        if not section_id or section_id == '':
            print(f"ERROR: Empty sectionId in question tuple - this will cause failure")
            raise Exception("Empty sectionId in question tuple")
        
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
        
        # Parse source data
        try:
            source_data = ast.literal_eval(source_str) if source_str else {}
            print(f"Successfully parsed source data: {source_data}")
        except Exception as e:
            print(f"WARNING: Failed to parse source data: {e}")
            source_data = {}
        
        question_variables = {
            "input": {
                'sectionId': section_id,
                'difficultyLevelId': difficulty_level_id,
                'skillId': skill_id,
                'category': category,
                'status': status,
                'content': content,
                'source': source_data,
                'sourceCategory': 'UDACITY'
            }
        }
        
        print(f"Question variables prepared successfully")
        
        question_payload = {
            "query": create_question_mutation,
            "variables": question_variables
        }
        
        # First, send the question mutation
        print(f"Sending question creation request...")
        r = requests.post(settings.ASSESSMENTS_API_URL, headers=settings.production_headers(), json=question_payload)
        
        print(f"Question creation response status: {r.status_code}")
        
        if r.status_code != 200:
            print(f"ERROR: Question mutation failed with status code {r.status_code}")
            print(f"ERROR: Response text: {r.text}")
            print(f"ERROR: Request payload: {question_payload}")
            raise Exception(f"Question mutation failed with status code {r.status_code}: {r.text}")
        
        response_json = r.json()
        print(f"Question creation response JSON keys: {list(response_json.keys())}")
        
        if 'data' not in response_json:
            print(f"ERROR: No 'data' key in question response: {response_json}")
            raise Exception("No 'data' key in question response")
        
        if 'createQuestion' not in response_json['data']:
            print(f"ERROR: No 'createQuestion' key in question response data: {response_json['data']}")
            raise Exception("No 'createQuestion' key in question response data")
        
        question_id = response_json['data']['createQuestion']['id']
        print(f"Successfully created question with ID: {question_id}")
        q_exception = None
        
    except Exception as e:
        print(f"ERROR: Failed to create question: {type(e).__name__}: {str(e)}")
        question_id = None
        q_exception = e
        print(f"Full traceback:")
        traceback.print_exc()
    
    # Now, process choices for this question.
    choice_results = []
    if question_id is not None:
        print(f"Processing {len(group)} choices for question {question_id}...")
        with concurrent.futures.ThreadPoolExecutor() as choice_executor:
            futures = {
                choice_executor.submit(process_choice, question_id, row): idx 
                for idx, row in group.iterrows()
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    choice_results.append(result)
                except Exception as e:
                    print(f"ERROR: Choice processing failed: {e}")
                    choice_results.append((None, None, e))
    else:
        print(f"Skipping choice processing due to question creation failure")
    
    print(f"Question group processing complete. Question ID: {question_id}, Choices processed: {len(choice_results)}")
    print(f"{'='*80}")
    
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
            if password != settings.PASSWORD:
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
