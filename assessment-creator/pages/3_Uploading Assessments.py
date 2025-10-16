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
        r = requests.post(settings.ASSESSMENTS_API_URL, headers=settings.production_headers(), json={"query": query})

        response_json = r.json()        
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

def create_skill(external_id, title):
    """Create a new skill in the assessments API"""
    try:
        create_skill_mutation = """
        mutation createSkill($input: CreateSkillInput!) {
          createSkill(input: $input) {
            id
            externalId
            title
            category
            status
          }
        }
        """
        
        skill_variables = {
            "input": {
                "externalId": external_id,
                "title": title,
                "category": "UTAXONOMY",
                "status": "ACTIVE"
            }
        }
        
        print(f"Creating skill: ExternalID='{external_id}', Title='{title}'")
        
        skill_payload = {
            "query": create_skill_mutation,
            "variables": skill_variables
        }
        
        response = requests.post(
            settings.ASSESSMENTS_API_URL,
            headers=settings.production_headers(),
            json=skill_payload
        )
        
        if response.status_code != 200:
            print(f"ERROR: Skill creation failed with status code {response.status_code}")
            print(f"ERROR: Response text: {response.text}")
            print(f"ERROR: Request payload: {json.dumps(skill_payload, indent=2)}")
            raise Exception(f"Skill mutation failed with status code {response.status_code}: {response.text}")
        
        response_json = response.json()
        print(f"Skill creation response: {json.dumps(response_json, indent=2)}")
        
        if 'data' not in response_json:
            print(f"ERROR: No 'data' key in skill response: {response_json}")
            raise Exception("No 'data' key in skill response")
        
        if 'createSkill' not in response_json['data']:
            print(f"ERROR: No 'createSkill' key in response data: {response_json['data']}")
            raise Exception("No 'createSkill' key in response data")
        
        skill_data = response_json['data']['createSkill']
        skill_id = skill_data['id']
        
        print(f"Successfully created skill with ID: {skill_id}")
        return skill_data
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"CRITICAL ERROR CREATING SKILL: {type(e).__name__}: {str(e)}")
        print(f"External ID: '{external_id}'")
        print(f"Title: '{title}'")
        print(f"Full traceback:")
        traceback.print_exc()
        print(f"{'='*80}")
        return None


def add_id_fields(df):
    """Add difficulty level and skill IDs to the DataFrame"""
    try:

        # Check for required columns
        required_columns = ['difficultyLevelUri', 'skillUri']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"ERROR: Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            raise Exception(f"Missing required columns: {missing_columns}")
        
        # Fetch data from API
        difficulty_levels = fetch_difficulty_levels()        
        skills = fetch_skills()
        
        # Create mappings using externalId for difficulty and skills; also prepare title-based fallback
        difficulty_mapping = {dl['externalId']: dl['id'] for dl in difficulty_levels if dl.get('externalId')}
        skill_mapping_external = {skill['externalId']: skill['id'] for skill in skills if skill.get('externalId')}
        skill_mapping_title = {skill['title'].strip().lower(): skill['id'] for skill in skills if skill.get('title')}

        
        # Check for missing values
        missing_difficulty_uris = df['difficultyLevelUri'].isna().sum()
        missing_skill_uris = df['skillUri'].isna().sum()
        print(f"Missing difficultyLevelUri values: {missing_difficulty_uris}")
        print(f"Missing skillUri values: {missing_skill_uris}")
        
        # Keep originals for logging and fallback
        orig_skill_names = df['skillId'].astype(str)
        
        # Add ID columns using cleaned values
        df['difficultyLevelId'] = df['difficultyLevelUri'].map(difficulty_mapping).fillna('')        
        df['skillId'] = df['skillUri'].map(skill_mapping_external).fillna('')
        
        # Fallback: map by skill title for rows still empty
        mask_unmapped = df['skillId'] == ''
        if mask_unmapped.any():
            print(f"Fallback mapping by title for {mask_unmapped.sum()} rows...")
            df.loc[mask_unmapped, 'skillId'] = (
                orig_skill_names[mask_unmapped].str.strip().str.lower().map(skill_mapping_title).fillna('')
            )
        
        # Find unmapped values
        unmapped_difficulties = df[df['difficultyLevelId'] == '']['difficultyLevelUri'].unique()
        unmapped_skill_uris = df[df['skillId'] == '']['skillUri'].unique()
        unmapped_skill_titles = orig_skill_names[df['skillId'] == ''].unique()
        
        if len(unmapped_difficulties) > 0:
            print(f"\nWARNING: {len(unmapped_difficulties)} unmapped difficulty URIs:")
            for diff in unmapped_difficulties[:10]:  # Show first 10
                print(f"  '{diff}'")
            if len(unmapped_difficulties) > 10:
                print(f"  ... and {len(unmapped_difficulties) - 10} more")
        
        # Handle unmapped skills by creating them
        if len(unmapped_skill_uris) > 0:
            print(f"\nWARNING: Unmapped skills remain after fallback.")
            print(f"- Unmapped skillUri values: {len(unmapped_skill_uris)}")
            
            # Create missing skills
            print(f"\nCreating {len(unmapped_skill_uris)} missing skills...")
            created_skills = []
            failed_skills = []
            
            # Get corresponding titles for unmapped skillUris
            unmapped_rows = df[df['skillId'] == '']
            skill_uri_to_title = {}
            
            for _, row in unmapped_rows.iterrows():
                skill_uri = row['skillUri']
                skill_title = row['skillId']  # Original skill name/title
                if skill_uri not in skill_uri_to_title:
                    skill_uri_to_title[skill_uri] = skill_title
            
            for skill_uri in unmapped_skill_uris:
                if pd.isna(skill_uri) or skill_uri == '':
                    print(f"  Skipping empty/NaN skillUri")
                    continue
                    
                skill_title = skill_uri_to_title.get(skill_uri, skill_uri)  # Use URI as title if no title found
                
                print(f"  Creating skill: URI='{skill_uri}', Title='{skill_title}'")
                created_skill = create_skill(skill_uri, skill_title)
                
                if created_skill:
                    created_skills.append(created_skill)
                    print(f"    SUCCESS: Created skill ID {created_skill['id']}")
                else:
                    failed_skills.append((skill_uri, skill_title))
                    print(f"    FAILED: Could not create skill")
            
            print(f"\nSkill creation summary:")
            print(f"  Successfully created: {len(created_skills)}")
            print(f"  Failed to create: {len(failed_skills)}")
            
            if failed_skills:
                print(f"  Failed skills:")
                for uri, title in failed_skills:
                    print(f"    - URI: '{uri}', Title: '{title}'")
            
            # If we successfully created skills, update our mappings and retry
            if created_skills:
                print(f"\nRetrying skill mapping with newly created skills...")
                
                # Update skill mappings with newly created skills
                for skill in created_skills:
                    skill_mapping_external[skill['externalId']] = skill['id']
                    skill_mapping_title[skill['title'].strip().lower()] = skill['id']
                
                # Retry mapping for previously unmapped skills
                df['skillId'] = df['skillUri'].map(skill_mapping_external).fillna('')
                
                # Fallback: map by skill title for rows still empty
                mask_unmapped = df['skillId'] == ''
                if mask_unmapped.any():
                    print(f"Final fallback mapping by title for {mask_unmapped.sum()} rows...")
                    df.loc[mask_unmapped, 'skillId'] = (
                        orig_skill_names[mask_unmapped].str.strip().str.lower().map(skill_mapping_title).fillna('')
                    )
                
                # Check final unmapped count
                final_unmapped_skill_uris = df[df['skillId'] == '']['skillUri'].unique()
                final_unmapped_skill_titles = orig_skill_names[df['skillId'] == ''].unique()
                
                print(f"Final unmapped skills after creation:")
                print(f"  - Unmapped skillUri values: {len(final_unmapped_skill_uris)}")
                print(f"  - Unmapped skill titles: {len(final_unmapped_skill_titles)}")
                
                if len(final_unmapped_skill_uris) > 0:
                    print(f"Remaining unmapped skillUri values:")
                    for skill in final_unmapped_skill_uris[:10]:
                        print(f"    '{skill}'")
                    if len(final_unmapped_skill_uris) > 10:
                        print(f"    ... and {len(final_unmapped_skill_uris) - 10} more")
        
        elif len(unmapped_skill_titles) > 0:
            print(f"- Unmapped skill titles: {len(unmapped_skill_titles)}")
            for title in unmapped_skill_titles[:10]:  # Show first 10
                print(f"  '{title}'")
            if len(unmapped_skill_titles) > 10:
                print(f"  ... and {len(unmapped_skill_titles) - 10} more")
        
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
    try:
        
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
                "termsAgreement": "",
                "assessmentType": "UNKNOWN" 
            }
        }
        
        print(f"Assessment variables prepared:")
        for key, value in assessment_variables["input"].items():
            print(f"  {key}: {value}")
        
        assessment_payload = {
            "query": create_assessment_mutation,
            "variables": assessment_variables
        }
        

        response = requests.post(
            settings.ASSESSMENTS_API_URL,
            headers=settings.production_headers(),
            json=assessment_payload
        )
 
        if response.status_code != 200:
            print(f"ERROR: Assessment creation failed with status code {response.status_code}")
            print(f"ERROR: Response text: {response.text}")
            print(f"ERROR: Request payload: {json.dumps(assessment_payload, indent=2)}")
            raise Exception(f"GraphQL mutation failed with status code {response.status_code}: {response.text}")
        
        response_json = response.json()
        print(f"Assessment creation response JSON keys: {list(response_json.keys())}")
        print(f"Assessment creation response: {json.dumps(response_json, indent=2)}")
        
        assessment_id = response_json['data']['createAssessment']['id']
        
        return assessment_id, response_json
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"CRITICAL ERROR CREATING ASSESSMENT: {type(e).__name__}: {str(e)}")
        print(f"Assessment title: '{assessment_title}'")
        print(f"Full traceback:")
        traceback.print_exc()
        print(f"{'='*80}")
        st.error(f"Error creating assessment: {e}")
        return None, None

def create_section(assessment_id, section_title, num_questions_to_ask=10):
    """Create a section for an assessment"""
    try:

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
        
        print(f"Section variables prepared:")
        for key, value in section_variables["input"].items():
            print(f"  {key}: {value}")
        
        section_payload = {
            "query": create_section_mutation,
            "variables": section_variables
        }
        
        print(f"Sending section creation request...")
        response = requests.post(
            settings.ASSESSMENTS_API_URL,
            headers=settings.production_headers(),
            json=section_payload
        )
        
        print(f"Section creation response status: {response.status_code}")
        print(f"Section creation response headers: {dict(response.headers)}")
        
        if response.status_code != 200:
            print(f"ERROR: Section creation failed with status code {response.status_code}")
            print(f"ERROR: Response text: {response.text}")
            print(f"ERROR: Request payload: {json.dumps(section_payload, indent=2)}")
            raise Exception(f"GraphQL mutation failed with status code {response.status_code}: {response.text}")
        
        response_json = response.json()
        print(f"Section creation response JSON keys: {list(response_json.keys())}")
        print(f"Section creation response: {json.dumps(response_json, indent=2)}")
        
        if 'data' not in response_json:
            print(f"ERROR: No 'data' key in section response: {response_json}")
            raise Exception("No 'data' key in section response")
        
        if 'createSection' not in response_json['data']:
            print(f"ERROR: No 'createSection' key in response data: {response_json['data']}")
            raise Exception("No 'createSection' key in response data")
        
        section_id = response_json['data']['createSection']['id']
        
        return section_id, response_json
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"CRITICAL ERROR CREATING SECTION: {type(e).__name__}: {str(e)}")
        print(f"Assessment ID: '{assessment_id}'")
        print(f"Section title: '{section_title}'")
        print(f"Full traceback:")
        traceback.print_exc()
        print(f"{'='*80}")
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

        if question_data['category'] == "MULTIPLE_CHOICE":
            category_enum = 0
        elif question_data['category'] == "SINGLE_CHOICE":
            category_enum = 3
        
        question_variables = {
            "input": {
                'sectionId': section_id,
                'difficultyLevelId': question_data['difficultyLevelId'],
                'skillId': question_data['skillId'],
                'category': category_enum,
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
    try:
        
        # Validate required fields
        required_fields = ['choice_content', 'choice_isCorrect', 'choice_orderIndex', 'choice_status']
        missing_fields = [field for field in required_fields if field not in row or pd.isna(row[field])]
        if missing_fields:
            print(f"ERROR: Missing required choice fields: {missing_fields}")
            print(f"Available row fields: {list(row.index)}")
            raise Exception(f"Missing required choice fields: {missing_fields}")
        
        # Log choice details
        print(f"Choice content: '{row['choice_content']}' (length: {len(str(row['choice_content']))})")
        print(f"Is correct: {row['choice_isCorrect']} (type: {type(row['choice_isCorrect'])})")
        print(f"Order index: {row['choice_orderIndex']} (type: {type(row['choice_orderIndex'])})")
        print(f"Status: '{row['choice_status']}'")
        
        # Validate choice data
        if not row['choice_content'] or str(row['choice_content']).strip() == '':
            print(f"ERROR: Empty choice content")
            raise Exception("Empty choice content")
        
        if not isinstance(row['choice_isCorrect'], bool):
            print(f"WARNING: choice_isCorrect is not boolean: {row['choice_isCorrect']} (type: {type(row['choice_isCorrect'])})")
            # Try to convert
            if str(row['choice_isCorrect']).lower() in ['true', '1', 'yes']:
                row['choice_isCorrect'] = True
                print(f"Converted to True")
            else:
                row['choice_isCorrect'] = False
                print(f"Converted to False")
        
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
        
        r = requests.post(settings.ASSESSMENTS_API_URL, headers=settings.production_headers(), json=choice_payload)

        if r.status_code != 200:
            print(f"ERROR: Choice mutation failed with status code {r.status_code}")
            print(f"ERROR: Response text: {r.text}")
            print(f"ERROR: Request payload: {json.dumps(choice_payload, indent=2)}")
            raise Exception(f"Choice mutation failed with status code {r.status_code}: {r.text}")
        
        response_json = r.json()

        if 'data' not in response_json:
            print(f"ERROR: No 'data' key in choice response: {response_json}")
            raise Exception("No 'data' key in choice response")
        
        if 'createChoice' not in response_json['data']:
            print(f"ERROR: No 'createChoice' key in response data: {response_json['data']}")
            raise Exception("No 'createChoice' key in response data")
        
        choice_id = response_json['data']['createChoice']['id']
        print(f"{'='*60}")
        
        return (choice_payload, response_json, None)
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"CRITICAL ERROR CREATING CHOICE: {type(e).__name__}: {str(e)}")
        print(f"Question ID: '{question_id}'")
        print(f"Row data: {dict(row) if 'row' in locals() else 'UNKNOWN'}")
        print(f"Full traceback:")
        traceback.print_exc()
        print(f"{'='*60}")
        return (choice_payload if 'choice_payload' in locals() else None, None, e)

def process_question_group(question_tuple, group):
    """
    For a given grouped question (question_tuple with associated rows in group),
    send the createQuestion mutation first, and then concurrently process each choice.
    Returns a tuple: (question_payload, question_id, question_exception, list_of_choice_results)
    Each element in list_of_choice_results is (choice_payload, response_json, exception)
    """
    # Ensure variables are always defined to avoid UnboundLocalError on early exceptions
    question_payload = None
    question_id = None
    q_exception = None
    choice_results = []
    
    try:
        
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
    if question_id is not None:
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
    
    
    return (question_payload, question_id, q_exception, choice_results)

def upload_assessment_to_api(df, assessment_title):
    """Upload the complete assessment to the API using concurrent processing"""
    try:

        # Check for critical missing data
        critical_columns = ['sectionId', 'difficultyLevelId', 'skillId', 'question_content']
        missing_columns = [col for col in critical_columns if col not in df.columns]
        if missing_columns:
            print(f"ERROR: Missing critical columns: {missing_columns}")
            raise Exception(f"Missing critical columns: {missing_columns}")
        
        # Check for empty values in critical columns
        for col in critical_columns:
            empty_count = (df[col].isna() | (df[col] == '')).sum()
            if empty_count > 0:
                print(f"WARNING: {empty_count} empty values in column '{col}'")
                print(f"Sample empty rows in {col}:")
                empty_rows = df[df[col].isna() | (df[col] == '')]
                for idx, row in empty_rows.head(3).iterrows():
                    print(f"  Row {idx}: {dict(row)}")
        
        # Step 1: Create assessment
        print(f"\n{'='*50}")
        print("STEP 1: CREATING ASSESSMENT")
        print(f"{'='*50}")
        
        with st.spinner("Creating assessment..."):
            assessment_id, assessment_response = create_assessment(assessment_title)
            if not assessment_id:
                print(f"ERROR: Assessment creation failed, aborting upload")
                return None
        
        print(f"Assessment created successfully with ID: {assessment_id}")

        # Step 2: Create sections
        print(f"\n{'='*50}")
        print("STEP 2: CREATING SECTIONS")
        print(f"{'='*50}")
        
        section_ids = {}
        unique_sections = df['sectionId'].unique()
        
        print(f"Found {len(unique_sections)} unique sections to create:")
        for i, section in enumerate(unique_sections):
            print(f"  {i+1}. '{section}'")
        
        with st.spinner(f"Creating {len(unique_sections)} sections..."):
            for section in unique_sections:
                print(f"\nCreating section: '{section}'")
                section_id, section_response = create_section(assessment_id, section)
                if section_id:
                    section_ids[section] = section_id
                    print(f"Section '{section}' created with ID: {section_id}")
                else:
                    print(f"ERROR: Failed to create section '{section}'")
        
        print(f"\nSection creation summary:")
        print(f"Sections requested: {len(unique_sections)}")
        print(f"Sections created: {len(section_ids)}")
        print(f"Section ID mapping: {section_ids}")
        
        if len(section_ids) == 0:
            print(f"ERROR: No sections were created, aborting upload")
            return None

        # Step 3: Update sectionId in DataFrame to use actual section IDs
        print(f"\n{'='*50}")
        print("STEP 3: UPDATING DATAFRAME WITH SECTION IDS")
        print(f"{'='*50}")
        
        print(f"Original sectionId values: {df['sectionId'].value_counts().to_dict()}")
        df['sectionId'] = df['sectionId'].map(section_ids)
        print(f"Mapped sectionId values: {df['sectionId'].value_counts().to_dict()}")
        
        # Check for unmapped sections
        unmapped_sections = df['sectionId'].isna().sum()
        if unmapped_sections > 0:
            print(f"ERROR: {unmapped_sections} rows have unmapped sectionId values")
            unmapped_rows = df[df['sectionId'].isna()]
            print(f"Sample unmapped rows:")
            for idx, row in unmapped_rows.head(3).iterrows():
                print(f"  Row {idx}: sectionId='{row.get('sectionId', 'MISSING')}', question_content='{str(row.get('question_content', ''))[:50]}...'")

        # Step 4: Main processing: Group questions and process them concurrently.
  
        
        question_problems = []
        choices_problems = []
        
        # Define question columns for grouping
        question_columns = ['sectionId', 'difficultyLevelId', 'skillId', 'category', 'question_status', 'question_content', 'source']
                
        # Check if all grouping columns exist
        missing_grouping_cols = [col for col in question_columns if col not in df.columns]
        if missing_grouping_cols:
            print(f"ERROR: Missing grouping columns: {missing_grouping_cols}")
            print(f"Available columns: {list(df.columns)}")
            raise Exception(f"Missing grouping columns: {missing_grouping_cols}")
        
        # Group the DataFrame
        grouped = df.groupby(question_columns)
        print(f"Created {grouped.ngroups} question groups")
        
        MAX_WORKERS = 5
        print(f"Using {MAX_WORKERS} concurrent workers for question processing")
        
        results = []
        with st.spinner("Creating questions and choices..."):
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                print(f"Submitting {grouped.ngroups} question groups to thread pool...")
                
                future_to_group = {
                    executor.submit(process_question_group, question, group): (i, question) 
                    for i, (question, group) in enumerate(grouped)
                }
                
                completed_groups = 0
                for future in concurrent.futures.as_completed(future_to_group):
                    group_info = future_to_group[future]
                    group_index, question_tuple = group_info
                    completed_groups += 1
                    
                    print(f"\nCompleted group {completed_groups}/{grouped.ngroups} (Group index: {group_index})")
                    
                    try:
                        res = future.result(timeout=60)
                        results.append(res)
                        
                        # Log result summary
                        question_payload, question_id, q_exception, choice_results = res
                        if q_exception is None and question_id:
                            pass
                        else:
                            print(f"  FAILURE: Question creation failed - {q_exception}")
                            
                    except Exception as e:
                        print(f"  ERROR: Exception processing group {group_index}: {type(e).__name__}: {str(e)}")
                        st.error(f"Error processing a question group: {e}")
        

        # Now, iterate over results to collect problems
        print(f"\n{'='*50}")
        print("STEP 5: ANALYZING RESULTS AND PROBLEMS")
        print(f"{'='*50}")
        
        for i, (question_payload, question_id, q_exception, choice_results) in enumerate(results):
            print(f"\nAnalyzing result {i+1}/{len(results)}:")
            print(f"  Question ID: {question_id}")
            print(f"  Question Exception: {q_exception}")
            print(f"  Choice Results Count: {len(choice_results)}")
            
            if q_exception is not None:
                question_problems.append((q_exception, question_payload))
                print(f"  Added to question problems: {type(q_exception).__name__}: {str(q_exception)}")
                
            for j, (choice_payload, response_json, c_exception) in enumerate(choice_results):
                if c_exception is not None:
                    choices_problems.append((c_exception, choice_payload))
                    print(f"    Choice {j+1} problem: {type(c_exception).__name__}: {str(c_exception)}")

        # Calculate success metrics
        total_questions = len(results)
        successful_questions = total_questions - len(question_problems)
        total_choices = sum(len(choice_results) for _, _, _, choice_results in results)
        successful_choices = total_choices - len(choices_problems)
        
        print(f"\n{'='*50}")
        print("UPLOAD SUMMARY")
        print(f"{'='*50}")
        print(f"Assessment ID: {assessment_id}")
        print(f"Sections created: {len(section_ids)}")
        print(f"Total question groups: {total_questions}")
        print(f"Successful questions: {successful_questions}")
        print(f"Failed questions: {len(question_problems)}")
        print(f"Total choices: {total_choices}")
        print(f"Successful choices: {successful_choices}")
        print(f"Failed choices: {len(choices_problems)}")
        
        if question_problems:
            print(f"\nQUESTION PROBLEMS SUMMARY:")
            for i, (exception, payload) in enumerate(question_problems):
                print(f"  Problem {i+1}: {type(exception).__name__}: {str(exception)}")
        
        if choices_problems:
            print(f"\nCHOICE PROBLEMS SUMMARY:")
            for i, (exception, payload) in enumerate(choices_problems):
                print(f"  Problem {i+1}: {type(exception).__name__}: {str(exception)}")
        
        print(f"{'='*80}")
        
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
        print(f"\n{'='*80}")
        print(f"CRITICAL ERROR IN UPLOAD_ASSESSMENT_TO_API: {type(e).__name__}: {str(e)}")
        print(f"Assessment title: '{assessment_title}'")
        print(f"DataFrame shape: {df.shape if 'df' in locals() else 'UNKNOWN'}")
        print(f"Full traceback:")
        traceback.print_exc()
        print(f"{'='*80}")
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
