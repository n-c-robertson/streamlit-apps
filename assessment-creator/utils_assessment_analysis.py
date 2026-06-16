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
import sys
import time
import traceback
from collections import Counter
from functools import lru_cache
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import graphql_queries
import settings
import prompts

#========================================
# CONSTANTS
#========================================

# Tables with one row per attempt can grow very large. Show this many rows in the
# UI preview and offer a download button for the full dataset.
_TABLE_PREVIEW_ROWS = 100

#========================================
# FUNCTIONS
#========================================

def extract_question_details(df):
    """Return a small per-question metadata DataFrame indexed by questionId.

    This must be called BEFORE slim_results_df() because slimming drops
    the text columns this function depends on.
    """
    text_cols = ['questionId', 'questionContent', 'questionChoices', 'conceptTitle']
    available = [c for c in text_cols if c in df.columns]
    if 'questionId' not in available:
        return pd.DataFrame()
    return (
        df[available]
        .drop_duplicates(subset='questionId')
        .set_index('questionId')
    )


def slim_results_df(df):
    """Drop large text columns and downcast numeric types to shrink memory.

    questionContent and questionChoices are repeated once per row (one row
    per question per attempt), so they scale with attempts × questions and
    dominate memory for large datasets. After calling extract_question_details()
    they are no longer needed in the main DataFrame.
    """
    drop_cols = ['questionContent', 'questionChoices']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    for col in ['questionScore', 'totalScore', 'sectionScore']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
    for col in ['sectionTitle', 'difficultyLabel', 'skillsTitle']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df

# Query template
ATTEMPTS_QUERY = """
query ($assessmentId: ID!, $limit: Int!, $page: Int!) {
  attempts(input: {
    assessmentId: $assessmentId
    limit: $limit
    page: $page
  }) {
    totalCount
    totalPages
    attempts {
      id
      userId
      createdAt
      status
      result
      report {
        totalScore
        sectionReports {
          sectionId
          sectionScore
          section {
            title
          }
          questionReports {
            questionId
            questionScore
            question {
              difficultyLevelId
              skillId
              content
              choices {
                content
                isCorrect
              }
              source {
                uri
                conceptTitle
              }
            }
          }
        }
      }
    }
  }
}
"""

def get_skills_recommendations(user_skills_df, results_df, difficulty_filter=None, program_type_filter=None, duration_filter=None, program_keys_filter=None):
    """
    Get recommendations from Skills API for learners based on their net skills (0 or less).
    Uses parallel processing to speed up API calls.
    
    Args:
        user_skills_df: DataFrame with user skills data
        results_df: DataFrame with assessment results
        difficulty_filter: List of difficulty levels to include (e.g., ['Beginner', 'Intermediate'])
        program_type_filter: List of program types to include (e.g., ['Course', 'Part'])
        duration_filter: List of duration ranges to include (e.g., ['Hours', 'Days'])
        program_keys_filter: String of comma-separated program keys or None
    """
    # Skills API endpoint
    skills_api_url = "https://skills.udacity.com/api/skills/search"
    
    headers = {
        'content-type': 'application/json',
        'Authorization': f'Bearer {settings.UDACITY_JWT}',
        'Accept': 'application/json'
    }
    
    # Get program keys from duration filter if specified
    duration_program_keys = []
    if duration_filter:
        duration_program_keys = get_programs_by_duration(duration_filter)
    
    # Build program keys filter once
    program_keys_for_filter = []
    if program_keys_filter:
        specific_keys = [key.strip() for key in program_keys_filter.split(',') if key.strip()]
        program_keys_for_filter.extend(specific_keys)
    if duration_program_keys:
        program_keys_for_filter.extend(duration_program_keys)
    
    # Prepare user tasks for parallel processing
    user_tasks = []
    for _, row in user_skills_df.iterrows():
        user_id = row['userId']
        net_skills = row['netSkills']
        skills_needing_improvement = [skill for skill, value in net_skills.items() if value <= 0]
        
        if skills_needing_improvement:  # Only process users with skills needing improvement
            user_tasks.append({
                'user_id': user_id,
                'total_score': row['totalScore'],
                'skills_needing_improvement': skills_needing_improvement
            })
    
    if not user_tasks:
        return pd.DataFrame()
    
    # Create progress tracking
    total_users = len(user_tasks)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Track API call statistics
    successful_calls = 0
    failed_calls = 0
    total_recommendations = 0
    
    def process_user_recommendations(user_task):
        """Process recommendations for a single user"""
        user_id = user_task['user_id']
        user_total_score = user_task['total_score']
        skills_needing_improvement = user_task['skills_needing_improvement']
        
        # Build Skills API payload
        payload = {
            'search': skills_needing_improvement,
            'searchField': "knowledge_component_names",
            'filter': {}
        }
        
        # Add filters
        if difficulty_filter:
            payload["filter"]["Difficulty"] = {"$in": difficulty_filter}
        if program_type_filter:
            payload["filter"]["parent_type"] = {"$in": program_type_filter}
        if program_keys_for_filter:
            payload["filter"]["parent_key"] = {"$in": program_keys_for_filter}
        
        user_recommendations = []
        
        try:
            response = requests.post(settings.SKILLS_API_URL, headers=headers, data=json.dumps(payload))
            
            if response.status_code == 200:
                try:
                    api_response = response.json()
                    
                    # Extract recommendations
                    for item in api_response:
                        try:
                            lesson_content = item['lesson']['content']
                            metadata = item['search']['metadata']
                            recommendation = {
                                  'userId': user_id,
                                  'totalScore': user_total_score,
                                  'weakSkills': skills_needing_improvement,
                                  'parentKey': metadata.get('parent_key', ''),
                                  'parentTitle': metadata.get('parent_title', ''),
                                  'lessonTitle': metadata.get('lesson_title', ''),
                                  'lessonId': lesson_content.get('id', ''),
                                  'content': lesson_content.get('summary', '')
                                    }  
                            user_recommendations.append(recommendation)
                          
                        except Exception as e:
                            print(e)
                            continue
                    
                    return {'status': 'success', 'recommendations': user_recommendations, 'user_id': user_id}
                    
                except json.JSONDecodeError:
                    return {'status': 'failed', 'recommendations': [], 'user_id': user_id}
            else:
                return {'status': 'failed', 'recommendations': [], 'user_id': user_id}
                
        except Exception:
            return {'status': 'failed', 'recommendations': [], 'user_id': user_id}
    
    # Process users in parallel
    recommendations_data = []
    completed_users = 0
    
    # Use ThreadPoolExecutor for parallel API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_user = {executor.submit(process_user_recommendations, user_task): user_task['user_id'] 
                         for user_task in user_tasks}
        
        # Process completed futures as they finish
        for future in concurrent.futures.as_completed(future_to_user):
            user_id = future_to_user[future]
            completed_users += 1
            
            # Update progress
            progress = completed_users / total_users
            progress_bar.progress(progress)
            status_text.text(f"Processing user {completed_users} of {total_users}: {user_id}")
            
            try:
                result = future.result()
                if result['status'] == 'success':
                    successful_calls += 1
                    recommendations_data.extend(result['recommendations'])
                    total_recommendations += len(result['recommendations'])
                else:
                    failed_calls += 1
            except Exception:
                failed_calls += 1
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(recommendations_data)

def get_programs_by_duration(duration_filter):
    """
    Get program keys from the catalog API based on duration filter.
    
    Args:
        duration_filter: List of duration ranges (e.g., ['Hours', 'Days'])
    
    Returns:
        List of program keys matching the duration criteria
    """
    # Map duration labels to raw duration ranges (in minutes)
    duration_mapping = {
        'Minutes': ['0 to 60'],           # 0-60 minutes
        'Hours': ['61 to 1439'],         # 1 hour to 23 hours 59 minutes
        'Days': ['1440 to 10079'],       # 1 day to 6 days 23 hours
        'Weeks': ['10080 to 43199'],     # 1 week to 29 days 23 hours
        'Months': ['43200 to 999999']    # 30+ days (using large upper bound)
    }
    
    # Build raw duration ranges based on selected filters
    raw_durations = []
    for duration_label in duration_filter:
        if duration_label in duration_mapping:
            raw_durations.extend(duration_mapping[duration_label])
    
    if not raw_durations:
        return []
    
    # Query catalog API
    catalog_url = 'https://api.udacity.com/api/unified-catalog/search'
    catalog_payload = {
        'pageSize': 1000,
        'SortBy': 'avgRating',
        'rawDurations': raw_durations
    }
    
    try:
        response = requests.post(catalog_url, json=catalog_payload)
        if response.status_code == 200:
            catalog_data = response.json()
            catalog_results = [r for r in catalog_data['searchResult']['hits'] if r['is_offered_to_public']]
            
            # Extract program keys
            program_keys = []
            for result in catalog_results:
                if 'key' in result:
                    program_keys.append(result['key'])
            
            return program_keys
        else:
            return []
            
    except Exception:
        return []

def fetch_page(assessment_id, limit, page):
    """Fetch a single page of attempts from the Assessments API."""
    response = requests.post(
        settings.ASSESSMENTS_API_URL,
        headers=settings.production_headers(),
        json={"query": ATTEMPTS_QUERY, "variables": {"assessmentId": assessment_id, "limit": limit, "page": page}},
    )
    if response.status_code != 200:
        raise Exception(f"Query failed on page {page}: {response.status_code}, {response.text}")
    return response.json()["data"]["attempts"]


def fetch_attempts(assessment_id, limit=50, progress_bar=None, status_text=None):
    """Fetch all attempt pages in parallel using ThreadPoolExecutor.

    Page 1 is fetched first to discover totalPages. Remaining pages are
    dispatched in parallel and results are collected via as_completed so
    the optional Streamlit progress_bar and status_text placeholders can
    be updated in the main thread without any thread-safety issues.
    """
    # Page 1 reveals pagination metadata
    page1_data = fetch_page(assessment_id, limit, 1)
    total_pages = page1_data["totalPages"]
    total_count = page1_data["totalCount"]
    all_attempts = list(page1_data["attempts"])

    if progress_bar is not None:
        progress_bar.progress(1 / max(total_pages, 1))
    if status_text is not None:
        status_text.text(f"Fetched page 1 of {total_pages} — {total_count:,} total responses...")

    if total_pages <= 1:
        if progress_bar is not None:
            progress_bar.progress(1.0)
        if status_text is not None:
            status_text.text(f"Fetched {len(all_attempts):,} responses.")
        return all_attempts

    completed = 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_page = {
            executor.submit(fetch_page, assessment_id, limit, p): p
            for p in range(2, total_pages + 1)
        }

        for future in concurrent.futures.as_completed(future_to_page):
            page_data = future.result()
            all_attempts.extend(page_data["attempts"])
            completed += 1

            if progress_bar is not None:
                progress_bar.progress(completed / total_pages)
            if status_text is not None:
                status_text.text(f"Fetched page {completed} of {total_pages} ({len(all_attempts):,} responses so far)...")

    if status_text is not None:
        status_text.text(f"Done — {len(all_attempts):,} responses fetched.")

    return all_attempts

ASSESSMENT_TITLE_QUERY = """
query AssessmentTitle($id: ID!) {
  assessment(id: $id) {
    id
    title
  }
}
"""


def fetch_assessment_title(assessment_id: str) -> str:
    """Resolve assessment display title from Assessments API; empty string on failure."""
    if not assessment_id:
        return ""
    try:
        response = requests.post(
            settings.ASSESSMENTS_API_URL,
            headers=settings.production_headers(),
            json={"query": ASSESSMENT_TITLE_QUERY, "variables": {"id": str(assessment_id).strip()}},
            timeout=30,
        )
        if response.status_code != 200:
            return ""
        payload = response.json()
        if payload.get("errors"):
            return ""
        node = (payload.get("data") or {}).get("assessment")
        if not node:
            return ""
        return str(node.get("title") or node.get("name") or "").strip()
    except Exception:
        return ""


def fetch_assessment_titles_map(assessment_ids):
    """Parallel fetch of titles for unique string IDs."""
    ids = list(dict.fromkeys(str(x).strip() for x in assessment_ids if x and str(x).strip()))
    if not ids:
        return {}
    out = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        fut_map = {executor.submit(fetch_assessment_title, aid): aid for aid in ids}
        for fut in concurrent.futures.as_completed(fut_map):
            aid = fut_map[fut]
            try:
                out[aid] = fut.result() or ""
            except Exception:
                out[aid] = ""
    return out


STUDENTS_USER_API = "https://students.udacity.com/api/user/users/{user_id}?projection=full"


@lru_cache(maxsize=4096)
def fetch_user_email(user_id: str) -> str:
    """Resolve learner email via Students API; falls back to user_id on failure."""
    if not user_id:
        return ""
    try:
        url = STUDENTS_USER_API.format(user_id=user_id)
        response = requests.get(url, headers=settings.production_headers(), timeout=30)
        if response.status_code != 200:
            return user_id
        payload = response.json()
        email = payload.get("email")
        return email if email else user_id
    except Exception:
        return user_id


def fetch_emails_for_user_ids(user_ids):
    """Parallel fetch for unique user IDs; lru_cache avoids duplicate HTTP for repeated IDs."""
    unique = list(dict.fromkeys(uid for uid in user_ids if uid))
    if not unique:
        return {}
    out = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_uid = {executor.submit(fetch_user_email, uid): uid for uid in unique}
        for future in concurrent.futures.as_completed(future_to_uid):
            uid = future_to_uid[future]
            try:
                out[uid] = future.result()
            except Exception:
                out[uid] = uid
    return out


def flatten_attempt(attempt):
    flat = {
        "id": attempt.get("id"),
        "userId": attempt.get("userId"),
        "createdAt": attempt.get("createdAt"),
        "status": attempt.get("status"),
        "result": attempt.get("result"),
        "totalScore": None,
    }

    report = attempt.get("report")
    if not report:
        return [flat]

    flat["totalScore"] = report.get("totalScore")

    section_reports = report.get("sectionReports") or []
    questions = []

    for section in section_reports:
        section_id = section.get("sectionId")
        section_score = section.get("sectionScore")
        section_title = section.get("section", {}).get("title") if section.get("section") else None
        question_reports = section.get("questionReports") or []

        for qr in question_reports:
            question = qr.get("question") or {}
            source = question.get("source") or {}

            q = {
                **flat,
                "sectionId": section_id,
                "sectionTitle": section_title,
                "sectionScore": section_score,
                "questionId": qr.get("questionId"),
                "questionScore": qr.get("questionScore"),
                "difficultyLevelId": question.get("difficultyLevelId"),
                "skillId": question.get("skillId"),
                "questionContent": question.get("content"),
                "questionChoices": json.dumps(question.get("choices", []), ensure_ascii=False),
                "uri": source.get("uri"),
                "conceptTitle": source.get("conceptTitle")
            }
            questions.append(q)

    return questions or [flat]

def get_attempts_dataframe(assessment_id, progress_bar=None, status_text=None):
    raw_attempts = fetch_attempts(assessment_id, progress_bar=progress_bar, status_text=status_text)
    flattened = []
    for attempt in raw_attempts:
        try:
            flattened.extend(flatten_attempt(attempt))
        except Exception:
            pass
    return pd.DataFrame(flattened)

def get_difficulty_levels():
    
    # Fetch difficulty.
    DIFFICULTY_LEVELS_QUERY = """
    query {
      difficultyLevels {
        id
        label
        labelValue
        externalId
      }
    }
    """

    response = requests.post(
                settings.ASSESSMENTS_API_URL,
                headers=settings.production_headers(),
                json={"query": DIFFICULTY_LEVELS_QUERY}
            )

    difficulty_levels = response.json()['data']['difficultyLevels']

    df_difficulty = pd.DataFrame(difficulty_levels)
    difficulty_map = df_difficulty.set_index('id')[['label','labelValue', 'externalId']]
    difficulty_map.columns = ['difficultyLabel','difficultyLabelValue','difficultyExternalId']
    return difficulty_map

def get_skills():

    # Fetch skills.
    SKILLS_QUERY = """
    query {
      skills {
        id
        title
        category
        externalId
        status
      }
    }
    """

    response = requests.post(
                settings.ASSESSMENTS_API_URL,
                headers=settings.production_headers(),
                json={"query": SKILLS_QUERY}
            )

    skills = response.json()['data']['skills']

    df_skills = pd.DataFrame(skills)
    skills_map = df_skills.set_index('id')[['title', 'category', 'externalId']]
    skills_map.columns = ['skillsTitle','skillsCategory','skillsExternalId']
    return skills_map

def _process_page_attempts(attempts):
    """Flatten a list of raw attempt objects and immediately split off text blobs.

    Returns:
        slim_rows     – list of row dicts with questionContent / questionChoices removed
        question_details – dict keyed by questionId with the text fields we stripped
    """
    slim_rows = []
    question_details = {}
    for attempt in attempts:
        try:
            rows = flatten_attempt(attempt)
            for row in rows:
                qid = row.get('questionId')
                if qid is not None and qid not in question_details:
                    question_details[qid] = {
                        'questionContent': row.pop('questionContent', ''),
                        'questionChoices': row.pop('questionChoices', '[]'),
                        'conceptTitle': row.get('conceptTitle', ''),
                    }
                else:
                    row.pop('questionContent', None)
                    row.pop('questionChoices', None)
                slim_rows.append(row)
        except Exception:
            pass
    return slim_rows, question_details


def fetch_and_build_dataframes(assessment_id, limit=50, progress_bar=None, status_text=None):
    """Streaming fetch-and-process pipeline.

    Each page is processed immediately as it arrives from the executor:
    attempts are flattened, text blobs extracted into a question-details
    dict, and the blob columns dropped before the rows are accumulated.
    This keeps peak memory proportional to (slim rows) + (unique questions)
    rather than (all raw JSON) + (all flattened rows with blobs).

    Returns:
        slim_df           – DataFrame without questionContent / questionChoices,
                            with score columns downcasted to float32
        question_details_df – small DataFrame indexed by questionId containing
                              the text fields that were stripped
    """
    def fetch_and_process(page):
        page_data = fetch_page(assessment_id, limit, page)
        slim_rows, qdetails = _process_page_attempts(page_data["attempts"])
        return slim_rows, qdetails, len(page_data["attempts"])

    # Page 1 reveals pagination metadata; process it immediately
    page1_data = fetch_page(assessment_id, limit, 1)
    total_pages = page1_data["totalPages"]
    total_count = page1_data["totalCount"]
    all_slim_rows, all_qdetails = _process_page_attempts(page1_data["attempts"])
    attempts_fetched = len(page1_data["attempts"])
    del page1_data  # release raw JSON

    if progress_bar is not None:
        progress_bar.progress(1 / max(total_pages, 1))
    if status_text is not None:
        status_text.text(f"Fetched page 1 of {total_pages} — {total_count:,} total attempts...")

    if total_pages > 1:
        completed = 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_page = {
                executor.submit(fetch_and_process, p): p
                for p in range(2, total_pages + 1)
            }
            for future in concurrent.futures.as_completed(future_to_page):
                slim_rows, qdetails, page_attempt_count = future.result()
                all_slim_rows.extend(slim_rows)
                attempts_fetched += page_attempt_count
                # First-seen wins for question details (avoids overwriting with duplicates)
                for qid, detail in qdetails.items():
                    if qid not in all_qdetails:
                        all_qdetails[qid] = detail
                completed += 1
                if progress_bar is not None:
                    progress_bar.progress(completed / total_pages)
                if status_text is not None:
                    status_text.text(
                        f"Fetched page {completed} of {total_pages} "
                        f"({attempts_fetched:,} attempts so far)..."
                    )

    if status_text is not None:
        status_text.text(f"Done — {attempts_fetched:,} attempts fetched.")

    slim_df = pd.DataFrame(all_slim_rows)
    question_details_df = pd.DataFrame.from_dict(all_qdetails, orient='index')
    question_details_df.index.name = 'questionId'

    # Downcast score columns that exist at this stage
    for col in ['questionScore', 'totalScore', 'sectionScore']:
        if col in slim_df.columns:
            slim_df[col] = pd.to_numeric(slim_df[col], errors='coerce').astype('float32')

    return slim_df, question_details_df


def get_results(assessment_id, progress_bar=None, status_text=None):
    slim_df, question_details_df = fetch_and_build_dataframes(
        assessment_id, progress_bar=progress_bar, status_text=status_text
    )
    difficulty_map = get_difficulty_levels()
    skills_map = get_skills()
    slim_df = slim_df.merge(difficulty_map, how='left', left_on='difficultyLevelId', right_index=True)
    slim_df = slim_df.merge(skills_map, how='left', left_on='skillId', right_index=True)
    slim_df['assessmentId'] = str(assessment_id).strip()
    # Downcast columns that only exist after the merge
    for col in ['sectionTitle', 'difficultyLabel', 'skillsTitle']:
        if col in slim_df.columns:
            slim_df[col] = slim_df[col].astype('category')
    return slim_df, question_details_df

def _enrich_attempts_df(df, difficulty_map, skills_map, assessment_id):
    """Merge difficulty/skills maps and tag rows with assessment_id."""
    if df is None or df.empty:
        return None
    out = df.merge(difficulty_map, how='left', left_on='difficultyLevelId', right_index=True)
    out = out.merge(skills_map, how='left', left_on='skillId', right_index=True)
    out['assessmentId'] = assessment_id
    return out

def get_results_multi(assessment_ids, max_workers=8, progress_bar=None):
    """
    Fetch attempts for many assessments, merge catalog maps once, concatenate with assessmentId.
    Failed IDs are skipped (returns partial data).
    """
    ids = [str(x).strip() for x in assessment_ids if str(x).strip()]
    if not ids:
        return pd.DataFrame()

    difficulty_map = get_difficulty_levels()
    skills_map = get_skills()

    def load_one(aid):
        try:
            raw = get_attempts_dataframe(aid)
            return _enrich_attempts_df(raw, difficulty_map, skills_map, aid)
        except Exception:
            return None

    frames = []
    total = len(ids)
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(load_one, aid): aid for aid in ids}
        for fut in concurrent.futures.as_completed(future_map):
            got = fut.result()
            if got is not None and not got.empty:
                frames.append(got)
            done += 1
            if progress_bar is not None:
                progress_bar.progress(min(done / total, 1.0))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def user_skills(df):
    # Get a list of unique users.
    user_list = df['userId'].unique()

    # Store results.
    results = []

    for user in user_list:
        # Filter for that user's results.
        user_df = df[df['userId'] == user]

        # Store strong / weak skills.
        strong_skills = {}
        weak_skills = {}

        for i, row in user_df.iterrows():
            skill = row['skillsTitle']
            if row['questionScore'] < 1:
                weak_skills[skill] = weak_skills.get(skill, 0) + 1
            elif row['questionScore'] == 1:
                strong_skills[skill] = strong_skills.get(skill, 0) + 1

        # Compute net skills: strong count - weak count for each skill
        all_skills = set(strong_skills.keys()).union(weak_skills.keys())
        net_skills = {
            skill: strong_skills.get(skill, 0) - weak_skills.get(skill, 0)
            for skill in all_skills
        }

        record = {
            'userId': user,
            'totalScore': user_df['totalScore'].iloc[0],
            'status': user_df['status'].iloc[0],
            'result': user_df['result'].iloc[0],
            'strongSkills': strong_skills,
            'weakSkills': weak_skills,
            'netSkills': net_skills
        }

        results.append(record)

    user_skills_df = pd.DataFrame(results)
    return user_skills_df 

def plot_net_skills_heatmap(user_skills_df: pd.DataFrame, results_df):
    """
    Creates a heatmap of percentage scores using Plotly, with:
    - y-axis labels showing % strong (based on value aggregation)
    - y-axis sorted by percentage of strong responses
    - Perfect alignment between bar chart and heatmap
    """

    # Build the skill-user matrix from percentage values
    skill_matrix = {}
    for _, row in user_skills_df.iterrows():
        user = row['userId']
        user_strong = row['strongSkills']
        user_weak = row['weakSkills']
        
        for skill in set(user_strong.keys()) | set(user_weak.keys()):
            strong = user_strong.get(skill, 0)
            weak = user_weak.get(skill, 0)
            total = strong + weak
            percent = (strong / total * 100) if total > 0 else 0
            skill_matrix.setdefault(skill, {})[user] = percent

    heatmap_df = pd.DataFrame.from_dict(skill_matrix, orient='index').fillna(0)

    # --- Aggregate strong and weak VALUES across all users ---
    strong_values = {}
    weak_values = {}

    for _, row in user_skills_df.iterrows():
        for skill, val in row['strongSkills'].items():
            strong_values[skill] = strong_values.get(skill, 0) + val
        for skill, val in row['weakSkills'].items():
            weak_values[skill] = weak_values.get(skill, 0) + val

    # Compute % strong and net total per skill
    all_skills = set(strong_values.keys()) | set(weak_values.keys())
    percent_labels = {}
    net_strength = {}
    percent_values = {}

    for skill in all_skills:
        strong = strong_values.get(skill, 0)
        weak = weak_values.get(skill, 0)
        total = strong + weak
        percent = (strong / total * 100) if total > 0 else 0
        percent_labels[skill] = skill  # Use just the skill name
        net_strength[skill] = strong - weak
        percent_values[skill] = percent

    # --- Sort skills by percentage values (descending) ---
    sorted_skills = sorted(percent_values.keys(), key=lambda s: percent_values[s], reverse=True)

    # Reorder heatmap rows and assign formatted labels
    heatmap_df = heatmap_df.reindex(sorted_skills)
    heatmap_df.index = [percent_labels[skill] for skill in sorted_skills]

    # Sort user IDs by total skill strength
    user_totals = heatmap_df.sum(axis=0)
    sorted_users = user_totals.sort_values(ascending=False).index
    heatmap_df = heatmap_df[sorted_users]

    # Extract percentage values and count unique questions
    skill_percentages = []
    skill_names = []
    skill_question_counts = []
    skill_attempt_labels = []
    
    for skill_name in heatmap_df.index:
        # Get percentage value for this skill
        percent_value = percent_values[skill_name]
        
        # Get strong and weak counts for this skill
        strong_count = strong_values.get(skill_name, 0)
        weak_count = weak_values.get(skill_name, 0)
        total_count = strong_count + weak_count
        
        # Count unique questions for this skill
        skill_questions = results_df[results_df['skillsTitle'] == skill_name]['questionId'].nunique()
        
        skill_percentages.append(percent_value)
        skill_names.append(skill_name)
        skill_question_counts.append(skill_questions)
        skill_attempt_labels.append(f"{skill_name} ({strong_count}/{total_count})")
    
    # Reverse the lists to match the descending order of the heatmap
    skill_percentages = skill_percentages[::-1]
    skill_names = skill_names[::-1]
    skill_question_counts = skill_question_counts[::-1]
    skill_attempt_labels = skill_attempt_labels[::-1]
    
    # Create simple labels with just skill names
    skill_labels = skill_attempt_labels
    
    # Reverse the heatmap data before setting the index to get the correct order
    heatmap_df = heatmap_df.iloc[::-1]
    
    # Update heatmap index to use the same attempt count labels
    heatmap_df.index = skill_attempt_labels
    
    # Calculate mean percentage for reference line
    mean_percentage = np.mean(skill_percentages)
    
    # Create Plotly subplot with shared y-axis
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.3, 0.7],
        subplot_titles=('Skill Performance Summary', 'Skills Performance Heatmap'),
        specs=[[{"type": "bar"}, {"type": "heatmap"}]],
        shared_yaxes=True,
        horizontal_spacing=0.02  # Reduce spacing between subplots
    )
    
    # Add horizontal bar chart (left)
    fig.add_trace(
        go.Bar(
            y=skill_labels,
            x=skill_percentages,
            orientation='h',
            marker=dict(
                color=skill_percentages,
                colorscale='RdYlGn',
                cmin=0,
                cmax=100,
                showscale=False
            ),
            text=[f'{p:.1f}%' for p in skill_percentages],
            textposition='auto',
            name='Success Rate'
        ),
        row=1, col=1
    )
    
    # Add heatmap (right)
    fig.add_trace(
        go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns,
            y=heatmap_df.index,
            colorscale='RdYlGn',
            zmin=0,
            zmax=100,
            showscale=True,
            colorbar=dict(
                title=dict(text="% Correct Responses"),
                tickmode="array",
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["0%", "25%", "50%", "75%", "100%"]
            )
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Skill Coverage",
        height=600,
        showlegend=False,
        xaxis_title="Overall Success Rate (%)",
        xaxis2_title="Users (Ranked by Total Percentage)",
        yaxis_title="",  # Remove y-axis label
        yaxis2_title="",  # Remove y-axis label
        margin=dict(l=50, r=50, t=80, b=50),  # Reduce margins for tighter layout
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        xaxis=dict(title_font=dict(color='black')),
        xaxis2=dict(title_font=dict(color='black'))
    )
    
    # Update axes
    fig.update_xaxes(range=[0, 100], row=1, col=1, showticklabels=False, color='black')  # Hide x-axis ticks on bar chart
    fig.update_xaxes(showticklabels=False, row=1, col=2, color='black', tickfont=dict(color='black'))  # Hide user IDs
    fig.update_yaxes(showticklabels=True, row=1, col=1, color='black', tickfont=dict(color='black'))
    fig.update_yaxes(showticklabels=False, row=1, col=2, color='black')  # Hide skill labels on heatmap
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

def plot_skill_correlation_heatmap(user_skills_df: pd.DataFrame):
    """
    Creates a correlation heatmap showing which skills tend to be performed similarly by users.
    """
    # Create user-skill performance matrix
    skill_matrix = {}
    for _, row in user_skills_df.iterrows():
        user = row['userId']
        user_strong = row['strongSkills']
        user_weak = row['weakSkills']
        
        for skill in set(user_strong.keys()) | set(user_weak.keys()):
            strong = user_strong.get(skill, 0)
            weak = user_weak.get(skill, 0)
            total = strong + weak
            percent = (strong / total * 100) if total > 0 else 0
            skill_matrix.setdefault(skill, {})[user] = percent

    # Convert to DataFrame and fill missing values
    skill_df = pd.DataFrame.from_dict(skill_matrix, orient='index').fillna(0)
    
    if len(skill_df) < 2:
        return
    
    # Calculate correlation matrix
    correlation_matrix = skill_df.T.corr()
    
    # Create upper triangle mask to show only one half
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Find top 5 and bottom 5 correlations for highlighting
    correlations_list = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.index)):  # Only upper triangle
            corr_value = correlation_matrix.iloc[j, i]
            if not np.isnan(corr_value):
                correlations_list.append({
                    'i': i, 'j': j,
                    'skill1': correlation_matrix.columns[i],
                    'skill2': correlation_matrix.index[j],
                    'correlation': corr_value
                })
    
    # Sort by correlation value and get top 5 and bottom 5
    correlations_list.sort(key=lambda x: x['correlation'], reverse=True)
    top_5_indices = [(c['i'], c['j']) for c in correlations_list[:5]]
    bottom_5_indices = [(c['i'], c['j']) for c in correlations_list[-5:]]
    
    st.subheader("Skill Correlation Analysis")
    
    # Create correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    
    # Create heatmap with RdYlBu colormap (red for negative, blue for positive correlations)
    im = ax.imshow(correlation_matrix, cmap='RdYlBu', vmin=-1, vmax=1, aspect='auto', alpha=0.8)
    
    # Apply mask to hide lower triangle
    masked_data = np.ma.masked_where(mask, correlation_matrix)
    im.set_array(masked_data)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient')
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.set_ticklabels(['-1.0', '-0.5', '0.0', '0.5', '1.0'])
    
    # Set up axes
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_yticks(range(len(correlation_matrix.index)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(correlation_matrix.index)
    
    # Add correlation values as text annotations and highlight top 5
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.index)):
            corr_value = correlation_matrix.iloc[j, i]
            if not np.isnan(corr_value) and not mask[j, i]:  # Only show upper triangle
                # Check if this is one of the top 5 correlations
                is_top_5 = (i, j) in top_5_indices
                is_bottom_5 = (i, j) in bottom_5_indices
                
                # Choose text color based on background and highlighting
                if is_top_5:
                    text_color = 'black'  # Light green background will be added
                    fontweight = 'bold'
                    fontsize = 10
                    bg_color = 'lightgreen'
                elif is_bottom_5:
                    text_color = 'black'  # Light red background will be added
                    fontweight = 'bold'
                    fontsize = 10
                    bg_color = 'lightcoral'
                elif abs(corr_value) > 0.5:
                    text_color = 'white'
                    fontweight = 'normal'
                    fontsize = 8
                    bg_color = None
                else:
                    text_color = 'black'
                    fontweight = 'normal'
                    fontsize = 8
                    bg_color = None
                
                # Add highlight for top 5 and bottom 5 correlations
                if is_top_5 or is_bottom_5:
                    rect = plt.Rectangle((i-0.4, j-0.4), 0.8, 0.8, 
                                       facecolor=bg_color, edgecolor=bg_color, alpha=0.7, zorder=1)
                    ax.add_patch(rect)
                
                ax.text(i, j, f'{corr_value:.2f}', ha='center', va='center', 
                       fontsize=fontsize, fontweight=fontweight, color=text_color, zorder=2)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_title('Skill Performance Correlations')
    
    plt.tight_layout()
    st.pyplot(fig)

def calculate_question_statistics(results_df, question_details=None):
    """
    Calculate difficulty and discrimination indices for each question using classical test theory.
    
    Difficulty = proportion of students who answered correctly
    Discrimination = correlation between question score and total score

    question_details: optional DataFrame indexed by questionId with columns
        questionContent, questionChoices, conceptTitle — used as a fallback
        when those columns have been dropped from results_df by slim_results_df().
    """
    question_stats = {}
    has_assessment = (
        'assessmentId' in results_df.columns
        and results_df['assessmentId'].notna().any()
    )

    def _qdetail(question_data, col, default):
        """Read a text field from the row data or from question_details lookup."""
        if col in question_data.columns:
            return question_data[col].iloc[0]
        if question_details is not None and not question_details.empty:
            qid = question_data['questionId'].iloc[0] if 'questionId' in question_data.columns else None
            if qid is not None and qid in question_details.index and col in question_details.columns:
                return question_details.loc[qid, col]
        return default

    def stats_for_slice(question_data, dict_key, assessment_id_val=None, question_id_val=None):
        if len(question_data) == 0:
            return
        difficulty = question_data['questionScore'].mean()
        if len(question_data) > 1 and question_data['questionScore'].std() > 0 and question_data['totalScore'].std() > 0:
            discrimination = question_data['questionScore'].corr(question_data['totalScore'])
        else:
            discrimination = 0.0
        skill_title = question_data['skillsTitle'].iloc[0] if 'skillsTitle' in question_data.columns else 'Unknown'
        difficulty_level = question_data['difficultyLabel'].iloc[0] if 'difficultyLabel' in question_data.columns else 'Unknown'
        concept_title = _qdetail(question_data, 'conceptTitle', 'Unknown')
        question_content = _qdetail(question_data, 'questionContent', 'No content available')
        question_choices = _qdetail(question_data, 'questionChoices', '[]')
        qid = question_id_val if question_id_val is not None else dict_key
        row = {
            'question_id': qid,
            'difficulty': difficulty,
            'discrimination': discrimination,
            'skill_title': skill_title,
            'difficulty_level': difficulty_level,
            'concept_title': concept_title,
            'question_content': question_content,
            'question_choices': question_choices,
            'n_attempts': len(question_data),
        }
        if assessment_id_val is not None:
            row['assessment_id'] = assessment_id_val
        question_stats[dict_key] = row

    if has_assessment:
        grouped = results_df.dropna(subset=['questionId']).groupby(
            ['assessmentId', 'questionId'], sort=False
        )
        for (aid, qid), question_data in grouped:
            stats_for_slice(question_data, f"{aid}::{qid}", assessment_id_val=aid, question_id_val=qid)
    else:
        for question_id in results_df['questionId'].dropna().unique():
            question_data = results_df[results_df['questionId'] == question_id]
            stats_for_slice(question_data, question_id)

    return question_stats

def assessment_level_summary_table(results_df, min_attempts_per_question=3):
    """
    Roll up question-level statistics (same eligibility filters as plot_question_analysis)
    to one row per assessment: weighted avg success rate and discrimination, question count,
    and distinct attempt count in the sample.
    """
    if results_df is None or results_df.empty:
        return pd.DataFrame()

    if 'assessmentId' not in results_df.columns:
        return pd.DataFrame()

    question_stats = calculate_question_statistics(results_df)
    stats_df = (
        pd.DataFrame.from_dict(question_stats, orient='index').reset_index(drop=True)
        if question_stats
        else pd.DataFrame()
    )
    if not stats_df.empty:
        stats_df = stats_df[stats_df['n_attempts'] >= min_attempts_per_question]
        stats_df = stats_df[stats_df['discrimination'].notna()]

    rows = []
    for aid in sorted(results_df['assessmentId'].dropna().astype(str).unique()):
        sub = results_df[results_df['assessmentId'].astype(str) == aid]
        n_attempts_distinct = int(sub.dropna(subset=['id'])['id'].nunique())

        if stats_df.empty or 'assessment_id' not in stats_df.columns:
            g = pd.DataFrame()
        else:
            g = stats_df[stats_df['assessment_id'].astype(str) == aid]

        if len(g) == 0:
            rows.append(
                {
                    'assessment_id': aid,
                    'questions_in_summary': 0,
                    'avg_success_rate': np.nan,
                    'avg_discrimination': np.nan,
                    'n_attempts': n_attempts_distinct,
                }
            )
            continue

        w = g['n_attempts'].sum()
        avg_sr = float((g['difficulty'] * g['n_attempts']).sum() / w) if w else np.nan
        avg_disc = float((g['discrimination'] * g['n_attempts']).sum() / w) if w else np.nan
        rows.append(
            {
                'assessment_id': aid,
                'questions_in_summary': int(len(g)),
                'avg_success_rate': avg_sr,
                'avg_discrimination': avg_disc,
                'n_attempts': n_attempts_distinct,
            }
        )

    return pd.DataFrame(rows)

def plot_assessment_level_summary_scatter(summary_df):
    """
    Scatter of weighted avg discrimination (x) vs success rate (y) per assessment row.
    Expects columns: avg_discrimination, avg_success_rate; optional assessment_title for labels.
    """
    if summary_df is None or summary_df.empty:
        return
    req = {"avg_discrimination", "avg_success_rate"}
    if not req.issubset(summary_df.columns):
        return
    plot_df = summary_df.dropna(subset=["avg_success_rate", "avg_discrimination"]).copy()
    if plot_df.empty:
        st.caption("No assessments with both average success rate and discrimination to chart.")
        return

    fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
    ax.scatter(
        plot_df["avg_discrimination"],
        plot_df["avg_success_rate"],
        s=80,
        c="black",
        alpha=0.85,
        edgecolors="white",
        linewidth=0.6,
        zorder=3,
    )

    x = np.linspace(-1, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    difficulty_quality = np.clip(1 - 4 * (Y - 0.5) ** 2, 0, 1)
    discrimination_quality = np.clip(X, 0, 1)
    combined_quality = difficulty_quality * discrimination_quality
    im = ax.imshow(
        combined_quality,
        extent=[-1, 1, 0, 1],
        origin="lower",
        cmap="RdYlGn",
        alpha=0.3,
        aspect="auto",
        zorder=1,
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Assessment quality score (heuristic)")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["Poor", "Fair", "Good", "Very Good", "Excellent"])

    ax.set_xlabel("Avg discrimination (weighted by responses)")
    ax.set_ylabel("Avg success rate (weighted by responses)")
    ax.set_title("Assessment-level averages: success rate vs discrimination")
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    label_col = "assessment_title" if "assessment_title" in plot_df.columns else None
    if label_col and len(plot_df) <= 30:
        for _, row in plot_df.iterrows():
            raw = str(row.get(label_col) or "").strip()
            aid = str(row.get("assessment_id", ""))[:8]
            label = (raw[:40] + "…") if len(raw) > 40 else (raw or aid)
            ax.annotate(
                label,
                (row["avg_discrimination"], row["avg_success_rate"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
                alpha=0.9,
                zorder=4,
            )

    st.pyplot(fig)

def plot_question_analysis(results_df, question_details=None):
    """
    Create a scatter plot of question difficulty vs discrimination using matplotlib/seaborn.
    """
    question_stats = calculate_question_statistics(results_df, question_details=question_details)
    
    if not question_stats:
        return
    
    stats_df = pd.DataFrame.from_dict(question_stats, orient='index').reset_index(drop=True)

    # Filter out questions with too few attempts or invalid statistics
    stats_df = stats_df[stats_df['n_attempts'] >= 3]  # Minimum 3 attempts
    stats_df = stats_df[stats_df['discrimination'].notna()]  # Valid discrimination values
    
    if len(stats_df) == 0:
        return
    
    st.subheader("Question Performance Analysis")
    
    # Add summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Questions", len(stats_df))
        st.metric("Avg Success Rate", f"{stats_df['difficulty'].mean():.3f}")
    
    with col2:
        st.metric("Avg Discrimination", f"{stats_df['discrimination'].mean():.3f}")
        st.metric("High Discrimination (>0.3)", len(stats_df[stats_df['discrimination'] > 0.3]))
    
    with col3:
        st.metric("Optimal Success Rate (0.3-0.7)", len(stats_df[(stats_df['difficulty'] >= 0.3) & (stats_df['difficulty'] <= 0.7)]))
        st.metric("Low Discrimination (<0.1)", len(stats_df[stats_df['discrimination'] < 0.1]))
    
    # Create the scatter plot using matplotlib/seaborn
    fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
    
    # Create scatter plot with size based on number of attempts
    scatter = ax.scatter(
        stats_df['discrimination'], 
        stats_df['difficulty'],
        s=50,  # Standard size for all points
        c='black',  # Black points for better visibility
        alpha=0.8,
        edgecolors='white',
        linewidth=0.5
    )
    
    # Add colored areas for question categories instead of reference lines
    # Create gradient overlays for question quality
    # Difficulty gradient (vertical)
    x = np.linspace(-1, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    # Difficulty quality score (0-1): optimal around 0.5, worse at extremes
    # Note: difficulty is actually success rate, so higher is better, but we still want optimal around 0.5
    difficulty_quality = 1 - 4 * (Y - 0.5)**2  # Parabolic function, max at 0.5
    difficulty_quality = np.clip(difficulty_quality, 0, 1)
    
    # Discrimination quality score (0-1): better with higher discrimination
    discrimination_quality = np.clip(X, 0, 1)  # Linear from 0 to 1
    
    # Combined quality score
    combined_quality = difficulty_quality * discrimination_quality
    
    # Create gradient overlay
    im = ax.imshow(combined_quality, extent=[-1, 1, 0, 1], origin='lower', 
                   cmap='RdYlGn', alpha=0.3, aspect='auto')
    
    # Add colorbar for quality reference
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Question Quality Score')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
    
    # Add labels and title
    ax.set_xlabel('Discrimination (Correlation with Total Score)')
    ax.set_ylabel('Success Rate (Proportion Correct)')
    ax.set_title('Question Success Rate vs Discrimination')
    
    # Set axis limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Add question details table (underlying data for the chart)
    table_df = stats_df.copy()
    if 'assessment_id' in table_df.columns:
        aids = table_df['assessment_id'].dropna().astype(str).unique().tolist()
        with st.spinner("Loading assessment titles..."):
            _title_map = fetch_assessment_titles_map(aids)
        table_df['assessment_title'] = table_df['assessment_id'].astype(str).map(
            lambda x: (_title_map.get(x) or '').strip() or '—'
        )

    table_cols = [
        'question_id',
        'difficulty',
        'discrimination',
        'skill_title',
        'difficulty_level',
        'n_attempts',
        'question_content',
        'question_choices',
    ]
    if 'assessment_id' in table_df.columns:
        table_cols = ['assessment_id', 'assessment_title'] + table_cols
    display_df = table_df[table_cols].copy()
    rename_map = {
        'assessment_id': 'Assessment ID',
        'assessment_title': 'Assessment title',
        'question_id': 'Question ID',
        'difficulty': 'Success Rate',
        'discrimination': 'Discrimination',
        'skill_title': 'Skill',
        'difficulty_level': 'Difficulty Level',
        'n_attempts': 'Attempts',
        'question_content': 'Question Content',
        'question_choices': 'Question Choices',
    }
    display_df = display_df.rename(columns=rename_map)
    display_df = display_df.sort_values('Discrimination', ascending=False)
    total_rows = len(display_df)
    preview_df = display_df.head(_TABLE_PREVIEW_ROWS)
    if total_rows > _TABLE_PREVIEW_ROWS:
        st.caption(f"Question performance data — showing {_TABLE_PREVIEW_ROWS} of {total_rows} rows")
    else:
        st.caption("Question performance data")
    st.dataframe(preview_df, use_container_width=True)
    if total_rows > _TABLE_PREVIEW_ROWS:
        st.download_button(
            label=f"Download full question data ({total_rows} rows)",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="question_performance.csv",
            mime="text/csv",
        )

def plot_total_score_histogram(results_df):
    """
    Creates a histogram of total scores, including all attempts made.
    """
    # Get all total scores (including multiple attempts per learner)
    all_scores = results_df['totalScore'].dropna()
    
    if len(all_scores) == 0:
        return
    
    st.subheader("Total Score Distribution")
    
    # Add summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Question Attempts", len(all_scores))
        st.metric("Mean Score", f"{all_scores.mean() * 100:.1f}%")
    
    with col2:
        st.metric("Median Score", f"{all_scores.median() * 100:.1f}%")
        st.metric("Standard Deviation", f"{all_scores.std() * 100:.1f}%")
    
    with col3:
        st.metric("Min Score", f"{all_scores.min() * 100:.1f}%")
        st.metric("Max Score", f"{all_scores.max() * 100:.1f}%")
    
    with col4:
        # Calculate percentiles
        p25 = all_scores.quantile(0.25)
        p75 = all_scores.quantile(0.75)
        st.metric("25th Percentile", f"{p25 * 100:.1f}%")
        st.metric("75th Percentile", f"{p75 * 100:.1f}%")
    
    # Create the histogram using matplotlib
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    
    # Add light grey overlay for 25th-75th percentile range
    p25 = all_scores.quantile(0.25)
    p75 = all_scores.quantile(0.75)
    ax.axvspan(p25, p75, alpha=.5, color='lightgrey', label='Middle 50 percent')
    
    # Create histogram with color gradient based on score values
    n, bins, patches = ax.hist(all_scores, bins=20, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Color the bars based on their position (score values)
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(0, 1)
    
    for i, (patch, bin_center) in enumerate(zip(patches, (bins[:-1] + bins[1:]) / 2)):
        patch.set_facecolor(cmap(norm(bin_center)))
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Score Quality')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
    
    # Update layout
    ax.set_xlabel('Total Score', color='black')
    ax.set_ylabel('Number of Attempts', color='black')
    ax.set_title('Distribution of Total Scores (All Attempts)', color='black')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set tick label colors to black
    ax.tick_params(axis='both', colors='black')
    
    # Set legend text color to black
    legend = ax.legend()
    plt.setp(legend.get_texts(), color='black')
    
    st.pyplot(fig)
    
    # Table: one row per attempt with attempt time, email, overall score
    attempt_df = results_df.dropna(subset=['id'])
    if attempt_df.empty:
        return
    agg_cols = {
        'createdAt': 'first',
        'userId': 'first',
        'totalScore': 'first',
    }
    if 'assessmentId' in attempt_df.columns:
        agg_cols['assessmentId'] = 'first'
    attempt_rows = attempt_df.groupby('id', as_index=False).agg(agg_cols)

    unique_aids = []
    if 'assessmentId' in attempt_df.columns:
        unique_aids = (
            attempt_df['assessmentId'].dropna().astype(str).unique().tolist()
        )

    # Sort first so the preview covers the most-recent attempts
    _ts_sort = pd.to_datetime(attempt_rows['createdAt'], utc=True, errors='coerce')
    attempt_rows = attempt_rows.assign(_ts_sort=_ts_sort).sort_values('_ts_sort', ascending=False, na_position='last').drop(columns=['_ts_sort'])

    # Fetch emails only for the preview slice to keep the UI fast
    preview_user_ids = attempt_rows.head(_TABLE_PREVIEW_ROWS)['userId'].tolist()
    with st.spinner("Loading learner emails..."):
        email_map = fetch_emails_for_user_ids(preview_user_ids)
        title_map = fetch_assessment_titles_map(unique_aids) if unique_aids else {}

    def _build_display_df(rows, include_email=True):
        out = rows.copy()
        if include_email:
            out['Email'] = out['userId'].map(lambda u: email_map.get(u, u))
        else:
            out['Email'] = out['userId']  # fall back to User ID in downloads
        _ts = pd.to_datetime(out['createdAt'], utc=True, errors='coerce')
        out['Attempt time'] = _ts.dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        out.loc[_ts.isna(), 'Attempt time'] = ''
        out['Overall Score'] = (out['totalScore'] * 100).round(1).astype(str) + '%'
        if 'assessmentId' in out.columns:
            out['Assessment ID'] = out['assessmentId'].astype(str)
            out['Assessment title'] = out['Assessment ID'].map(
                lambda x: (title_map.get(x) or '').strip() or '—'
            )
            cols = ['Assessment ID', 'Assessment title', 'Attempt time', 'Email', 'Overall Score']
        else:
            cols = ['Attempt time', 'Email', 'Overall Score']
        return out[cols]

    total_rows = len(attempt_rows)
    preview_display = _build_display_df(attempt_rows.head(_TABLE_PREVIEW_ROWS), include_email=True)
    if total_rows > _TABLE_PREVIEW_ROWS:
        st.caption(f"Total score distribution data (one row per attempt) — showing {_TABLE_PREVIEW_ROWS} of {total_rows} rows. Download includes all rows (User ID shown in place of email).")
    else:
        st.caption("Total score distribution data (one row per attempt)")
    st.dataframe(preview_display, use_container_width=True)
    if total_rows > _TABLE_PREVIEW_ROWS:
        full_display = _build_display_df(attempt_rows, include_email=False)
        st.download_button(
            label=f"Download full scores table ({total_rows} rows)",
            data=full_display.to_csv(index=False).encode("utf-8"),
            file_name="total_scores.csv",
            mime="text/csv",
        )

def plot_section_scores(results_df):
    """
    Creates analysis of section-level scores, ensuring one score per attempt per section.
    """
    # Get unique section scores per attempt per section
    section_scores = results_df.dropna(subset=['id', 'sectionId']).groupby(
        ['id', 'createdAt', 'userId', 'sectionId', 'sectionTitle']
    )['sectionScore'].first().reset_index()
    
    if len(section_scores) == 0:
        return
    
    st.subheader("Section Performance Analysis")
    
    # Calculate section-level statistics
    section_stats = section_scores.groupby(['sectionId', 'sectionTitle']).agg({
        'sectionScore': ['count', 'mean', 'std', 'min', 'max']
    }).round(3)
    
    # Flatten column names
    section_stats.columns = ['Learners', 'Mean Score', 'Std Dev', 'Min Score', 'Max Score']
    section_stats = section_stats.reset_index()
    
    # Convert scores to percentages
    section_stats['Mean Score %'] = (section_stats['Mean Score'] * 100).round(1)
    section_stats['Std Dev %'] = (section_stats['Std Dev'] * 100).round(1)
    section_stats['Min Score %'] = (section_stats['Min Score'] * 100).round(1)
    section_stats['Max Score %'] = (section_stats['Max Score'] * 100).round(1)
    
    # Calculate overall average for the violin plot
    avg_section_score = section_stats['Mean Score %'].mean()
    
    # Create section performance chart using matplotlib
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    
    # Prepare data for violin plot with color gradient
    section_data = []
    section_labels = []
    
    for i, section in section_stats.iterrows():
        section_id = section['sectionId']
        section_title = section['sectionTitle']
        
        # Get all scores for this section
        section_scores_data = section_scores[section_scores['sectionId'] == section_id]['sectionScore'] * 100
        
        section_data.append(section_scores_data.values)
        section_labels.append(section_title)
    
    # Create violin plot
    violin_parts = ax.violinplot(section_data, positions=range(1, len(section_data) + 1))
    
    # Store the violin paths before removing the default elements
    violin_paths = []
    for violin in violin_parts['bodies']:
        violin_paths.append(violin.get_paths()[0].vertices)
        violin.remove()
    
    # Get axis limits
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    
    # Create a numpy image to use as a gradient
    Nx, Ny = 1, 1000
    imgArr = np.tile(np.linspace(0, 1, Ny), (Nx, 1)).T
    cmap = 'RdYlGn'
    
    # Apply gradient to each violin plot
    for i, vertices in enumerate(violin_paths):
        # Create path from stored violin vertices
        path = Path(vertices)
        patch = PathPatch(path, facecolor='none', edgecolor='black')
        ax.add_patch(patch)
        
        # Create gradient image clipped to violin shape
        img = ax.imshow(imgArr, origin="lower", extent=[xmin, xmax, ymin, ymax], aspect="auto",
                       cmap=cmap, clip_path=patch, alpha=0.8)
    
    # Add box plots and mean lines
    for i, data in enumerate(section_data):
        position = i + 1
        
        # Calculate statistics
        mean_val = np.mean(data)
        p75 = np.percentile(data, 75)
        p25 = np.percentile(data, 25)
        median_val = np.median(data)
        
        # Create box plot elements
        # Box (25th to 75th percentile)
        box_height = p75 - p25
        box_bottom = p25
        
        # Draw the box (narrower to fit within violin)
        rect = plt.Rectangle((position - 0.15, box_bottom), 0.3, box_height, 
                           facecolor='white', edgecolor='black', linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        
        # Draw median line
        ax.plot([position - 0.15, position + 0.15], [median_val, median_val], 
               color='black', linewidth=2)
        
        # Draw whiskers (narrower to fit within violin)
        iqr = p75 - p25
        lower_whisker = max(np.min(data), p25 - .75 * iqr)
        upper_whisker = min(np.max(data), p75 + .75 * iqr)
        
        # Whisker lines
        ax.plot([position, position], [lower_whisker, p25], color='black', linewidth=1.5)
        ax.plot([position, position], [p75, upper_whisker], color='black', linewidth=1.5)
        
        # Whisker caps (narrower)
        ax.plot([position - 0.05, position + 0.05], [lower_whisker, lower_whisker], color='black', linewidth=1.5)
        ax.plot([position - 0.05, position + 0.05], [upper_whisker, upper_whisker], color='black', linewidth=1.5)
        
        # Add score annotations inside the box
        ax.annotate(f'{p75:.1f}%', (position, p75 - box_height*0.1), 
                   fontsize=7, color='black', ha='center', va='center')
        ax.annotate(f'{p25:.1f}%', (position, p25 + box_height*0.1), 
                   fontsize=7, color='black', ha='center', va='center')
    
    # Add colorbar for reference
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    import matplotlib 
    import matplotlib.colors as mcolors
    
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="5%", pad="2%")
    norm = mcolors.Normalize(vmin=0, vmax=100)
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=matplotlib.colormaps[cmap],
                                        norm=norm, orientation='vertical')
    cb.set_ticks([0, 25, 50, 75, 100])
    cb.set_ticklabels(['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
    
    # Update layout
    ax.set_ylabel('Section Score (%)')
    ax.set_title('Section Performance Distribution')
    ax.set_xticks(range(1, len(section_data) + 1))
    ax.set_xticklabels(section_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    st.pyplot(fig)
    
    # Table: one row per attempt, attempt time and email first, then one column per section
    per_attempt = section_scores.groupby('id', as_index=False).agg({
        'createdAt': 'first',
        'userId': 'first',
    })
    pivot_scores = section_scores.pivot_table(
        index='id', columns='sectionTitle', values='sectionScore', aggfunc='first'
    )
    pivot_scores = pivot_scores.reset_index()
    section_table = per_attempt.merge(pivot_scores, on='id', how='inner')
    _pts = pd.to_datetime(section_table['createdAt'], utc=True, errors='coerce')
    section_table = section_table.assign(_pts=_pts).sort_values('_pts', ascending=False, na_position='last').drop(columns=['_pts'])

    # Fetch emails only for the preview slice
    preview_user_ids = section_table.head(_TABLE_PREVIEW_ROWS)['userId'].tolist()
    with st.spinner("Loading learner emails..."):
        email_map = fetch_emails_for_user_ids(preview_user_ids)

    def _build_section_display(rows, include_email=True):
        out = rows.copy()
        _ts = pd.to_datetime(out['createdAt'], utc=True, errors='coerce')
        out['Attempt time'] = _ts.dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        out.loc[_ts.isna(), 'Attempt time'] = ''
        if include_email:
            out['Email'] = out['userId'].map(lambda u: email_map.get(u, u))
        else:
            out['Email'] = out['userId']
        out = out.drop(columns=['id', 'createdAt', 'userId'])
        for col in out.columns:
            if col in ('Attempt time', 'Email'):
                continue
            out[col] = (out[col] * 100).round(1)
        ordered = ['Attempt time', 'Email'] + [c for c in out.columns if c not in ('Attempt time', 'Email')]
        return out[ordered]

    total_rows = len(section_table)
    preview_display = _build_section_display(section_table.head(_TABLE_PREVIEW_ROWS), include_email=True)
    if total_rows > _TABLE_PREVIEW_ROWS:
        st.caption(f"Section performance data (one row per attempt, one column per section) — showing {_TABLE_PREVIEW_ROWS} of {total_rows} rows. Download includes all rows (User ID shown in place of email).")
    else:
        st.caption("Section performance data (one row per attempt, one column per section)")
    st.dataframe(preview_display, use_container_width=True)
    if total_rows > _TABLE_PREVIEW_ROWS:
        full_display = _build_section_display(section_table, include_email=False)
        st.download_button(
            label=f"Download full section data ({total_rows} rows)",
            data=full_display.to_csv(index=False).encode("utf-8"),
            file_name="section_scores.csv",
            mime="text/csv",
        )

def plot_recommendation_charts(recommendations_df):
    """
    Create charts for most common parent keys, lessons recommended, and program recommendations analysis.
    """
    if recommendations_df.empty:
        return
    
    # Most common parent keys (count most recommended program per learner)
    learner_program_counts = []
    for user_id in recommendations_df['userId'].unique():
        user_recommendations = recommendations_df[recommendations_df['userId'] == user_id]
        program_counts = user_recommendations['parentKey'].value_counts()
        if not program_counts.empty:
            top_program_key = program_counts.index[0]
            top_program_title = user_recommendations[user_recommendations['parentKey'] == top_program_key]['parentTitle'].iloc[0]
            learner_program_counts.append((top_program_key, top_program_title))
    
    parent_key_counts = pd.DataFrame(learner_program_counts, columns=['parentKey', 'parentTitle'])
    parent_key_counts = parent_key_counts.groupby(['parentKey', 'parentTitle']).size().reset_index(name='count')
    parent_key_counts = parent_key_counts.sort_values('count', ascending=False).head(10)
    
    # Most common lessons
    lesson_counts = recommendations_df.groupby(['lessonTitle', 'parentTitle']).size().reset_index(name='count')
    lesson_counts = lesson_counts.sort_values('count', ascending=False).head(10)
    
    # Create charts - now with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=200)
    
    # Parent keys chart - reverse order for descending display
    bars1 = ax1.barh(range(len(parent_key_counts)), parent_key_counts['count'], 
                     color='skyblue', edgecolor='black', alpha=0.8)
    ax1.set_yticks(range(len(parent_key_counts)))
    # Reverse the labels to show highest count at top
    ax1.set_yticklabels([f"{title[:30]}..." if len(title) > 30 else title 
                        for title in parent_key_counts['parentTitle'][::-1]])
    ax1.set_xlabel('Number of Learners')
    ax1.set_title('Most Recommended Programs (Top Program per Learner)')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', ha='left', va='center', fontweight='bold')
    
    # Lessons chart - reverse order for descending display
    bars2 = ax2.barh(range(len(lesson_counts)), lesson_counts['count'], 
                     color='lightcoral', edgecolor='black', alpha=0.8)
    ax2.set_yticks(range(len(lesson_counts)))
    # Reverse the labels to show highest count at top
    ax2.set_yticklabels([f"{title[:30]}..." if len(title) > 30 else title 
                        for title in lesson_counts['lessonTitle'][::-1]])
    ax2.set_xlabel('Number of Recommendations')
    ax2.set_title('Most Recommended Lessons')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Add the program recommendations heatmap
    st.subheader("Program Recommendations Analysis")
    plot_program_recommendations_heatmap(recommendations_df)
    
    # Add the program recommendations table
    st.subheader("Program Recommendations Summary Table")
    program_table_df = create_program_recommendations_table(recommendations_df)
    
    if program_table_df is not None:
        # Display the table
        st.dataframe(program_table_df, use_container_width=True)
    
    # Add the lesson recommendations table
    st.subheader("Lesson Recommendations Summary Table")
    lesson_table_df = create_lesson_recommendations_table(recommendations_df)
    
    if lesson_table_df is not None:
        # Display the table
        st.dataframe(lesson_table_df, use_container_width=True)

def create_lesson_recommendations_table(recommendations_df):
    """
    Create a summary table showing lesson recommendations aggregated by lesson.
    """
    print("INPUT TO CREATE_LESSON_RECOMMENDATIONS_TABLE")
    print(recommendations_df.shape)
    print(recommendations_df.head())
  
    if recommendations_df.empty:
        return None
    
    # Filter out rows with empty/null values for key fields
    filtered_df = recommendations_df.dropna(subset=['lessonTitle', 'parentTitle', 'parentKey'])
    filtered_df = filtered_df[
        (filtered_df['lessonTitle'].str.strip() != '') & 
        (filtered_df['parentTitle'].str.strip() != '') & 
        (filtered_df['parentKey'].str.strip() != '')
    ]
    
    if filtered_df.empty:
        return None
    
    # Aggregate by lesson title, program title, and program key
    lesson_counts = filtered_df.groupby(['lessonTitle', 'parentTitle', 'parentKey']).size().reset_index(name='recommendationCount')
    lesson_counts = lesson_counts.sort_values('recommendationCount', ascending=False)
    
    # Add top skills addressed (count unique users with gaps in each skill - only for users who are being recommended this specific lesson)
    lesson_skills = []
    for _, row in lesson_counts.iterrows():
        lesson_title = row['lessonTitle']
        parent_key = row['parentKey']
        
        # Get users who were recommended this specific lesson
        lesson_users = set()
        for user_id in filtered_df['userId'].unique():
            user_recommendations = filtered_df[filtered_df['userId'] == user_id]
            # Check if this user was recommended this specific lesson
            user_lesson_recommendations = user_recommendations[
                (user_recommendations['lessonTitle'] == lesson_title) & 
                (user_recommendations['parentKey'] == parent_key)
            ]
            if not user_lesson_recommendations.empty:
                lesson_users.add(user_id)
        
        # Get all weak skills for users who were recommended this lesson
        skill_user_counts = {}
        for user_id in lesson_users:
            user_recommendations = filtered_df[filtered_df['userId'] == user_id]
            if not user_recommendations.empty:
                weak_skills = user_recommendations.iloc[0]['weakSkills']
                for skill in weak_skills:
                    if skill not in skill_user_counts:
                        skill_user_counts[skill] = set()
                    skill_user_counts[skill].add(user_id)
        
        # Count unique users per skill and get top 5
        skill_counts = {skill: len(users) for skill, users in skill_user_counts.items()}
        top_5_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_skills_text = ', '.join([f"{skill} ({count} users)" for skill, count in top_5_skills])
        
        lesson_skills.append(top_skills_text)
    
    lesson_counts['topSkillsAddressed'] = lesson_skills
    
    # Rename columns for display
    lesson_counts = lesson_counts.rename(columns={
        'parentKey': 'Key (Program Key)',
        'parentTitle': 'Program Title',
        'lessonTitle': 'Lesson Title',
        'recommendationCount': '# of Users Recommended'
    })
    
    # Reorder columns
    lesson_counts = lesson_counts[['Key (Program Key)', 'Program Title', 'Lesson Title', '# of Users Recommended', 'topSkillsAddressed']]
    
    return lesson_counts

def create_learner_recommendations_table(recommendations_df):
    """
    Create downloadable table with one record per learner and pivoted columns for lessons and programs.
    The first column is learner email (resolved via Students API) instead of user ID.
    """
    if recommendations_df.empty:
        return None
    
    # Filter out rows with empty/null values for key fields
    filtered_df = recommendations_df.dropna(subset=['lessonTitle', 'parentTitle', 'parentKey'])
    filtered_df = filtered_df[
        (filtered_df['lessonTitle'].str.strip() != '') & 
        (filtered_df['parentTitle'].str.strip() != '') & 
        (filtered_df['parentKey'].str.strip() != '')
    ]
    
    if filtered_df.empty:
        return None
    
    # Create pivoted data structure
    pivoted_data = []
    
    for user_id in filtered_df['userId'].unique():
        user_recommendations = filtered_df[filtered_df['userId'] == user_id]
        
        # Get user info
        user_data = user_recommendations.iloc[0]
        
        # Initialize user record with all possible lesson columns
        user_record = {
            'userId': user_id,
            'totalScore': user_data['totalScore'],
            'weakSkills': ', '.join(user_data['weakSkills'])
        }
        
        # Initialize all lesson columns as empty
        for i in range(1, 6):  # lesson_1 through lesson_5
            user_record[f'lesson_{i}'] = ''
        
        # Get top 5 lessons per learner (by taking first 5 unique lessons in order)
        user_lessons = user_recommendations.drop_duplicates(subset=['lessonTitle']).head(5)
        
        # Add lesson recommendations (treat first lesson as lesson_1, second as lesson_2, etc.)
        for i, (_, lesson) in enumerate(user_lessons.iterrows()):
            lesson_num = i + 1
            # Format as "Lesson Title (Parent Title)" but only if both have values
            if pd.notna(lesson['lessonTitle']) and pd.notna(lesson['parentTitle']) and lesson['lessonTitle'].strip() and lesson['parentTitle'].strip():
                lesson_display = f"{lesson['lessonTitle']} ({lesson['parentTitle']})"
                user_record[f'lesson_{lesson_num}'] = lesson_display
        
        # Get top 1 program per learner (most common parent key)
        program_counts = user_recommendations['parentKey'].value_counts()
        if not program_counts.empty:
            # Get the most common program key that's not empty
            for program_key in program_counts.index:
                if pd.notna(program_key) and program_key.strip():
                    program_data = user_recommendations[user_recommendations['parentKey'] == program_key].iloc[0]
                    if pd.notna(program_data['parentTitle']) and program_data['parentTitle'].strip():
                        user_record['program_1'] = program_data['parentTitle']
                        break
        
        # Only add the record if it has at least one lesson or program
        if any(user_record[f'lesson_{i}'] for i in range(1, 6)) or user_record.get('program_1'):
            pivoted_data.append(user_record)
    
    if not pivoted_data:
        return None
    
    # Convert to DataFrame
    pivoted_df = pd.DataFrame(pivoted_data)
    
    with st.spinner("Loading learner emails..."):
        email_map = fetch_emails_for_user_ids(pivoted_df['userId'].tolist())
    pivoted_df['email'] = pivoted_df['userId'].map(lambda u: email_map.get(u, u))
    pivoted_df = pivoted_df.drop(columns=['userId'])
    
    # Reorder columns to have learner info first, then lesson recommendations, then program
    base_columns = ['email', 'totalScore', 'weakSkills']
    lesson_columns = []
    program_columns = []
    
    # Collect all lesson and program columns
    for col in pivoted_df.columns:
        if col.startswith('lesson_'):
            lesson_columns.append(col)
        elif col.startswith('program_'):
            program_columns.append(col)
    
    # Sort lesson columns by number
    lesson_columns.sort(key=lambda x: int(x.split('_')[1]))
    
    # Create final column order
    final_columns = base_columns + lesson_columns + program_columns
    pivoted_df = pivoted_df[final_columns]
    
    # Rename columns for better display
    column_mapping = {}
    for col in pivoted_df.columns:
        if col == 'email':
            column_mapping[col] = 'Email'
        elif col == 'totalScore':
            column_mapping[col] = 'Total Score'
        elif col == 'weakSkills':
            column_mapping[col] = 'Weak Skills'
        elif col.startswith('lesson_'):
            lesson_num = col.split('_')[1]
            column_mapping[col] = f'Lesson {lesson_num}'
        elif col.startswith('program_'):
            program_num = col.split('_')[1]
            column_mapping[col] = f'Program {program_num}'
    
    pivoted_df = pivoted_df.rename(columns=column_mapping)
    
    return pivoted_df

def plot_program_recommendations_heatmap(recommendations_df):
    """
    Create a bar chart + heatmap combination for program recommendations.
    Bar chart shows # of hits for #1 recommendation per program.
    Heatmap shows user-level recommendations (green if recommended, grey if not).
    """
    if recommendations_df.empty:
        return
    
    # Calculate program popularity (number of users who were recommended each program as their top choice)
    learner_program_counts = []
    for user_id in recommendations_df['userId'].unique():
        user_recommendations = recommendations_df[recommendations_df['userId'] == user_id]
        program_counts = user_recommendations['parentKey'].value_counts()
        if not program_counts.empty:
            top_program_key = program_counts.index[0]
            top_program_title = user_recommendations[user_recommendations['parentKey'] == top_program_key]['parentTitle'].iloc[0]
            learner_program_counts.append((top_program_key, top_program_title))
    
    program_counts_df = pd.DataFrame(learner_program_counts, columns=['parentKey', 'parentTitle'])
    program_counts_df = program_counts_df.groupby(['parentKey', 'parentTitle']).size().reset_index(name='count')
    program_counts_df = program_counts_df.sort_values('count', ascending=False)
    
    # Create user-program matrix for heatmap
    user_program_matrix = {}
    all_users = recommendations_df['userId'].unique()
    all_programs = program_counts_df['parentKey'].tolist()
    
    for user_id in all_users:
        user_program_matrix[user_id] = {}
        user_recommendations = recommendations_df[recommendations_df['userId'] == user_id]
        
        for program_key in all_programs:
            # Check if this program was recommended to this user
            is_recommended = program_key in user_recommendations['parentKey'].values
            user_program_matrix[user_id][program_key] = 1 if is_recommended else 0
    
    heatmap_df = pd.DataFrame.from_dict(user_program_matrix, orient='index')
    
    # Sort programs by popularity (same order as bar chart)
    heatmap_df = heatmap_df[program_counts_df['parentKey'].tolist()]
    
    # Sort users by total recommendations (most recommended users first)
    user_totals = heatmap_df.sum(axis=1)
    sorted_users = user_totals.sort_values(ascending=False).index
    heatmap_df = heatmap_df.loc[sorted_users]
    
    # Create program labels for y-axis
    program_labels = []
    for program_key in program_counts_df['parentKey']:
        program_title = program_counts_df[program_counts_df['parentKey'] == program_key]['parentTitle'].iloc[0]
        program_count = program_counts_df[program_counts_df['parentKey'] == program_key]['count'].iloc[0]
        program_labels.append(f"{program_title} ({program_count})")
    
    # Create Plotly subplot with shared y-axis
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.3, 0.7],
        subplot_titles=('Program Recommendation Summary', 'User-Program Recommendations'),
        specs=[[{"type": "bar"}, {"type": "heatmap"}]],
        shared_yaxes=True,
        horizontal_spacing=0.02
    )
    
    # Add horizontal bar chart (left)
    fig.add_trace(
        go.Bar(
            y=program_labels,
            x=program_counts_df['count'],
            orientation='h',
            marker=dict(
                color=program_counts_df['count'],
                colorscale='Viridis',
                showscale=False
            ),
            text=[f'{count}' for count in program_counts_df['count']],
            textposition='auto',
            name='Top Recommendations'
        ),
        row=1, col=1
    )
    
    # Add heatmap (right)
    fig.add_trace(
        go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns,
            y=heatmap_df.index,
            colorscale=[[0, 'lightgrey'], [1, 'green']],
            showscale=True,
            colorbar=dict(
                title=dict(text="Recommended"),
                tickmode="array",
                tickvals=[0, 1],
                ticktext=["No", "Yes"]
            )
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Program Recommendation Analysis",
        height=600,
        showlegend=False,
        xaxis_title="Number of Top Recommendations",
        xaxis2_title="Programs (Ranked by Popularity)",
        yaxis_title="",
        yaxis2_title="Users (Ranked by Total Recommendations)",
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        xaxis=dict(title_font=dict(color='black')),
        xaxis2=dict(title_font=dict(color='black'))
    )
    
    # Update axes
    fig.update_xaxes(showticklabels=False, row=1, col=1, color='black')
    fig.update_xaxes(showticklabels=True, row=1, col=2, color='black', tickfont=dict(color='black'))
    fig.update_yaxes(showticklabels=True, row=1, col=1, color='black', tickfont=dict(color='black'))
    fig.update_yaxes(showticklabels=True, row=1, col=2, color='black', tickfont=dict(color='black'))
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

def create_program_recommendations_table(recommendations_df):
    """
    Create a downloadable table showing program recommendations.
    """
    if recommendations_df.empty:
        return None
    
    # Filter out rows with empty/null values for key fields
    filtered_df = recommendations_df.dropna(subset=['parentTitle', 'parentKey'])
    filtered_df = filtered_df[
        (filtered_df['parentTitle'].str.strip() != '') & 
        (filtered_df['parentKey'].str.strip() != '')
    ]
    
    if filtered_df.empty:
        return None
    
    # Use direct aggregation like the lesson table (simpler and more reliable)
    program_counts = filtered_df.groupby(['parentKey', 'parentTitle']).size().reset_index(name='usersRecommended')
    program_counts = program_counts.sort_values('usersRecommended', ascending=False)
    
    # Add top skills addressed (count unique users with gaps in each skill - only for users who are being recommended this program)
    program_skills = []
    for _, row in program_counts.iterrows():
        program_key = row['parentKey']
        
        # Get users who were recommended this specific program
        program_users = set()
        program_recommendations = filtered_df[filtered_df['parentKey'] == program_key]
        for user_id in program_recommendations['userId'].unique():
            program_users.add(user_id)
        
        # Get all weak skills for users who were recommended this program
        skill_user_counts = {}
        for user_id in program_users:
            user_recommendations = filtered_df[filtered_df['userId'] == user_id]
            if not user_recommendations.empty:
                weak_skills = user_recommendations.iloc[0]['weakSkills']
                for skill in weak_skills:
                    if skill not in skill_user_counts:
                        skill_user_counts[skill] = set()
                    skill_user_counts[skill].add(user_id)
        
        # Count unique users per skill and get top 5
        skill_counts = {skill: len(users) for skill, users in skill_user_counts.items()}
        top_5_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_skills_text = ', '.join([f"{skill} ({count} users)" for skill, count in top_5_skills])
        
        program_skills.append(top_skills_text)
    
    program_counts['topSkillsAddressed'] = program_skills
    
    # Rename columns for display
    program_counts = program_counts.rename(columns={
        'parentKey': 'Key',
        'parentTitle': 'Program Title',
        'usersRecommended': '# of Users Recommended'
    })
    
    return program_counts 
