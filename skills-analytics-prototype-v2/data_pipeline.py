"""
Data Pipeline for Skills Analytics
Consolidates data fetching and processing from multiple sources:
- EMC Content API
- Udacity Assessments API
- Workera API
- Classroom Content API
- Skills APIs
"""

# Standard library imports
import json
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
import requests
import pandas as pd

# Local imports
import settings
import importlib

# Force reload settings to get latest values (helps with Streamlit caching)
importlib.reload(settings)


# =============================================================================
# CONFIGURATION & API URLs
# =============================================================================

# Authentication - now provided dynamically via session state
# These will be set when the app initializes with user-provided credentials
_UDACITY_STAFF_JWT = None
_WORKERA_API_KEY = None
_COMPANY_ID = None

def set_credentials(jwt_token, workera_api_key, company_id=None):
    """Set the credentials for API calls. Called after authentication."""
    global _UDACITY_STAFF_JWT, _WORKERA_API_KEY, _COMPANY_ID
    _UDACITY_STAFF_JWT = jwt_token
    _WORKERA_API_KEY = workera_api_key
    _COMPANY_ID = company_id

def get_jwt_token():
    """Get the current JWT token."""
    if _UDACITY_STAFF_JWT is None:
        raise ValueError("JWT token not set. Please authenticate first.")
    return _UDACITY_STAFF_JWT

def get_workera_api_key():
    """Get the current Workera API key."""
    if _WORKERA_API_KEY is None:
        raise ValueError("Workera API key not set. Please authenticate first.")
    return _WORKERA_API_KEY

def get_company_id():
    """Get the current company ID."""
    return _COMPANY_ID

# API Endpoints
EMC_CONTENT_API_URL = settings.emc_content_api_url
ASSESSMENTS_API_URL = settings.assessments_api_url
CLASSROOM_CONTENT_API_URL = settings.classroom_content_api_url
UTAXONOMY_API_URL = settings.utaxonomy_api_url
SKILLS_API_URL = settings.skills_api_url
SKILLS_SEARCH_API_URL = settings.skills_search_api_url
UDACITY_SKILLS_SEARCH_API_URL = settings.udacity_skills_api_url
LEARNING_ACTIVITY_API_URL = settings.learning_activity_api_url
WORKERA_DOMAINS_URL = settings.workera_url_domains
WORKERA_SIGNALS_CAT_URL = settings.workera_url_signals_cat


# =============================================================================
# AUTHENTICATION FUNCTIONS
# =============================================================================

def fetch_workera_api_key(company_id, jwt_token):
    """
    Fetch a Workera API key for a given company_id using the Udacity API.
    
    Args:
        company_id: The company ID to fetch the API key for
        jwt_token: A valid STAFF or SERVICE JWT for calling this endpoint
    
    Returns:
        str: The Workera API key
        
    Raises:
        Exception: If the API call fails
    """
    url = f"{settings.udacity_workera_api_keys_url}/{company_id}"
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/json",
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch Workera API key (status={response.status_code}): {response.text}"
        )
    
    data = response.json()
    # The API returns: {"id": "...", "company_id": ..., "value": "<API_KEY>", ...}
    if isinstance(data, dict):
        return data.get('value')
    return str(data)


# =============================================================================
# WORKERA PIPELINE FUNCTIONS
# =============================================================================

def fetch_workera_results_paginated(url):
    """
    Fetch paginated data from Workera API.
    
    Args:
        url: API endpoint URL
        
    Returns:
        pd.DataFrame: Results from all pages
    """
    headers = {
        "Authorization": f"Bearer {get_workera_api_key()}",
        "Accept": "application/json"
    }
    
    all_results = []
    
    while url:
        response = requests.get(url, headers=headers)
        data = response.json()
        
        for item in data['data']:
            all_results.append(item)
        
        if data.get('has_more'):
            url = data.get('next_page')
        else:
            url = None
    
    return pd.DataFrame(all_results)


def fetch_workera_results():
    """
    Fetch Workera assessment results with user email, scores, and skill data.
    
    Returns:
        pd.DataFrame: Merged Workera data with domains and signals
    """
    domains = fetch_workera_results_paginated(WORKERA_DOMAINS_URL)[['title', 'identifier']]
    signals_cat = fetch_workera_results_paginated(WORKERA_SIGNALS_CAT_URL)[[
        'user', 'score', 'target_score', 'iteration_number', 'domain_identifier',
        'strong_skills', 'needs_improvement_skills', 'created_at'
    ]]
    
    signals_cat['user'] = [user['email'] for user in signals_cat['user']]
    
    workera_data = signals_cat.merge(domains, left_on='domain_identifier', right_on='identifier')
    workera_data = workera_data.drop(columns=['identifier'])
    
    return workera_data


# =============================================================================
# LESSON RECOMMENDER FUNCTIONS
# =============================================================================

def skills_api_search(weak_skills):
    """
    Search for lessons related to skills using Udacity Skills API.
    
    Args:
        weak_skills: List of skill names to search
        
    Returns:
        dict: JSON response with lesson recommendations
    """
    headers = {
        'content-type': 'application/json',
        'Authorization': f'Bearer {get_jwt_token()}',
        'Accept': 'application/json'
    }
    
    payload = {
        'search': weak_skills,
        'searchField': "knowledge_component_names",
    }
    
    response = requests.post(UDACITY_SKILLS_SEARCH_API_URL, headers=headers, data=json.dumps(payload))
    return response.json()


def process_lessons(lessons):
    """
    Process raw lesson data into structured format.
    
    Args:
        lessons: Raw lesson data from skills API
        
    Returns:
        list: Processed lesson dictionaries
    """
    processed_data = []
    
    for lesson in lessons:
        data = {
            'id': lesson['lesson']['id'],
            'label': lesson['lesson']['content']['label'],
            'duration': lesson['lesson']['content']['duration'],
            'parent_key': lesson['search']['metadata']['parent_key'],
            'parent_title': lesson['search']['metadata']['parent_title'],
            'parent_duration': lesson['search']['metadata']['parent_duration']
        }
        processed_data.append(data)
    
    return processed_data


def bulk_process_lessons(data):
    """
    Process lessons for multiple learners (parallel processing).
    
    Args:
        data: DataFrame with 'needs_improvement_skills' column
        
    Returns:
        list: Flattened list of all processed lessons
    """
    def process_single_learner(learner_data):
        """Helper function to process a single learner's lessons."""
        lessons = skills_api_search(learner_data['needs_improvement_skills'])
        return process_lessons(lessons)
    
    results = []
    
    # Convert DataFrame rows to list of dicts for parallel processing
    learners = [learner for _, learner in data.iterrows()]
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_single_learner, learner) for learner in learners]
        
        for future in as_completed(futures):
            try:
                processed_lessons = future.result()
                results.append(processed_lessons)
            except Exception as e:
                print(f"Error processing learner lessons: {str(e)}")
                results.append([])
    
    flattened_lessons = [
        lesson
        for learner in results
        for lesson in learner
    ]
    
    return flattened_lessons


def create_recommended_lessons_dataset(data):
    """
    Create dataset of recommended lessons with duplicate counts.
    
    Args:
        data: DataFrame with 'needs_improvement_skills' column
        
    Returns:
        pd.DataFrame: Recommended lessons with counts, sorted by frequency
    """
    raw_data = bulk_process_lessons(data)
    recommended_lessons_df = pd.DataFrame(raw_data)
    
    recommended_lessons_df = recommended_lessons_df.groupby(
        recommended_lessons_df.columns.tolist(), 
        dropna=False
    ).size().reset_index(name='count')
    
    recommended_lessons_df = recommended_lessons_df.sort_values(
        by='count', 
        ascending=False
    ).reset_index(drop=True)
    
    return recommended_lessons_df


# =============================================================================
# EMC CONTENT API FUNCTIONS
# =============================================================================

def get_emc_content_data():
    """
    Fetch company learners data from the EMC Content API.
    
    Returns:
        dict: JSON response containing company learners data, or None if error
    """
    company_id = get_company_id() or settings.default_company_id
    query = """
   {
      company(id: %d) {
        learners {
          edges {
            node {
              email
              id
              userKey
              groups {
                name
              }
              roster {
                enrollmentId
                programInfo {
                  programKey
                }
                learnerActivity {
                  totalProjectsPassed
                  conceptsViewed
                  questionsAnswered
                  enrolledAt
                  graduatedAt
                  unenrolledAt
                  submissions {
                    projectId
                    projectName
                    submissionDate
                    status
                  }
                } 
              }
            }
          }
        }
      }
    }
    """ % company_id
    
    headers = {
        "Authorization": f"Bearer {get_jwt_token()}",
        "Content-Type": "application/json"
    }
    
    payload = {"query": query}
    
    #response = requests.post('https://api.udacity.com/api/emc/gql/query/public', headers=headers, json=payload)
    response = requests.post(EMC_CONTENT_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    
    # Return data even if it has GraphQL errors - let caller handle it
    return data


# =============================================================================
# UDACITY ASSESSMENTS API FUNCTIONS
# =============================================================================

def get_user_assessment_attempts(user_key):
    """
    Fetch assessment attempts for a specific user.
    
    Args:
        user_key: The Udacity user key
        
    Returns:
        list: List of assessment attempts for the user
    """
    query = f"""
    {{
      attempts(input: {{
        userId: "{user_key}"
        limit: 10000
        page: 1
      }}) {{
        attempts {{
          id
          assessmentId
          createdAt
          userId
          status
          result
          report {{
            result
            totalScore
            sectionReports {{
              questionReports {{
                questionId
                questionScore
                question {{
                  skillId
                }}
              }}
            }}
          }}
        }}
      }}
    }}
    """
    
    headers = {
        "Authorization": f"Bearer {get_jwt_token()}",
        "Content-Type": "application/json"
    }
    
    payload = {"query": query}
    
    try:
        response = requests.post(ASSESSMENTS_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get('data', {}).get('attempts', {}).get('attempts', [])
    except Exception:
        return []


def get_all_assessment_attempts(emc_data):
    """
    Get assessment attempts for all users from EMC content data (parallel processing).
    
    Args:
        emc_data: Response data from get_emc_content_data()
        
    Returns:
        list: List of dicts with user_id and their attempts
    """
    all_attempts = []
    
    learners = emc_data.get('data', {}).get('company', {}).get('learners', {}).get('edges', [])
    user_keys = [edge.get('node', {}).get('userKey') for edge in learners if edge.get('node', {}).get('userKey')]
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_user = {
            executor.submit(get_user_assessment_attempts, user_key): user_key 
            for user_key in user_keys
        }
        
        for future in as_completed(future_to_user):
            user_id = future_to_user[future]
            attempts = future.result()
            
            if attempts:
                all_attempts.append({
                    'user_id': user_id,
                    'attempts': attempts
                })
    
    return all_attempts


def get_skill_name(skill_id):
    """
    Get skill name from skill ID using GraphQL query.
    
    Args:
        skill_id: UUID of the skill
        
    Returns:
        str: Skill name or 'Unknown Skill' if not found
    """
    query = """
    query getSkill {
      skill(input:{id: "%s"}) {
        id
        title
      }
    }
    """ % skill_id
    
    headers = {
        "Authorization": f"Bearer {get_jwt_token()}",
        "Content-Type": "application/json"
    }
    
    payload = {"query": query}
    
    try:
        response = requests.post(ASSESSMENTS_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Check for GraphQL errors
        if data.get('errors'):
            print(f"GraphQL error for skill {skill_id}: {data['errors']}")
            return 'Unknown Skill'
        
        if data.get('data') and data['data'].get('skill'):
            return data['data']['skill'].get('title', 'Unknown Skill')
        
        return 'Unknown Skill'
    except requests.exceptions.RequestException as e:
        print(f"Request error fetching skill {skill_id}: {str(e)}")
        return 'Unknown Skill'
    except Exception as e:
        print(f"Unexpected error fetching skill {skill_id}: {str(e)}")
        return 'Unknown Skill'


def get_all_skill_names(skill_ids):
    """
    Batch fetch skill names for multiple skill IDs (parallel processing).
    
    Args:
        skill_ids: Set or list of skill IDs
        
    Returns:
        dict: Mapping of skill_id to name
    """
    skill_names = {}
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_id = {
            executor.submit(get_skill_name, skill_id): skill_id
            for skill_id in skill_ids
        }
        
        for future in as_completed(future_to_id):
            skill_id = future_to_id[future]
            try:
                name = future.result()
                skill_names[skill_id] = name
            except Exception as e:
                print(f"Error processing skill {skill_id}: {str(e)}")
                skill_names[skill_id] = 'Unknown Skill'
    
    return skill_names


def get_udacity_assessment_title(assessment_id):
    """
    Get assessment title from assessment ID.
    
    Args:
        assessment_id: UUID of the assessment
        
    Returns:
        str: Assessment title or 'Unknown Assessment' if not found
    """
    query = """
    query getAssessment {
      assessment(id:"%s") {
        id
        title
      }
    }
    """ % assessment_id
    
    headers = {
        "Authorization": f"Bearer {get_jwt_token()}",
        "Content-Type": "application/json"
    }
    
    payload = {"query": query}
    
    try:
        response = requests.post(ASSESSMENTS_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Check for GraphQL errors
        if data.get('errors'):
            print(f"GraphQL error for assessment {assessment_id}: {data['errors']}")
            return 'Unknown Assessment'
        
        if data.get('data') and data['data'].get('assessment'):
            return data['data']['assessment'].get('title', 'Unknown Assessment')
        
        print(f"No assessment data found for {assessment_id}")
        return 'Unknown Assessment'
    except requests.exceptions.RequestException as e:
        print(f"Request error fetching assessment {assessment_id}: {str(e)}")
        return 'Unknown Assessment'
    except Exception as e:
        print(f"Unexpected error fetching assessment {assessment_id}: {str(e)}")
        return 'Unknown Assessment'


def get_all_udacity_assessment_titles(assessment_ids):
    """
    Batch fetch assessment titles for multiple assessment IDs (parallel processing).
    
    Args:
        assessment_ids: Set or list of assessment IDs
        
    Returns:
        dict: Mapping of assessment_id to title
    """
    assessment_titles = {}
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_id = {
            executor.submit(get_udacity_assessment_title, assessment_id): assessment_id
            for assessment_id in assessment_ids
        }
        
        for future in as_completed(future_to_id):
            assessment_id = future_to_id[future]
            try:
                title = future.result()
                assessment_titles[assessment_id] = title
            except Exception as e:
                print(f"Error processing assessment {assessment_id}: {str(e)}")
                assessment_titles[assessment_id] = 'Unknown Assessment'
    
    return assessment_titles


def get_all_assessment_attempts_combined(udacity_attempts_data, emc_content_data):
    """
    Combine Udacity and Workera assessment attempts into single dataset.
    
    Args:
        udacity_attempts_data: List of Udacity assessment attempts
        emc_content_data: EMC content data for user mapping
        
    Returns:
        pd.DataFrame: Combined assessment attempts with email, scores, and metadata
    """
    user_key_to_email = {}
    learners = emc_content_data.get('data', {}).get('company', {}).get('learners', {}).get('edges', [])
    for edge in learners:
        node = edge.get('node', {})
        user_key = node.get('userKey')
        email = node.get('email')
        if user_key and email:
            user_key_to_email[user_key] = email
    
    # Process Udacity assessments
    udacity_records = []
    assessment_ids = set()
    
    for user_data in udacity_attempts_data:
        user_key = user_data['user_id']
        email = user_key_to_email.get(user_key, 'Unknown')
        
        for attempt in user_data['attempts']:
            assessment_id = attempt.get('assessmentId')
            if assessment_id:
                assessment_ids.add(assessment_id)
            
            created_at = attempt.get('createdAt', '').split('T')[0]
            report = attempt.get('report', {})
            score = report.get('totalScore', 0) if report else 0
            
            udacity_records.append({
                'assessmentSource': 'Udacity',
                'email': email,
                'assessment_id': assessment_id,
                'assessment_name': None,
                'score': score,
                'created_at': created_at
            })
    
    # Fetch assessment titles
    assessment_titles = get_all_udacity_assessment_titles(assessment_ids)
    
    # Apply assessment titles
    for record in udacity_records:
        assessment_id = record['assessment_id']
        record['assessment_name'] = assessment_titles.get(assessment_id, 'Unknown Assessment')
    
    # Process Workera assessments
    workera_data = fetch_workera_results()
    workera_records = []
    
    for idx, row in workera_data.iterrows():
        workera_records.append({
            'assessmentSource': 'Workera',
            'email': row['user'],
            'assessment_id': row['domain_identifier'],
            'assessment_name': row['title'],
            'score': row['score'],
            'created_at': row['created_at'].split('T')[0] if isinstance(row['created_at'], str) else str(row['created_at']).split(' ')[0]
        })
    
    # Combine both sources
    all_records = udacity_records + workera_records
    df = pd.DataFrame(all_records)
    
    if len(df) > 0:
        df['created_at'] = pd.to_datetime(df['created_at'])
    
    return df


def filter_latest_by_date_range(df_all_attempts, start_date, end_date):
    """
    Filter assessment attempts by date range and return latest attempt per user per assessment.
    
    Args:
        df_all_attempts: DataFrame with all assessment attempts
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        
    Returns:
        pd.DataFrame: Filtered and deduplicated attempts
    """
    df_filtered = df_all_attempts[
        (df_all_attempts['created_at'] >= start_date) &
        (df_all_attempts['created_at'] <= end_date)
    ].copy()
    
    df_latest = df_filtered.sort_values('created_at', ascending=False).groupby(
        ['assessmentSource', 'email', 'assessment_id']
    ).first().reset_index()
    
    df_latest = df_latest.sort_values(['assessmentSource', 'created_at']).reset_index(drop=True)
    
    return df_latest


# =============================================================================
# TIMESERIES & AGGREGATION FUNCTIONS
# =============================================================================

def get_passed_projects_timeseries(data):
    """
    Aggregate passed project submissions by date.
    
    Args:
        data: Response data from get_emc_content_data()
        
    Returns:
        dict: {date_string: count} - Number of passed projects per date
    """
    date_counts = defaultdict(int)
    
    learners = data.get('data', {}).get('company', {}).get('learners', {}).get('edges', [])
    
    for learner in learners:
        node = learner.get('node', {})
        for enrollment in node.get('roster', []):
            learner_activity = enrollment.get('learnerActivity', {})
            submissions = learner_activity.get('submissions', [])
            
            if submissions:
                for submission in submissions:
                    if submission.get('status') == 'passed':
                        submission_date = submission.get('submissionDate', '').split('T')[0]
                        if submission_date:
                            date_counts[submission_date] += 1
    
    return dict(sorted(date_counts.items()))


def get_passed_projects_timeseries_detailed(emc_data):
    """
    Create detailed timeseries of passed projects by program, project name, and date.
    
    Args:
        emc_data: Response data from get_emc_content_data()
        
    Returns:
        pd.DataFrame: Columns [programKey, projectName, date, passed_projects]
    """
    submissions_data = []
    
    learners = emc_data.get('data', {}).get('company', {}).get('learners', {}).get('edges', [])
    
    for edge in learners:
        node = edge.get('node', {})
        roster = node.get('roster', [])
        
        for enrollment in roster:
            program_info = enrollment.get('programInfo', {})
            program_key = program_info.get('programKey', 'Unknown')
            
            learner_activity = enrollment.get('learnerActivity', {})
            submissions = learner_activity.get('submissions', [])
            
            if submissions:
                for submission in submissions:
                    if submission.get('status') == 'passed':
                        project_name = submission.get('projectName', 'Unknown')
                        submission_date = submission.get('submissionDate', '').split('T')[0]
                        
                        if submission_date:
                            submissions_data.append({
                                'programKey': program_key,
                                'projectName': project_name,
                                'date': submission_date
                            })
    
    if submissions_data:
        df = pd.DataFrame(submissions_data)
        df_agg = df.groupby(['programKey', 'projectName', 'date']).size().reset_index(name='passed_projects')
        df_agg['date'] = pd.to_datetime(df_agg['date'])
        df_agg = df_agg.sort_values('date').reset_index(drop=True)
        return df_agg
    else:
        return pd.DataFrame(columns=['programKey', 'projectName', 'date', 'passed_projects'])


def get_passed_assessments_timeseries(assessment_attempts_data):
    """
    Create timeseries of passed Udacity assessments.
    
    Args:
        assessment_attempts_data: List of assessment attempts from get_all_assessment_attempts()
        
    Returns:
        dict: {date_string: count} - Number of passed assessments per date
    """
    date_counts = defaultdict(int)
    
    for user_data in assessment_attempts_data:
        attempts = user_data.get('attempts', [])
        
        for attempt in attempts:
            if attempt.get('status') == 'COMPLETED' and attempt.get('result') == 'PASSED':
                created_at = attempt.get('createdAt', '')
                assessment_date = created_at.split('T')[0]
                
                if assessment_date:
                    date_counts[assessment_date] += 1
    
    return dict(sorted(date_counts.items()))


def get_user_progress(user_key):
    """
    Fetch user progress/activity data from learning-activity API.
    
    Args:
        user_key: The Udacity user key
        
    Returns:
        list: List of activity records
    """
    query = f"""
    {{
      user(id: "{user_key}") {{
        programs {{
          period(
            start: "2015-01-01T00:00:00Z" 
            end: "2030-07-01T00:00:00Z"
          ) {{
            activity {{
              source
              occurred_at
            }}
          }}
        }}
      }}
    }}
    """
    
    headers = {
        "Authorization": f"Bearer {get_jwt_token()}",
        "Content-Type": "application/json"
    }
    
    payload = {"query": query}
    
    try:
        response = requests.post(
            LEARNING_ACTIVITY_API_URL,
            headers=headers,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        # Check for GraphQL errors
        if 'errors' in data:
            print(f"GraphQL errors for user {user_key}: {data['errors']}")
            return []
        
        user_data = data.get('data', {}).get('user', {})
        if not user_data:
            print(f"No user data returned for {user_key}")
            return []
            
        programs = user_data.get('programs', [])
        
        all_activity = []
        for program in programs:
            period = program.get('period', {})
            activity = period.get('activity', [])
            all_activity.extend(activity)
        
        return all_activity
    except requests.exceptions.RequestException as e:
        print(f"API request failed for user {user_key}: {str(e)}")
        return []
    except Exception as e:
        print(f"Error processing user {user_key}: {str(e)}")
        return []


def get_all_user_progress(emc_data):
    """
    Get progress/activity data for all users from EMC content data.
    
    Args:
        emc_data: Response data from get_emc_content_data()
        
    Returns:
        list: List of dicts with user_key and their activity
    """
    all_progress = []
    
    learners = emc_data.get('data', {}).get('company', {}).get('learners', {}).get('edges', [])
    user_keys = [edge.get('node', {}).get('userKey') for edge in learners if edge.get('node', {}).get('userKey')]
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_user = {
            executor.submit(get_user_progress, user_key): user_key 
            for user_key in user_keys
        }
        
        for future in as_completed(future_to_user):
            user_key = future_to_user[future]
            activity = future.result()
            
            if activity:
                all_progress.append({
                    'user_key': user_key,
                    'activity': activity
                })
    
    return all_progress


def get_learner_frequency_by_month(user_progress_data):
    """
    Calculate average number of active days per month for learners who were active.
    
    Args:
        user_progress_data: List of dicts with user_key and activity
        
    Returns:
        pd.DataFrame: Columns [month, average_active_days, active_learners]
    """
    # Track unique days per user per month
    user_month_days = defaultdict(lambda: defaultdict(set))
    
    for user_data in user_progress_data:
        user_key = user_data['user_key']
        
        for activity in user_data['activity']:
            occurred_at = activity.get('occurred_at')
            
            if occurred_at:
                # Parse the datetime
                dt = pd.to_datetime(occurred_at)
                month_key = dt.strftime('%Y-%m')  # e.g., "2025-01"
                day_key = dt.strftime('%Y-%m-%d')  # e.g., "2025-01-15"
                
                # Track unique days per user per month
                user_month_days[user_key][month_key].add(day_key)
    
    # Calculate average active days per month
    month_stats = defaultdict(list)
    
    for user_key, months in user_month_days.items():
        for month, days in months.items():
            num_active_days = len(days)
            if num_active_days > 0:  # Only include users who were active
                month_stats[month].append(num_active_days)
    
    # Create DataFrame
    df_data = []
    for month in sorted(month_stats.keys()):
        active_days_list = month_stats[month]
        avg_active_days = sum(active_days_list) / len(active_days_list)
        
        df_data.append({
            'month': month,
            'average_active_days': avg_active_days,
            'active_learners': len(active_days_list)
        })
    
    df = pd.DataFrame(df_data)
    
    if len(df) > 0:
        df['month'] = pd.to_datetime(df['month'])
    
    return df


# =============================================================================
# SKILLS HIERARCHY & TAXONOMY FUNCTIONS
# =============================================================================

def get_skill_hierarchy(skill_name):
    """
    Fetch the complete hierarchy for a skill (skill -> subject -> domain).
    Uses the taxonomy GraphQL API to search by name, then the skills REST API to get parent relationships.
    
    Args:
        skill_name (str): The skill name to look up
        
    Returns:
        dict: Dictionary containing skill_id, skill_name, subject_id, subject_name, domain_id, domain_name
              Returns None if the skill is not found or API call fails
    """
    headers = {
        "Authorization": f"Bearer {get_jwt_token()}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        # Step 1: Search for the skill by name using GraphQL
        skill_search_query = f"""
        query {{
          topics(input: {{
            typeUri: "model:Udaciskill", 
            nameSearch: "{skill_name}", 
            limit: 1, 
            offset: 0
          }}) {{
            displayName
            uri
          }}
        }}
        """
        
        payload = {"query": skill_search_query}
        response = requests.post(UTAXONOMY_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        if 'errors' in data or not data.get('data', {}).get('topics'):
            return None
        
        topics = data['data']['topics']
        if len(topics) == 0:
            return None
        
        skill = topics[0]
        skill_uri = skill.get('uri')
        skill_display_name = skill.get('displayName', 'Unknown Skill')
        skill_uuid = skill_uri.replace('taxonomy:', '') if skill_uri else None
        
        # Step 2: Get skill's parent (subject) using REST API
        skill_parents_url = f"{SKILLS_API_URL}/{skill_uuid}/dir/in"
        response = requests.get(skill_parents_url, headers=headers)
        response.raise_for_status()
        parents_response = response.json()
        
        parent_nodes = parents_response.get('nodes', [])
        if len(parent_nodes) == 0:
            return None
        
        subject = parent_nodes[0]
        subject_id = subject.get('id')
        subject_content = subject.get('content', {})
        subject_name = subject_content.get('label', 'Unknown Subject')
        
        # Step 3: Get subject's parent (domain) - but keep climbing to find meaningful top level
        # Exclude generic root nodes like "Udacity Domain"
        current_id = subject_id
        current_name = subject_name
        domain_id = subject_id
        domain_name = subject_name
        previous_id = subject_id
        previous_name = subject_name
        
        # Keep climbing the hierarchy until we reach the top or a root node
        max_levels = 5  # Safety limit to prevent infinite loops
        level = 0
        
        # Root node identifiers to stop before
        root_node_names = ['Udacity Domain', 'udacity domain', 'root']
        
        while level < max_levels:
            parent_url = f"{SKILLS_API_URL}/{current_id}/dir/in"
            try:
                response = requests.get(parent_url, headers=headers)
                response.raise_for_status()
                parent_response = response.json()
                
                parent_nodes = parent_response.get('nodes', [])
                
                if len(parent_nodes) == 0:
                    # No more parents - use current level as domain
                    break
                
                # Move up to the parent
                parent = parent_nodes[0]
                parent_id = parent.get('id')
                parent_content = parent.get('content', {})
                parent_name = parent_content.get('label', current_name)
                
                # Check if we've reached a root node
                if parent_name.lower() in root_node_names:
                    # Use the previous level (one below root) as the domain
                    break
                
                # Save current as previous before moving up
                previous_id = current_id
                previous_name = current_name
                
                # Move to parent
                current_id = parent_id
                current_name = parent_name
                
                # Update domain to this level
                domain_id = current_id
                domain_name = current_name
                
                level += 1
            except Exception:
                # If we can't fetch the parent, use current level
                break
        
        return {
            'skill_id': skill_uuid,
            'skill_name': skill_display_name,
            'subject_id': subject_id,
            'subject_name': subject_name,
            'domain_id': domain_id,
            'domain_name': domain_name
        }
        
    except Exception:
        return None


def convert_skill_to_udacity_skill(skill_description, scope=None):
    """
    Convert any skill description to a canonical Udacity skill name using the Skills Search API.
    
    Args:
        skill_description (str): Single skill description to search
        scope (str, optional): Scope to limit search
        
    Returns:
        str: Name of top matching skill category or None if no results found
    """
    headers = {
        "Authorization": f"Bearer {get_jwt_token()}",
        "Content-Type": "application/json",
        "Accept": "*/*"
    }
    
    payload = {
        "search": [skill_description],
        "searchField": "knowledge_component_desc",
        "filter": {},
        "limit": 1,
        "deduplicate": False,
        "scopeLimit": 5
    }
    
    if scope:
        payload["scope"] = scope
    
    try:
        response = requests.post(SKILLS_SEARCH_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        groups = data.get('groups', {})
        if not groups:
            return None
        
        # Find group with highest score
        top_skill = None
        top_score = -1
        for skill_name, items in groups.items():
            if items and len(items) > 0:
                item_score = items[0].get('score', 0)
                if item_score > top_score:
                    top_score = item_score
                    top_skill = skill_name
        
        return top_skill
    except Exception as e:
        print(f"Error converting skill '{skill_description}': {str(e)}")
        return None


def batch_convert_skills_to_udacity(skill_descriptions, scope=None):
    """
    Batch convert multiple skill descriptions to Udacity skills (parallel processing).
    
    Args:
        skill_descriptions: List or set of skill descriptions
        scope (str, optional): Scope to limit search
        
    Returns:
        dict: Mapping of original skill description to Udacity skill name
    """
    skill_mapping = {}
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_skill = {
            executor.submit(convert_skill_to_udacity_skill, skill, scope): skill
            for skill in skill_descriptions
        }
        
        for future in as_completed(future_to_skill):
            original_skill = future_to_skill[future]
            try:
                udacity_skill = future.result()
                skill_mapping[original_skill] = udacity_skill
            except Exception as e:
                print(f"Error processing skill conversion '{original_skill}': {str(e)}")
                skill_mapping[original_skill] = None
    
    return skill_mapping


def batch_get_skill_hierarchies(skill_names):
    """
    Batch fetch hierarchies for multiple skill names (parallel processing).
    
    Args:
        skill_names: List or set of skill names
        
    Returns:
        dict: Mapping of skill name to hierarchy dict
    """
    hierarchy_mapping = {}
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_skill = {
            executor.submit(get_skill_hierarchy, skill_name): skill_name
            for skill_name in skill_names
        }
        
        for future in as_completed(future_to_skill):
            skill_name = future_to_skill[future]
            try:
                hierarchy = future.result()
                hierarchy_mapping[skill_name] = hierarchy
            except Exception as e:
                print(f"Error processing skill hierarchy '{skill_name}': {str(e)}")
                hierarchy_mapping[skill_name] = None
    
    return hierarchy_mapping


# =============================================================================
# SKILL ACQUISITION FUNCTIONS
# =============================================================================

def get_all_skill_acquisitions(emc_data, udacity_assessment_data, project_skills_csv_path='data/project_skills_enriched.csv'):
    """
    Track skill acquisitions from passed project submissions, Udacity assessments, and Workera assessments.
    
    Args:
        emc_data: Response data from get_emc_content_data()
        udacity_assessment_data: List of dicts with user assessment attempts
        project_skills_csv_path: Path to project_skills_enriched.csv
        
    Returns:
        pd.DataFrame: Columns [source, date, email, skill_id, skill_name, subject_id, subject_name, domain_id, domain_name]
    """
    # Load project skills mapping
    project_skills_df = pd.read_csv(project_skills_csv_path)
    
    project_to_skills = {}
    for _, row in project_skills_df.iterrows():
        project_id = str(row['project_id'])
        skill_name = row['skill_name']
        skill_uri = row['skill_uri']
        skill_id = skill_uri.replace('taxonomy:', '') if pd.notna(skill_uri) else None
        
        if project_id not in project_to_skills:
            project_to_skills[project_id] = []
        
        project_to_skills[project_id].append({
            'skill_id': skill_id,
            'skill_name': skill_name
        })
    
    skill_acquisitions = []
    
    # Process Projects
    learners = emc_data.get('data', {}).get('company', {}).get('learners', {}).get('edges', [])
    
    for learner_edge in learners:
        node = learner_edge.get('node', {})
        learner_email = node.get('email', 'Unknown')
        roster = node.get('roster', [])
        
        for enrollment in roster:
            learner_activity = enrollment.get('learnerActivity', {})
            submissions = learner_activity.get('submissions', [])
            
            if not submissions:
                continue
            
            for submission in submissions:
                if submission.get('status') == 'passed':
                    project_id = str(submission.get('projectId', ''))
                    submission_date = submission.get('submissionDate', '').split('T')[0]
                    
                    if project_id in project_to_skills:
                        skills = project_to_skills[project_id]
                        
                        for skill in skills:
                            skill_acquisitions.append({
                                'source': 'Project',
                                'date': submission_date,
                                'email': learner_email,
                                'skill_id': skill['skill_id'],
                                'skill_name': skill['skill_name'],
                                'subject_id': None,
                                'subject_name': None,
                                'domain_id': None,
                                'domain_name': None
                            })
    
    # Process Udacity Assessments
    user_key_to_email = {}
    for learner_edge in learners:
        node = learner_edge.get('node', {})
        user_key = node.get('userKey')
        email = node.get('email')
        if user_key and email:
            user_key_to_email[user_key] = email
    
    def get_passed_skills(attempt):
        """Return skills where user got >= 2/3 questions correct."""
        passed_skills = []
        try:
            report = attempt.get('report')
            if not report or not isinstance(report, dict):
                return passed_skills
            
            section_reports = report.get('sectionReports', [])
            if not section_reports:
                return passed_skills
            
            skill_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
            
            for section in section_reports:
                question_reports = section.get('questionReports', [])
                for q_report in question_reports:
                    question = q_report.get('question', {})
                    skill_id = question.get('skillId')
                    
                    if skill_id:
                        skill_stats[skill_id]['total'] += 1
                        question_score = q_report.get('questionScore', 0)
                        if question_score and question_score > 0:
                            skill_stats[skill_id]['correct'] += 1
            
            for skill_id, stats in skill_stats.items():
                if stats['total'] > 0:
                    accuracy = stats['correct'] / stats['total']
                    if accuracy >= 2/3:
                        passed_skills.append(skill_id)
        except Exception:
            pass
        
        return passed_skills
    
    all_skill_ids = set()
    assessment_skill_records = []
    
    for user_data in udacity_assessment_data:
        user_key = user_data['user_id']
        email = user_key_to_email.get(user_key, 'Unknown')
        
        assessment_attempts = {}
        for attempt in user_data['attempts']:
            if attempt.get('status') == 'COMPLETED' and attempt.get('result') == 'PASSED':
                assessment_id = attempt.get('assessmentId')
                created_at = attempt.get('createdAt')
                
                if assessment_id:
                    if assessment_id not in assessment_attempts or created_at > assessment_attempts[assessment_id]['createdAt']:
                        assessment_attempts[assessment_id] = attempt
        
        for assessment_id, attempt in assessment_attempts.items():
            created_at = attempt.get('createdAt', '').split('T')[0]
            passed_skill_ids = get_passed_skills(attempt)
            
            if passed_skill_ids:
                all_skill_ids.update(passed_skill_ids)
                assessment_skill_records.append({
                    'email': email,
                    'date': created_at,
                    'skill_ids': passed_skill_ids
                })
    
    # Batch fetch all skill names in parallel
    skill_id_to_name = get_all_skill_names(all_skill_ids)
    
    for record in assessment_skill_records:
        for skill_id in record['skill_ids']:
            skill_name = skill_id_to_name.get(skill_id, 'Unknown Skill')
            if skill_name != 'Unknown Skill':
                skill_acquisitions.append({
                    'source': 'Udacity Assessment',
                    'date': record['date'],
                    'email': record['email'],
                    'skill_id': skill_id,
                    'skill_name': skill_name,
                    'subject_id': None,
                    'subject_name': None,
                    'domain_id': None,
                    'domain_name': None
                })
    
    # Process Workera Assessments
    workera_data = fetch_workera_results()
    
    workera_data['created_at'] = pd.to_datetime(workera_data['created_at'])
    workera_latest = workera_data.sort_values('created_at', ascending=False).groupby(['user', 'domain_identifier']).first().reset_index()
    workera_passed = workera_latest[workera_latest['score'] > workera_latest['target_score']].copy()
    
    # Collect all unique Workera skills first
    all_workera_skills = set()
    for idx, row in workera_passed.iterrows():
        strong_skills = row['strong_skills']
        if strong_skills and isinstance(strong_skills, list):
            all_workera_skills.update(strong_skills)
    
    print(f"[Workera] Converting {len(all_workera_skills)} unique Workera skills to Udacity skills (parallel)")
    
    # Batch convert all Workera skills in parallel
    workera_skill_conversion_cache = batch_convert_skills_to_udacity(all_workera_skills)
    
    print(f"[Workera] Conversion complete")
    
    # Now process the records using the cache
    for idx, row in workera_passed.iterrows():
        email = row['user']
        date = row['created_at'].strftime('%Y-%m-%d')
        strong_skills = row['strong_skills']
        
        if not strong_skills or not isinstance(strong_skills, list):
            continue
        
        for workera_skill in strong_skills:
            udacity_skill = workera_skill_conversion_cache.get(workera_skill)
            
            if udacity_skill:
                skill_acquisitions.append({
                    'source': 'Workera Assessment',
                    'date': date,
                    'email': email,
                    'skill_id': None,
                    'skill_name': udacity_skill,
                    'subject_id': None,
                    'subject_name': None,
                    'domain_id': None,
                    'domain_name': None
                })
    
    # Create DataFrame
    df = pd.DataFrame(skill_acquisitions)
    
    if len(df) == 0:
        return df
    
    # Enrich with hierarchy information (parallel)
    unique_skills = [skill for skill in df['skill_name'].unique() if pd.notna(skill)]
    
    print(f"[Hierarchy] Fetching hierarchy for {len(unique_skills)} unique skills (parallel)")
    
    # Batch fetch all hierarchies in parallel
    hierarchy_cache = batch_get_skill_hierarchies(unique_skills)
    
    print(f"[Hierarchy] Enriching {len(df)} skill acquisition records")
    
    for idx, row in df.iterrows():
        skill_name = row['skill_name']
        if skill_name in hierarchy_cache and hierarchy_cache[skill_name]:
            hierarchy = hierarchy_cache[skill_name]
            df.at[idx, 'subject_id'] = hierarchy.get('subject_id')
            df.at[idx, 'subject_name'] = hierarchy.get('subject_name')
            df.at[idx, 'domain_id'] = hierarchy.get('domain_id')
            df.at[idx, 'domain_name'] = hierarchy.get('domain_name')
    
    # Fix Udacity Assessment skills that failed enrichment (parallel)
    udacity_assessment_missing = df[(df['source'] == 'Udacity Assessment') & (df['subject_name'].isna())]
    
    if len(udacity_assessment_missing) > 0:
        print(f"[Fix Enrichment] Processing {len(udacity_assessment_missing)} failed Udacity Assessment skills")
        
        # Get unique skills that need conversion
        unique_failed_skills = udacity_assessment_missing['skill_name'].unique()
        
        # Batch convert all failed skills in parallel
        skill_conversion_cache = batch_convert_skills_to_udacity(unique_failed_skills)
        
        # Collect new skill names that need hierarchy lookup
        new_skills_to_fetch = set()
        for old_skill_name, new_skill_name in skill_conversion_cache.items():
            if new_skill_name and new_skill_name != old_skill_name and new_skill_name not in hierarchy_cache:
                new_skills_to_fetch.add(new_skill_name)
        
        # Batch fetch hierarchies for new skills in parallel
        if new_skills_to_fetch:
            print(f"[Fix Enrichment] Fetching hierarchies for {len(new_skills_to_fetch)} converted skills")
            new_hierarchies = batch_get_skill_hierarchies(new_skills_to_fetch)
            hierarchy_cache.update(new_hierarchies)
        
        # Apply the conversions and hierarchies
        for idx, row in udacity_assessment_missing.iterrows():
            old_skill_name = row['skill_name']
            new_skill_name = skill_conversion_cache.get(old_skill_name)
            
            if new_skill_name and new_skill_name != old_skill_name:
                df.at[idx, 'skill_name'] = new_skill_name
                
                hierarchy = hierarchy_cache.get(new_skill_name)
                if hierarchy:
                    df.at[idx, 'subject_id'] = hierarchy.get('subject_id')
                    df.at[idx, 'subject_name'] = hierarchy.get('subject_name')
                    df.at[idx, 'domain_id'] = hierarchy.get('domain_id')
                    df.at[idx, 'domain_name'] = hierarchy.get('domain_name')
    
    df['date'] = pd.to_datetime(df['date'])
    
    return df


# =============================================================================
# PROGRAM SKILLS & ENROLLMENT FUNCTIONS
# =============================================================================

def get_program_skills(program_key):
    """
    Fetch skills taught by a program using the classroom-content GraphQL API.
    
    Args:
        program_key (str): The program key (e.g., 'nd001', 'cd13303')
        
    Returns:
        list: List of dicts with 'uri' and 'name' for each skill
    """
    headers = {
        "Authorization": f"Bearer {get_jwt_token()}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    query = f"""
    {{
      component(key: "{program_key}" locale: "en-us") {{
        id
        latest_release {{
          component {{
            title
            metadata {{
              teaches_skills {{
                uri
                name
              }}
            }}
          }}
        }}
      }}
    }}
    """
    
    try:
        payload = {"query": query}
        response = requests.post(CLASSROOM_CONTENT_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Check for GraphQL errors
        if data.get('errors'):
            print(f"GraphQL error for program {program_key}: {data['errors']}")
            return []
        
        component = data.get('data', {}).get('component')
        if not component:
            print(f"No component data found for program {program_key}")
            return []
        
        latest_release = component.get('latest_release')
        if not latest_release:
            print(f"No latest_release found for program {program_key}")
            return []
        
        component_data = latest_release.get('component', {})
        metadata = component_data.get('metadata', {})
        teaches_skills = metadata.get('teaches_skills', [])
        
        if not teaches_skills:
            print(f"No teaches_skills found for program {program_key}")
        
        return teaches_skills if teaches_skills else []
        
    except requests.exceptions.RequestException as e:
        print(f"Request error fetching skills for program {program_key}: {str(e)}")
        return []
    except Exception as e:
        print(f"Unexpected error fetching skills for program {program_key}: {str(e)}")
        return []


def get_skills_by_enrollments(emc_data):
    """
    Track skills by enrollments. For each enrollment, fetch the skills taught by that program.
    
    Args:
        emc_data: Response data from get_emc_content_data()
        
    Returns:
        pd.DataFrame: Columns [email, program_key, skill, date]
    """
    enrollments = []
    learners = emc_data.get('data', {}).get('company', {}).get('learners', {}).get('edges', [])
    
    print(f"[Enrollments] Processing {len(learners)} learners")
    
    for learner_edge in learners:
        node = learner_edge.get('node', {})
        email = node.get('email')
        roster = node.get('roster', [])
        
        for enrollment in roster:
            program_info = enrollment.get('programInfo', {})
            program_key = program_info.get('programKey')
            
            learner_activity = enrollment.get('learnerActivity', {})
            enrolled_at = learner_activity.get('enrolledAt', '')
            
            if email and program_key and enrolled_at:
                enrollment_date = enrolled_at.split('T')[0]
                enrollments.append({
                    'email': email,
                    'program_key': program_key,
                    'date': enrollment_date
                })
    
    print(f"[Enrollments] Found {len(enrollments)} total enrollments")
    
    unique_program_keys = list(set([e['program_key'] for e in enrollments]))
    print(f"[Enrollments] Fetching skills for {len(unique_program_keys)} unique programs (parallel)")
    
    program_skills_cache = {}
    programs_with_skills = 0
    
    # Parallel fetch program skills
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_program = {
            executor.submit(get_program_skills, program_key): program_key
            for program_key in unique_program_keys
        }
        
        for future in as_completed(future_to_program):
            program_key = future_to_program[future]
            try:
                skills = future.result()
                program_skills_cache[program_key] = skills
                if skills:
                    programs_with_skills += 1
            except Exception as e:
                print(f"Error fetching skills for program {program_key}: {str(e)}")
                program_skills_cache[program_key] = []
    
    print(f"[Enrollments] {programs_with_skills}/{len(unique_program_keys)} programs returned skills")
    
    skill_enrollment_records = []
    
    for enrollment in enrollments:
        program_key = enrollment['program_key']
        skills = program_skills_cache.get(program_key, [])
        
        if skills:
            for skill in skills:
                skill_name = skill.get('name', 'Unknown')
                skill_enrollment_records.append({
                    'email': enrollment['email'],
                    'program_key': program_key,
                    'skill': skill_name,
                    'date': enrollment['date']
                })
    
    print(f"[Enrollments] Created {len(skill_enrollment_records)} skill enrollment records")
    
    df = pd.DataFrame(skill_enrollment_records)
    
    if len(df) > 0:
        df['date'] = pd.to_datetime(df['date'])
    
    return df


def get_skills_by_graduations(emc_data):
    """
    Track skills by graduations. For each graduation, fetch the skills taught by that program.
    Only includes learners who have graduated (graduatedAt is not null).
    
    Args:
        emc_data: Response data from get_emc_content_data()
        
    Returns:
        pd.DataFrame: Columns [email, program_key, skill, date]
    """
    graduations = []
    learners = emc_data.get('data', {}).get('company', {}).get('learners', {}).get('edges', [])
    
    print(f"[Graduations] Processing {len(learners)} learners")
    
    for learner_edge in learners:
        node = learner_edge.get('node', {})
        email = node.get('email')
        roster = node.get('roster', [])
        
        for enrollment in roster:
            program_info = enrollment.get('programInfo', {})
            program_key = program_info.get('programKey')
            
            learner_activity = enrollment.get('learnerActivity', {})
            graduated_at = learner_activity.get('graduatedAt', '')
            
            if email and program_key and graduated_at:
                graduation_date = graduated_at.split('T')[0]
                graduations.append({
                    'email': email,
                    'program_key': program_key,
                    'date': graduation_date
                })
    
    print(f"[Graduations] Found {len(graduations)} total graduations")
    
    if len(graduations) == 0:
        print("[Graduations] No graduations found, returning empty DataFrame")
        return pd.DataFrame(columns=['email', 'program_key', 'skill', 'date'])
    
    unique_program_keys = list(set([g['program_key'] for g in graduations]))
    print(f"[Graduations] Fetching skills for {len(unique_program_keys)} unique programs (parallel)")
    
    program_skills_cache = {}
    programs_with_skills = 0
    
    # Parallel fetch program skills
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_program = {
            executor.submit(get_program_skills, program_key): program_key
            for program_key in unique_program_keys
        }
        
        for future in as_completed(future_to_program):
            program_key = future_to_program[future]
            try:
                skills = future.result()
                program_skills_cache[program_key] = skills
                if skills:
                    programs_with_skills += 1
            except Exception as e:
                print(f"Error fetching skills for program {program_key}: {str(e)}")
                program_skills_cache[program_key] = []
    
    print(f"[Graduations] {programs_with_skills}/{len(unique_program_keys)} programs returned skills")
    
    skill_graduation_records = []
    
    for graduation in graduations:
        program_key = graduation['program_key']
        skills = program_skills_cache.get(program_key, [])
        
        if skills:
            for skill in skills:
                skill_name = skill.get('name', 'Unknown')
                skill_graduation_records.append({
                    'email': graduation['email'],
                    'program_key': program_key,
                    'skill': skill_name,
                    'date': graduation['date']
                })
    
    print(f"[Graduations] Created {len(skill_graduation_records)} skill graduation records")
    
    df = pd.DataFrame(skill_graduation_records)
    
    if len(df) > 0:
        df['date'] = pd.to_datetime(df['date'])
    
    return df

