"""
Skills Challenge Analysis - Streamlit App
Analyzes Workera assessment data by mapping skills to the Udacity taxonomy (domain > subject > skill)
via the Udacity Skills API, then identifies skill gaps and coverage. No taxonomy file upload required.
"""

import streamlit as st
import pandas as pd
import numpy as np
import ast
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import plotly.express as px
import plotly.graph_objects as go

from settings import (
    WORKERA_DOMAINS_URL,
    WORKERA_BENCHMARKS_URL,
    WORKERA_SIGNALS_URL,
    WORKERA_SCORES_URL,
    WORKERA_SCORES_DETAIL_URL,
    UDACITY_WORKERA_API_KEYS_URL,
    SKILLS_SEARCH_URL,
    OPENAI_API_KEY,
)
from agent import run_conversation
from udacity_skills_mapping import (
    batch_convert_skills_to_udacity,
    batch_get_skill_hierarchies,
    build_taxonomy_from_hierarchies,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skills Challenge Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;400;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #12121a 50%, #0d0d14 100%);
    }
    
    .main .block-container {
        padding-top: 2rem;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        color: #f0f0f5 !important;
    }
    
    p, label {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Exclude Material Icon elements from font override */
    span:not([data-testid="stIconMaterial"]) {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Ensure Material Icons render correctly */
    [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded' !important;
        font-size: 24px;
        -webkit-font-smoothing: antialiased;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1a1a24 0%, #14141c 100%);
        border: 1px solid rgba(155, 0, 245, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 2.5rem;
        font-weight: 700;
        color: #9B00F5;
        text-shadow: 0 0 30px rgba(155, 0, 245, 0.5);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #8888a0;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.5rem;
    }
    
    .section-header {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.5rem;
        font-weight: 600;
        color: #f0f0f5;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(155, 0, 245, 0.4);
    }
    
    .stFileUploader > div {
        background: linear-gradient(145deg, #1a1a24 0%, #14141c 100%);
        border: 2px dashed rgba(155, 0, 245, 0.4);
        border-radius: 12px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #9B00F5 0%, #7a00c4 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(155, 0, 245, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 30px rgba(155, 0, 245, 0.6);
    }
    
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d14 0%, #0a0a10 100%);
        border-right: 1px solid rgba(155, 0, 245, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# API Functions
# ─────────────────────────────────────────────────────────────────────────────

def fetch_paginated_results(url: str, headers: dict, params: dict | None = None) -> list:
    """Fetch all paginated results from a Workera API endpoint."""
    all_results = []
    params = params or {}
    
    while url:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
        
        data = response.json()
        
        for item in data.get('data', []):
            all_results.append(item)
        
        if data.get('has_more'):
            url = data.get('next_page')
            params = {}  # next_page is full URL, don't re-apply limit
        else:
            url = None
    
    return all_results


def fetch_domains(api_key: str) -> pd.DataFrame:
    """Fetch domains from the Workera API."""
    url = WORKERA_DOMAINS_URL
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    
    domains = fetch_paginated_results(url, headers)
    
    domain_data = {
        'domain_id': [d.get('identifier') for d in domains],
        'title': [d.get('title') for d in domains],
        'description': [d.get('description') for d in domains],
        'is_signature': [d.get('is_signature_domain') for d in domains],
        'status': [d.get('status') for d in domains],
    }
    
    return pd.DataFrame(domain_data)


def fetch_benchmarks(api_key: str) -> list:
    """Fetch benchmarks from the Workera API."""
    url = WORKERA_BENCHMARKS_URL
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    return fetch_paginated_results(url, headers)


def fetch_signals(api_key: str) -> pd.DataFrame:
    """Fetch CAT assessment signals from the Workera API (legacy)."""
    url = WORKERA_SIGNALS_URL
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    
    signals = fetch_paginated_results(url, headers)
    
    records = []
    for signal in signals:
        user_data = signal.get('user', {})
        records.append({
            'domain_identifier': signal.get('domain_identifier'),
            'user_identifier': user_data.get('identifier'),
            'email': user_data.get('email'),
            'score': signal.get('score'),
            'target_score': signal.get('target_score'),
            'strong_skills': signal.get('strong_skills', []),
            'needs_improvement_skills': signal.get('needs_improvement_skills', []),
            'competency_model_identifier': signal.get('competency_model_identifier'),
            'created_at': signal.get('created_at'),
            'updated_at': signal.get('updated_at'),
        })
    
    return pd.DataFrame(records)


def fetch_scores(api_key: str) -> list:
    """Fetch high-level scores from the Workera API (no skill details)."""
    url = WORKERA_SCORES_URL
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    params = {"limit": 100}
    return fetch_paginated_results(url, headers, params=params)


def fetch_score_details(api_key: str, score_ids: list, progress_callback=None) -> list:
    """
    Fetch detailed score for each id (includes results.skill_ratings).
    progress_callback(current, total) is called after each request if provided.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    details = []
    for i, sid in enumerate(score_ids):
        url = WORKERA_SCORES_DETAIL_URL.format(id=sid)
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                details.append(resp.json())
            # Rate limit: back off if header indicates limit reached
            remaining = resp.headers.get("x-ratelimit-remaining")
            if remaining is not None and int(remaining) == 0:
                import time
                time.sleep(60)
        except Exception:
            pass
        if progress_callback and (i + 1) % 10 == 0:
            progress_callback(i + 1, len(score_ids))
    if progress_callback:
        progress_callback(len(score_ids), len(score_ids))
    return details


def _skill_ratings_to_lists(skill_ratings: list) -> tuple[list, list]:
    """Convert results.skill_ratings into strong_skills and needs_improvement_skills name lists."""
    strong = []
    needs = []
    for s in skill_ratings or []:
        name = s.get("name")
        if not name:
            continue
        level = (s.get("proficiency_level") or "").lower()
        if level == "strong":
            strong.append(name)
        elif level == "needs_improvement":
            needs.append(name)
    return strong, needs


def scores_to_signals_df(scores: list, domains_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a signals-shaped DataFrame from scores list (no skill details yet).
    Merges with domains_df for title and benchmark columns. strong_skills/needs_improvement_skills are empty.
    """
    if not scores:
        return pd.DataFrame(columns=[
            "identifier", "domain_identifier", "user_identifier", "email", "score",
            "strong_skills", "needs_improvement_skills", "created_at", "updated_at",
            "domain_id", "title", "benchmark_enterprise_avg", "benchmark_enterprise_75_perc",
        ])
    records = []
    for s in scores:
        user = s.get("user") or {}
        domain = s.get("domain") or {}
        records.append({
            "identifier": s.get("identifier"),
            "domain_identifier": domain.get("identifier"),
            "domain_name": domain.get("name"),
            "user_identifier": user.get("identifier"),
            "email": user.get("email"),
            "score": s.get("score"),
            "strong_skills": [],
            "needs_improvement_skills": [],
            "created_at": s.get("created_at"),
            "updated_at": s.get("updated_at"),
        })
    df = pd.DataFrame(records)
    if not domains_df.empty:
        df = df.merge(
            domains_df[["domain_id", "title", "benchmark_enterprise_avg", "benchmark_enterprise_75_perc"]],
            left_on="domain_identifier",
            right_on="domain_id",
            how="left",
        )
    if "title" not in df.columns:
        df["title"] = df.get("domain_name", "")
    else:
        df["title"] = df["title"].fillna(df.get("domain_name", ""))
    return df


def add_benchmark_scores(domains_df: pd.DataFrame, benchmarks: list) -> pd.DataFrame:
    """Add benchmark scores to domains DataFrame."""
    benchmark_lookup = {b['domain_identifier']: b for b in benchmarks}
    
    domains_df['benchmark_enterprise_avg'] = domains_df['domain_id'].apply(
        lambda x: benchmark_lookup.get(x, {}).get('enterprise_average_score')
    )
    domains_df['benchmark_enterprise_75_perc'] = domains_df['domain_id'].apply(
        lambda x: benchmark_lookup.get(x, {}).get('enterprise_percentile_75_score')
    )
    
    return domains_df


@st.cache_data(ttl=300)
def fetch_all_api_data(api_key: str):
    """Fetch all data from Workera API (cached for 5 minutes). Uses scores + scores/{id} flow."""
    domains_df = fetch_domains(api_key)
    benchmarks_raw = fetch_benchmarks(api_key)
    domains_df = add_benchmark_scores(domains_df, benchmarks_raw)
    scores = fetch_scores(api_key)
    signals_df = scores_to_signals_df(scores, domains_df)
    
    # Store raw benchmarks and scores list for later detail fetches
    benchmarks_df = pd.DataFrame(benchmarks_raw) if benchmarks_raw else pd.DataFrame()
    
    return domains_df, signals_df, benchmarks_df, scores


# ─────────────────────────────────────────────────────────────────────────────
# Workera API Key Fetch Function
# ─────────────────────────────────────────────────────────────────────────────

def get_workera_api_key(company_id: int, jwt_token: str) -> str:
    """
    Fetch a Workera API key for a given company_id.
    
    Args:
        company_id: The company ID to fetch the API key for
        jwt_token: A valid STAFF or SERVICE JWT for calling this endpoint
    
    Returns:
        str: The Workera API key
    """
    url = f"{UDACITY_WORKERA_API_KEYS_URL}/{company_id}"
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/json",
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch API key (status={response.status_code}): {response.text}"
        )
    
    data = response.json()
    # The API returns: {"id": "...", "company_id": ..., "value": "<API_KEY>", ...}
    if isinstance(data, dict):
        return data.get('value')
    return str(data)


# ─────────────────────────────────────────────────────────────────────────────
# Skills Search API Function
# ─────────────────────────────────────────────────────────────────────────────

def fetch_skills_recommendations(weak_skills: list, jwt_token: str) -> list:
    """
    Fetch content recommendations from the Udacity Skills Search API.
    
    Args:
        weak_skills: List of skill names to search for
        jwt_token: JWT token for authentication
    
    Returns:
        List of recommendation objects
    """
    url = SKILLS_SEARCH_URL
    headers = {
        'content-type': 'application/json',
        'Authorization': f'Bearer {jwt_token}',
    }
    
    payload = {
        "search": weak_skills,
        "searchField": "knowledge_component_desc",
    }
    
    try:
        import json
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def listify(x):
    """Convert string representation of list to actual list."""
    if isinstance(x, list):
        return x
    if x is None:
        return []
    try:
        if pd.isna(x):
            return []
    except (ValueError, TypeError):
        pass
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    return []


def explode_skills(df, column, taxonomy_skills, raw_to_udacity=None):
    """Explode skills column and merge with taxonomy. If raw_to_udacity is provided, map raw skill -> Udacity skill first."""
    frame = df[['workera_user_email', column]].explode(column).rename(columns={column: 'raw_skill'})
    if raw_to_udacity is not None:
        frame['Skill'] = frame['raw_skill'].map(raw_to_udacity)
        frame = frame.dropna(subset=['Skill'])
    else:
        frame = frame.rename(columns={'raw_skill': 'Skill'})
    return pd.merge(frame, taxonomy_skills, on='Skill', how='left').dropna(subset=['Topic'])


def render_metric_card(value, label):
    """Render a styled metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def create_subdomain_topic_heatmap(strong_df, weak_df, subdomain, taxonomy_skills):
    """
    Create a heatmap for a specific subdomain showing topic proficiency per user.
    Matches the exact style from the notebook.
    """
    # Calculate user-level topic scores
    strong_agg = strong_df.groupby(['Subdomain', 'Topic', 'workera_user_email']).size().reset_index()
    strong_agg.columns = ['Subdomain', 'Topic', 'workera_user_email', 'strong_count']
    
    weak_agg = weak_df.groupby(['Subdomain', 'Topic', 'workera_user_email']).size().reset_index()
    weak_agg.columns = ['Subdomain', 'Topic', 'workera_user_email', 'weak_count']
    
    merged = pd.merge(strong_agg, weak_agg, on=['Subdomain', 'Topic', 'workera_user_email'], how='outer').fillna(0)
    denom = merged['strong_count'] + merged['weak_count']
    merged['ratio_strong'] = np.where(denom == 0, np.nan, merged['strong_count'] / denom)
    
    # Filter for the specific subdomain
    merged = merged[merged['Subdomain'] == subdomain]
    
    if len(merged) == 0:
        return None
    
    # Turn into a pivot heatmap
    heatmap_df = merged.pivot_table(
        index="Topic",
        columns="workera_user_email",
        values="ratio_strong"
    )
    
    if heatmap_df.empty:
        return None
    
    # Re-sort the index based on the overall topic score
    topic_order = heatmap_df.sum(axis=1).sort_values(ascending=False).index
    heatmap_df = heatmap_df.loc[topic_order]
    
    # Re-sort learners based on overall score
    user_order = heatmap_df.mean(axis=0).sort_values(ascending=False).index
    heatmap_df = heatmap_df[user_order]
    
    # Create the heatmap with exact notebook styling
    fig, ax = plt.subplots(figsize=(16, max(5, len(heatmap_df) * 0.5)), dpi=150)
    
    # Custom color gradient matching notebook
    colors = [
        "white",
        "lightgrey",
        "#c0c2cf",
        "#beb2d1",
        "#bca2d3",
        "#ba92d5",
        "#b882d7",
        "#9B00F5",
    ]
    
    cmap = LinearSegmentedColormap.from_list('GrayToPurple', colors, N=256)
    cmap.set_bad(color='lightgrey')  # Grey for untested (NaN) skills
    
    sns.heatmap(
        heatmap_df,
        cmap=cmap,
        vmin=0, vmax=1,
        linewidths=0.1, linecolor='black',
        cbar_kws={'label': 'Strong Skill Ratio'},
        xticklabels=False,
        yticklabels=True,
        ax=ax
    )
    
    # Overlay cross-hatching for NaNs
    mask = heatmap_df.isna().to_numpy()
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                ax.add_patch(Rectangle(
                    (j, i), 1, 1,
                    fill=False,
                    hatch='///',
                    edgecolor='darkgrey',
                    linewidth=0
                ))
    
    ax.set_ylabel("Topic")
    ax.set_xlabel("User")
    ax.set_title(f"Strong Skill Ratio Heatmap - {subdomain}")
    
    plt.tight_layout()
    return fig


def create_subdomain_skill_heatmap(strong_df, weak_df, subdomain, taxonomy_skills):
    """
    Create a heatmap for a specific subdomain showing skill-level proficiency per user.
    Matches the exact style from the notebook.
    """
    # Calculate user-level skill scores
    strong_agg = strong_df.groupby(['Subdomain', 'Skill', 'workera_user_email']).size().reset_index()
    strong_agg.columns = ['Subdomain', 'Skill', 'workera_user_email', 'strong_count']
    
    weak_agg = weak_df.groupby(['Subdomain', 'Skill', 'workera_user_email']).size().reset_index()
    weak_agg.columns = ['Subdomain', 'Skill', 'workera_user_email', 'weak_count']
    
    merged = pd.merge(strong_agg, weak_agg, on=['Subdomain', 'Skill', 'workera_user_email'], how='outer').fillna(0)
    denom = merged['strong_count'] + merged['weak_count']
    merged['ratio_strong'] = np.where(denom == 0, np.nan, merged['strong_count'] / denom)
    
    # Filter for the specific subdomain
    merged = merged[merged['Subdomain'] == subdomain]
    
    if len(merged) == 0:
        return None
    
    # Turn into a pivot heatmap
    heatmap_df = merged.pivot_table(
        index="Skill",
        columns="workera_user_email",
        values="ratio_strong"
    )
    
    if heatmap_df.empty:
        return None
    
    # Re-sort the index based on the overall skill score
    skill_order = heatmap_df.sum(axis=1).sort_values(ascending=False).index
    heatmap_df = heatmap_df.loc[skill_order]
    
    # Re-sort learners based on overall score
    user_order = heatmap_df.mean(axis=0).sort_values(ascending=False).index
    heatmap_df = heatmap_df[user_order]
    
    # Create the heatmap with exact notebook styling
    fig, ax = plt.subplots(figsize=(16, max(5, len(heatmap_df) * 0.4)), dpi=150)
    
    # Simpler color gradient for skill-level (from notebook)
    cmap = LinearSegmentedColormap.from_list('GrayToBlue', ['white', 'lightgrey', '#9B00F5'])
    cmap.set_bad(color='lightgrey')
    
    sns.heatmap(
        heatmap_df,
        cmap=cmap,
        vmin=0, vmax=1,
        linewidths=0.1, linecolor='black',
        cbar_kws={'label': 'Strong Skill Ratio'},
        xticklabels=False,
        yticklabels=True,
        ax=ax
    )
    
    # Overlay cross-hatching for NaNs
    mask = heatmap_df.isna().to_numpy()
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                ax.add_patch(Rectangle(
                    (j, i), 1, 1,
                    fill=False,
                    hatch='///',
                    edgecolor='darkgrey',
                    linewidth=0
                ))
    
    ax.set_ylabel("Skill")
    ax.set_xlabel("User")
    ax.set_title(f"Strong Skill Ratio Heatmap - {subdomain}")
    
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #9B00F5 0%, #00f59b 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Skills Challenge Analysis
        </h1>
        <p style="color: #8888a0; font-size: 1.1rem;">
            Analyze Workera assessment data to identify skill gaps and learning opportunities
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        # Section 1: Connect to Workera API
        st.markdown("### Connect to Workera API")
        
        jwt_token = st.text_input(
            "Udacity JWT Token",
            type="password",
            help="Enter your Udacity JWT token"
        )
        
        company_id = st.text_input(
            "Company ID",
            value="",
            help="Enter the Workera company ID (e.g., 2122)"
        )
        
        fetch_button = st.button("Fetch Data from API", use_container_width=True)
        
        # Section 2: Run Analysis (only shown after data is fetched)
        if st.session_state.get('api_fetched'):
            st.markdown("---")
            st.markdown("### Run Analysis")
            
            signals_df = st.session_state['signals_df']
            available_assessments = sorted(signals_df['title'].dropna().unique().tolist())
            
            selected_assessment = st.selectbox(
                "Select Assessment",
                options=available_assessments,
                index=None,
                placeholder="Choose an assessment...",
                help="Filter data to a specific assessment"
            )
            
            run_analysis = st.button("Run Analysis", use_container_width=True, type="primary")
        else:
            selected_assessment = None
            run_analysis = False
        
    # Main content area - check for JWT and company ID first
    if not jwt_token or not company_id:
        st.info("Enter your JWT token and Company ID in the sidebar, then click 'Fetch Data from API'.")
        return
    
    # Fetch data when button is clicked
    if fetch_button:
        with st.spinner("Fetching Workera API key and data..."):
            try:
                # Clean inputs - strip whitespace and any non-ASCII characters
                clean_jwt = jwt_token.strip()
                clean_company_id = company_id.strip()
                
                # Remove any non-printable or special unicode characters from JWT
                clean_jwt = ''.join(c for c in clean_jwt if ord(c) < 128)
                
                # First, fetch the Workera API key using JWT and company ID
                api_key = get_workera_api_key(int(clean_company_id), clean_jwt)
                st.session_state['workera_api_key'] = api_key
                st.session_state['jwt_token'] = clean_jwt  # Store for Skills API calls
                
                # Now fetch the data using the API key (scores-based flow)
                domains_df, signals_df, benchmarks_df, scores = fetch_all_api_data(api_key)
                st.session_state['domains_df'] = domains_df
                st.session_state['signals_df'] = signals_df
                st.session_state['benchmarks_df'] = benchmarks_df
                st.session_state['scores'] = scores
                st.session_state['api_fetched'] = True
                # Clear any previous assessment selection when new data is fetched
                if 'selected_assessment' in st.session_state:
                    del st.session_state['selected_assessment']
                if 'analysis_run' in st.session_state:
                    del st.session_state['analysis_run']
                st.rerun()
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return
    
    if 'api_fetched' not in st.session_state or not st.session_state.get('api_fetched'):
        st.info("👆 Click 'Fetch Data from API' to load assessment data.")
        return
    
    domains_df = st.session_state['domains_df']
    signals_df = st.session_state['signals_df']
    benchmarks_df = st.session_state.get('benchmarks_df', pd.DataFrame())
    
    # Check if assessment is selected
    if selected_assessment is None:
        st.info("👈 Select an assessment from the sidebar to continue.")
        return
    
    if not run_analysis and 'analysis_run' not in st.session_state:
        return
    
    if run_analysis:
        st.session_state['analysis_run'] = True
        st.session_state['selected_assessment'] = selected_assessment
    
    selected_assessment = st.session_state.get('selected_assessment', selected_assessment)
    
    with st.spinner("🔄 Processing data..."):
        try:
            # Filter scores/signals for selected assessment
            filtered_signals = signals_df[signals_df['title'] == selected_assessment].copy()
            
            if len(filtered_signals) == 0:
                st.warning(f"No data found for assessment: {selected_assessment}")
                return
            
            # Fetch score details (scores/{id}) to get skill_ratings, then reshape to strong_skills / needs_improvement_skills
            score_ids = filtered_signals['identifier'].dropna().astype(str).tolist()
            if not score_ids:
                st.warning("No score identifiers to fetch details for.")
                return
            api_key = st.session_state.get('workera_api_key')
            if not api_key:
                st.error("Workera API key not found. Re-fetch data from the API.")
                return
            progress_placeholder = st.empty()
            def _progress(current, total):
                if total:
                    progress_placeholder.progress(min(1.0, current / total), text=f"Fetching score details {current}/{total}...")
            details = fetch_score_details(api_key, score_ids, progress_callback=_progress)
            progress_placeholder.empty()
            detail_lookup = {}
            for d in details:
                sid = d.get('identifier')
                if sid is None:
                    continue
                ratings = (d.get('results') or {}).get('skill_ratings') or []
                strong, needs = _skill_ratings_to_lists(ratings)
                detail_lookup[str(sid)] = (strong, needs)
            filtered_signals['strong_skills'] = filtered_signals['identifier'].astype(str).map(
                lambda id: detail_lookup.get(id, ([], []))[0]
            )
            filtered_signals['needs_improvement_skills'] = filtered_signals['identifier'].astype(str).map(
                lambda id: detail_lookup.get(id, ([], []))[1]
            )
            
            # Prepare the attempts dataframe
            attempts_df = filtered_signals.copy()
            attempts_df = attempts_df.rename(columns={'email': 'workera_user_email'})
            
            # Filter to keep only each user's FIRST attempt (earliest created_at)
            if 'created_at' in attempts_df.columns:
                attempts_df['created_at'] = pd.to_datetime(attempts_df['created_at'])
                attempts_df = attempts_df.sort_values('created_at').groupby('workera_user_email').first().reset_index()
            elif 'updated_at' in attempts_df.columns:
                attempts_df['updated_at'] = pd.to_datetime(attempts_df['updated_at'])
                attempts_df = attempts_df.sort_values('updated_at').groupby('workera_user_email').first().reset_index()
            
            # Ensure skills are lists
            attempts_df['needs_improvement_skills'] = attempts_df['needs_improvement_skills'].apply(
                lambda x: x if isinstance(x, list) else (x if x is not None else [])
            )
            attempts_df['strong_skills'] = attempts_df['strong_skills'].apply(
                lambda x: x if isinstance(x, list) else (x if x is not None else [])
            )
            
            # Add benchmark columns
            if 'benchmark_enterprise_avg' in attempts_df.columns:
                attempts_df['enterprise_average_score'] = attempts_df['benchmark_enterprise_avg']
            if 'benchmark_enterprise_75_perc' in attempts_df.columns:
                attempts_df['enterprise_percentile_75_score'] = attempts_df['benchmark_enterprise_75_perc']
            
            # Build taxonomy from Udacity Skills API: raw skill -> Udacity skill -> subject/domain
            jwt_token = st.session_state.get('jwt_token')
            if not jwt_token:
                st.error("JWT token is required to map skills to the Udacity taxonomy. Re-fetch data from the API.")
                return
            all_raw_skills = []
            for _, row in attempts_df.iterrows():
                all_raw_skills.extend(row.get('strong_skills') or [])
                all_raw_skills.extend(row.get('needs_improvement_skills') or [])
            unique_raw = list(dict.fromkeys(s for s in all_raw_skills if s))
            raw_to_udacity = batch_convert_skills_to_udacity(unique_raw, jwt_token)
            unique_udacity = list(dict.fromkeys(v for v in raw_to_udacity.values() if v))
            hierarchies = batch_get_skill_hierarchies(unique_udacity, jwt_token)
            taxonomy_rows = build_taxonomy_from_hierarchies(hierarchies)
            taxonomy_skills = pd.DataFrame(taxonomy_rows)
            if taxonomy_skills.empty:
                taxonomy_skills = pd.DataFrame(columns=['Skill', 'Topic', 'Subdomain'])
            
            # Create exploded dataframes (map raw -> Udacity skill, then merge with taxonomy)
            strong_df = explode_skills(attempts_df, 'strong_skills', taxonomy_skills, raw_to_udacity=raw_to_udacity)
            weak_df = explode_skills(attempts_df, 'needs_improvement_skills', taxonomy_skills, raw_to_udacity=raw_to_udacity)
            
            # Store in session state for AI assistant
            st.session_state['current_attempts_df'] = attempts_df
            st.session_state['current_strong_df'] = strong_df
            st.session_state['current_weak_df'] = weak_df
            st.session_state['current_taxonomy'] = taxonomy_skills
            st.session_state['current_assessment'] = selected_assessment
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return
    
    # Display selected assessment
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(155, 0, 245, 0.15) 0%, rgba(155, 0, 245, 0.05) 100%); 
                border: 1px solid rgba(155, 0, 245, 0.3); border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem;">
        <span style="color: #8888a0;">Analyzing:</span> 
        <strong style="color: #9B00F5; font-size: 1.2rem;">{selected_assessment}</strong>
        <span style="color: #8888a0; font-size: 0.9rem; margin-left: 1rem;">(First attempt per user only)</span>
    </div>
    """, unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────────────────────────
    # Summary Metric Cards (matching notebook style)
    # ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Summary Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    avg_score = round(attempts_df['score'].mean()) if not attempts_df['score'].isna().all() else 0
    num_users = len(attempts_df)
    
    with col1:
        st.markdown(render_metric_card(avg_score, "Customer Average Score"), unsafe_allow_html=True)
    with col2:
        st.markdown(render_metric_card(num_users, "# of Users"), unsafe_allow_html=True)
    
    # Benchmarks (from benchmarks API endpoint, joined on domain_identifier)
    ent_avg = attempts_df['enterprise_average_score'].max() if 'enterprise_average_score' in attempts_df.columns else np.nan
    ent_75 = attempts_df['enterprise_percentile_75_score'].max() if 'enterprise_percentile_75_score' in attempts_df.columns else np.nan
    
    with col3:
        if pd.notna(ent_avg):
            st.markdown(render_metric_card(f"{ent_avg:.0f}", "Enterprise Average"), unsafe_allow_html=True)
        else:
            st.markdown(render_metric_card("N/A", "Enterprise Average"), unsafe_allow_html=True)
    
    with col4:
        if pd.notna(ent_75):
            st.markdown(render_metric_card(f"{ent_75:.0f}", "Enterprise 75th %ile"), unsafe_allow_html=True)
        else:
            st.markdown(render_metric_card("N/A", "Enterprise 75th %ile"), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ─────────────────────────────────────────────────────────────────
    # Coverage Summary Tables
    # ─────────────────────────────────────────────────────────────────
    with st.expander("Coverage Summary Tables", expanded=True):
        
        # Subdomain Coverage
        st.markdown("#### Subdomain Coverage")
        
        if len(weak_df) > 0 or len(strong_df) > 0:
            weak_subdomain_counts = weak_df.Subdomain.value_counts().reset_index() if len(weak_df) > 0 else pd.DataFrame(columns=['Subdomain', 'count'])
            weak_subdomain_counts.columns = ['Subdomain', 'weak_skill_count']
            
            strong_subdomain_counts = strong_df.Subdomain.value_counts().reset_index() if len(strong_df) > 0 else pd.DataFrame(columns=['Subdomain', 'count'])
            strong_subdomain_counts.columns = ['Subdomain', 'strong_skill_count']
            
            subdomain_summary = pd.merge(weak_subdomain_counts, strong_subdomain_counts, on='Subdomain', how='outer').fillna(0)
            subdomain_summary['subdomain_coverage_%'] = (
                subdomain_summary['strong_skill_count'] / 
                (subdomain_summary['strong_skill_count'] + subdomain_summary['weak_skill_count']) * 100
            ).round(1)
            subdomain_summary = subdomain_summary[['Subdomain', 'weak_skill_count', 'strong_skill_count', 'subdomain_coverage_%']]
            subdomain_summary = subdomain_summary.sort_values(by='subdomain_coverage_%', ascending=False)
            
            st.dataframe(subdomain_summary, hide_index=True, use_container_width=True)
        else:
            st.warning("No skill data available for subdomain coverage analysis.")
        
        st.markdown("---")
        
        # Topic Coverage
        st.markdown("#### Topic Coverage")
        
        if len(weak_df) > 0 or len(strong_df) > 0:
            weak_topic_counts = weak_df.Topic.value_counts().reset_index() if len(weak_df) > 0 else pd.DataFrame(columns=['Topic', 'count'])
            weak_topic_counts.columns = ['Topic', 'weak_skill_count']
            
            strong_topic_counts = strong_df.Topic.value_counts().reset_index() if len(strong_df) > 0 else pd.DataFrame(columns=['Topic', 'count'])
            strong_topic_counts.columns = ['Topic', 'strong_skill_count']
            
            topic_summary = pd.merge(weak_topic_counts, strong_topic_counts, on='Topic', how='outer').fillna(0)
            topic_summary['topic_coverage_%'] = (
                topic_summary['strong_skill_count'] / 
                (topic_summary['strong_skill_count'] + topic_summary['weak_skill_count']) * 100
            ).round(1)
            topic_summary = pd.merge(topic_summary, taxonomy_skills[['Topic', 'Subdomain']].drop_duplicates(), on='Topic', how='left')
            topic_summary = topic_summary[['Subdomain', 'Topic', 'weak_skill_count', 'strong_skill_count', 'topic_coverage_%']]
            topic_summary = topic_summary.sort_values(by=['Subdomain', 'topic_coverage_%'], ascending=[True, False])
            
            st.dataframe(topic_summary, hide_index=True, use_container_width=True)
        else:
            st.warning("No skill data available for topic coverage analysis.")
    
    # ─────────────────────────────────────────────────────────────────
    # Subdomain-Topic Heatmaps (matching notebook exactly)
    # ─────────────────────────────────────────────────────────────────
    with st.expander("🔥 Subdomain-Topic Heatmaps", expanded=False):
        if len(strong_df) > 0 or len(weak_df) > 0:
            subdomains = taxonomy_skills['Subdomain'].unique()
            
            for subdomain in subdomains:
                st.markdown(f"**{subdomain}**")
                
                fig = create_subdomain_topic_heatmap(strong_df, weak_df, subdomain, taxonomy_skills)
                
                if fig is not None:
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info(f"No data available for {subdomain}")
                
                st.markdown("---")
        else:
            st.warning("No skill data available for heatmap analysis.")
    
    # ─────────────────────────────────────────────────────────────────
    # Subdomain-Skill Heatmaps (matching notebook exactly)
    # ─────────────────────────────────────────────────────────────────
    with st.expander("🔬 Subdomain-Skill Heatmaps", expanded=False):
        if len(strong_df) > 0 or len(weak_df) > 0:
            subdomains = taxonomy_skills['Subdomain'].unique()
            
            for subdomain in subdomains:
                st.markdown(f"**{subdomain}**")
                
                fig = create_subdomain_skill_heatmap(strong_df, weak_df, subdomain, taxonomy_skills)
                
                if fig is not None:
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info(f"No data available for {subdomain}")
                
                st.markdown("---")
        else:
            st.warning("No skill data available for skill-level heatmap analysis.")
    
    # ─────────────────────────────────────────────────────────────────
    # Experimental Analytics
    # ─────────────────────────────────────────────────────────────────
    with st.expander("🧪 Experimental Analytics", expanded=False):
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(245, 160, 0, 0.15) 0%, rgba(245, 160, 0, 0.05) 100%); 
                    border: 1px solid rgba(245, 160, 0, 0.3); border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;">
            <span style="color: #f5a000;">Experimental:</span> 
            <span style="color: #8888a0;">These analytics are exploratory and may evolve.</span>
        </div>
        """, unsafe_allow_html=True)
        
        # ─────────────────────────────────────────────────────────────
        # 1. Score Distribution Histogram
        # ─────────────────────────────────────────────────────────────
        st.markdown("#### Score Distribution")
        
        if len(attempts_df) > 0 and not attempts_df['score'].isna().all():
            scores = attempts_df['score'].dropna()
            avg_score = scores.mean()
            max_score = max(300, scores.max() + 20)
            
            # Create Plotly histogram with skill level bands
            fig_hist = go.Figure()
            
            # Add background bands for skill levels
            fig_hist.add_vrect(x0=0, x1=100, fillcolor='#4A216B', opacity=0.15, 
                             layer='below', line_width=0,
                             annotation_text='Beginner', annotation_position='top left',
                             annotation=dict(font_size=10, font_color='#8888a0'))
            fig_hist.add_vrect(x0=100, x1=200, fillcolor='#C353A5', opacity=0.15,
                             layer='below', line_width=0,
                             annotation_text='Developing', annotation_position='top left',
                             annotation=dict(font_size=10, font_color='#8888a0'))
            fig_hist.add_vrect(x0=200, x1=300, fillcolor='#E79859', opacity=0.15,
                             layer='below', line_width=0,
                             annotation_text='Advanced', annotation_position='top left',
                             annotation=dict(font_size=10, font_color='#8888a0'))
            
            # Add histogram
            fig_hist.add_trace(go.Histogram(
                x=scores,
                nbinsx=20,
                marker_color='#9B00F5',
                opacity=0.8,
                name='Score Distribution',
                hovertemplate='Score: %{x}<br>Users: %{y}<extra></extra>'
            ))
            
            # Add vertical lines for key metrics
            fig_hist.add_vline(x=avg_score, line_dash='dash', line_color='#ff4444', line_width=2,
                              annotation_text=f'Avg: {avg_score:.0f}', annotation_position='top',
                              annotation=dict(font_color='#ff4444'))
            
            # Enterprise benchmarks if available
            if 'enterprise_average_score' in attempts_df.columns:
                ent_avg = attempts_df['enterprise_average_score'].max()
                if pd.notna(ent_avg):
                    fig_hist.add_vline(x=ent_avg, line_dash='dash', line_color='#ffffff', line_width=2,
                                      annotation_text=f'Ent Avg: {ent_avg:.0f}', annotation_position='top',
                                      annotation=dict(font_color='#ffffff'))
            
            if 'enterprise_percentile_75_score' in attempts_df.columns:
                ent_75 = attempts_df['enterprise_percentile_75_score'].max()
                if pd.notna(ent_75):
                    fig_hist.add_vline(x=ent_75, line_dash='dash', line_color='#00a0f5', line_width=2,
                                      annotation_text=f'Ent 75th: {ent_75:.0f}', annotation_position='top',
                                      annotation=dict(font_color='#00a0f5'))
            
            fig_hist.update_layout(
                xaxis_title='Score',
                yaxis_title='Number of Users',
                title=dict(text='Score Distribution with Skill Level Bands', font=dict(size=14, color='#f0f0f5')),
                xaxis=dict(range=[0, max_score], gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f0f0f5'),
                height=400,
                margin=dict(t=50, l=50, r=30, b=50),
                showlegend=False
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No score data available for distribution chart.")
        
        st.markdown("---")
        
        # ─────────────────────────────────────────────────────────────
        # 2. Topic Coverage Radar/Spider Chart
        # ─────────────────────────────────────────────────────────────
        st.markdown("#### Topic Coverage Radar")
        st.markdown("*Quick view of organizational strengths vs. gaps across all topics. Each shape represents a subdomain.*")
        
        if len(weak_df) > 0 or len(strong_df) > 0:
            # Calculate coverage per topic with subdomain info
            all_skills_combined = pd.concat([weak_df, strong_df]) if len(weak_df) > 0 and len(strong_df) > 0 else (weak_df if len(weak_df) > 0 else strong_df)
            topic_subdomain_map = all_skills_combined.groupby('Topic')['Subdomain'].first().to_dict()
            
            weak_counts = weak_df.Topic.value_counts() if len(weak_df) > 0 else pd.Series()
            strong_counts = strong_df.Topic.value_counts() if len(strong_df) > 0 else pd.Series()
            
            all_topics = list(set(weak_counts.index.tolist() + strong_counts.index.tolist()))
            
            radar_data = []
            for topic in all_topics:
                weak = weak_counts.get(topic, 0)
                strong = strong_counts.get(topic, 0)
                coverage = strong / (strong + weak) if (strong + weak) > 0 else 0
                subdomain = topic_subdomain_map.get(topic, 'Unknown')
                radar_data.append({'Topic': topic, 'Coverage': coverage, 'Subdomain': subdomain})
            
            radar_df = pd.DataFrame(radar_data)
            
            # Sort by Subdomain first, then by Coverage within each subdomain
            # This groups topics by subdomain so each subdomain's shape is contiguous
            radar_df = radar_df.sort_values(['Subdomain', 'Coverage'], ascending=[True, False])
            
            # Limit to top 18 topics for readability (but keep subdomain grouping)
            if len(radar_df) > 18:
                # Keep at least some topics from each subdomain
                radar_df = radar_df.groupby('Subdomain').head(6).reset_index(drop=True)
                if len(radar_df) > 18:
                    radar_df = radar_df.head(18)
            
            if len(radar_df) >= 3:  # Need at least 3 points for radar
                # Define subdomain colors (same as sunburst)
                color_palette = [
                    '#9B00F5', '#00f59b', '#f5a000', '#00a0f5', '#f50050',
                    '#50f500', '#f500f5', '#00f5f5', '#f5f500', '#5000f5',
                    '#f55000', '#00f550'
                ]
                unique_subdomains = radar_df['Subdomain'].unique()
                subdomain_color_map = {sd: color_palette[i % len(color_palette)] for i, sd in enumerate(unique_subdomains)}
                
                # Get all topics ordered by subdomain (topics from same subdomain are adjacent)
                all_topic_names = radar_df['Topic'].tolist()
                
                fig_radar = go.Figure()
                
                # Create a separate trace (shape) for each subdomain
                for subdomain in unique_subdomains:
                    subdomain_data = radar_df[radar_df['Subdomain'] == subdomain]
                    color = subdomain_color_map[subdomain]
                    
                    # For each subdomain, we need values for ALL topics (0 for topics not in this subdomain)
                    # Since topics are now ordered by subdomain, each subdomain's non-zero values are contiguous
                    values_for_subdomain = []
                    for topic in all_topic_names:
                        topic_row = subdomain_data[subdomain_data['Topic'] == topic]
                        if len(topic_row) > 0:
                            values_for_subdomain.append(topic_row['Coverage'].values[0])
                        else:
                            values_for_subdomain.append(0)
                    
                    # Parse hex color for rgba
                    hex_color = color.lstrip('#')
                    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values_for_subdomain,
                        theta=all_topic_names,
                        fill='toself',
                        fillcolor=f'rgba({r}, {g}, {b}, 0.25)',
                        line=dict(color=color, width=2),
                        marker=dict(size=8, color=color),
                        name=subdomain[:30] + ('...' if len(subdomain) > 30 else ''),
                        hovertemplate='<b>%{theta}</b><br>Subdomain: ' + subdomain + '<br>Coverage: %{r:.0%}<extra></extra>'
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1],
                            tickformat='.0%',
                            gridcolor='rgba(255,255,255,0.2)',
                            linecolor='rgba(255,255,255,0.2)'
                        ),
                        angularaxis=dict(
                            gridcolor='rgba(255,255,255,0.2)',
                            linecolor='rgba(255,255,255,0.2)'
                        ),
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    showlegend=True,
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=-0.2,
                        xanchor='center',
                        x=0.5,
                        font=dict(size=10)
                    ),
                    height=600,
                    margin=dict(t=50, l=80, r=80, b=100),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f0f0f5', size=10),
                    title=dict(
                        text='Topic Coverage by Subdomain (0% = All Weak, 100% = All Strong)',
                        font=dict(size=14, color='#f0f0f5'),
                        x=0.5
                    )
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.info("Need at least 3 topics for radar chart.")
        else:
            st.info("No skill data available for radar chart.")
        
        st.markdown("---")
        
        # ─────────────────────────────────────────────────────────────
        # 4. Skills Sunburst Chart
        # ─────────────────────────────────────────────────────────────
        st.markdown("#### Skills Hierarchy Sunburst")
        st.markdown("*Hierarchical view: Domain → Subdomain → Topic → Skill. Saturation indicates strength (vivid = stronger, gray = weaker).*")
        
        if len(weak_df) > 0 or len(strong_df) > 0:
            # Build hierarchical data with scores
            all_skills_data = []
            
            # Combine strong and weak skill data
            for _, row in strong_df.iterrows():
                all_skills_data.append({
                    'Skill': row['Skill'],
                    'Topic': row['Topic'],
                    'Subdomain': row['Subdomain'],
                    'is_strong': 1
                })
            
            for _, row in weak_df.iterrows():
                all_skills_data.append({
                    'Skill': row['Skill'],
                    'Topic': row['Topic'],
                    'Subdomain': row['Subdomain'],
                    'is_strong': 0
                })
            
            if all_skills_data:
                skills_hierarchy_df = pd.DataFrame(all_skills_data)
                
                # Calculate average score per skill
                skill_scores = skills_hierarchy_df.groupby(['Subdomain', 'Topic', 'Skill']).agg(
                    score=('is_strong', 'mean'),
                    count=('is_strong', 'count')
                ).reset_index()
                
                # Add Domain column for the hierarchy
                skill_scores['Domain'] = selected_assessment
                
                # Define colors for subdomains
                color_palette = [
                    '#9B00F5', '#00f59b', '#f5a000', '#00a0f5', '#f50050',
                    '#50f500', '#f500f5', '#00f5f5', '#f5f500', '#5000f5',
                    '#f55000', '#00f550'
                ]
                unique_subdomains = skill_scores['Subdomain'].unique()
                subdomain_color_map = {sd: color_palette[i % len(color_palette)] for i, sd in enumerate(unique_subdomains)}
                
                # Map subdomain colors to each skill row
                skill_scores['base_color'] = skill_scores['Subdomain'].map(subdomain_color_map)
                
                # Create RGBA color with opacity based on score
                def score_to_rgba(row):
                    hex_color = row['base_color'].lstrip('#')
                    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                    opacity = max(0.3, row['score'])  # Min 30% for visibility
                    return f'rgba({r}, {g}, {b}, {opacity})'
                
                skill_scores['color'] = skill_scores.apply(score_to_rgba, axis=1)
                
                # Create sunburst using plotly express
                fig_sunburst = px.sunburst(
                    skill_scores,
                    path=['Domain', 'Subdomain', 'Topic', 'Skill'],
                    values='count',
                    color='Subdomain',
                    color_discrete_map=subdomain_color_map
                )
                
                # Build color array and custom data for hover with aggregated scores
                def build_colors_and_hover(fig, skill_scores, subdomain_color_map):
                    import colorsys
                    
                    def hex_to_hsl(hex_color):
                        """Convert hex to HSL."""
                        hex_color = hex_color.lstrip('#')
                        r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
                        h, l, s = colorsys.rgb_to_hls(r, g, b)
                        return h, s, l
                    
                    def hsl_to_rgb(h, s, l):
                        """Convert HSL to RGB (0-255)."""
                        r, g, b = colorsys.hls_to_rgb(h, l, s)
                        return int(r * 255), int(g * 255), int(b * 255)
                    
                    def get_color_with_saturation(hex_color, score):
                        """Adjust saturation based on score. Low score = grayish, high score = vivid."""
                        h, s, l = hex_to_hsl(hex_color)
                        # Scale saturation: min 10% saturation for weak, up to full saturation for strong
                        adjusted_s = 0.1 + (score * 0.9)  # Range: 0.1 to 1.0
                        r, g, b = hsl_to_rgb(h, adjusted_s, l)
                        return f'rgb({r}, {g}, {b})'
                    
                    ids = fig.data[0].ids
                    colors = []
                    hover_scores = []
                    hover_counts = []
                    
                    for id_val in ids:
                        if id_val is None:
                            colors.append('rgb(128, 128, 128)')
                            hover_scores.append(0)
                            hover_counts.append(0)
                            continue
                            
                        parts = id_val.split('/')
                        
                        if len(parts) == 1:
                            # Domain level - gray with brightness based on score
                            avg_score = skill_scores['score'].mean()
                            total_count = skill_scores['count'].sum()
                            gray_val = int(80 + (avg_score * 48))  # Range 80-128
                            colors.append(f'rgb({gray_val}, {gray_val}, {gray_val})')
                            hover_scores.append(avg_score)
                            hover_counts.append(total_count)
                        elif len(parts) == 2:
                            # Subdomain level
                            subdomain = parts[1]
                            subset = skill_scores[skill_scores['Subdomain'] == subdomain]
                            avg_score = subset['score'].mean() if len(subset) > 0 else 0.5
                            total_count = subset['count'].sum() if len(subset) > 0 else 0
                            hex_color = subdomain_color_map.get(subdomain, '#9B00F5')
                            colors.append(get_color_with_saturation(hex_color, avg_score))
                            hover_scores.append(avg_score)
                            hover_counts.append(total_count)
                        elif len(parts) == 3:
                            # Topic level
                            subdomain = parts[1]
                            topic = parts[2]
                            subset = skill_scores[(skill_scores['Subdomain'] == subdomain) & (skill_scores['Topic'] == topic)]
                            avg_score = subset['score'].mean() if len(subset) > 0 else 0.5
                            total_count = subset['count'].sum() if len(subset) > 0 else 0
                            hex_color = subdomain_color_map.get(subdomain, '#9B00F5')
                            colors.append(get_color_with_saturation(hex_color, avg_score))
                            hover_scores.append(avg_score)
                            hover_counts.append(total_count)
                        elif len(parts) == 4:
                            # Skill level
                            subdomain = parts[1]
                            topic = parts[2]
                            skill = parts[3]
                            subset = skill_scores[(skill_scores['Subdomain'] == subdomain) & 
                                                  (skill_scores['Topic'] == topic) & 
                                                  (skill_scores['Skill'] == skill)]
                            avg_score = subset['score'].mean() if len(subset) > 0 else 0.5
                            total_count = subset['count'].sum() if len(subset) > 0 else 0
                            hex_color = subdomain_color_map.get(subdomain, '#9B00F5')
                            colors.append(get_color_with_saturation(hex_color, avg_score))
                            hover_scores.append(avg_score)
                            hover_counts.append(total_count)
                        else:
                            colors.append('rgb(128, 128, 128)')
                            hover_scores.append(0)
                            hover_counts.append(0)
                    
                    return colors, hover_scores, hover_counts
                
                # Apply custom colors and hover data
                custom_colors, hover_scores, hover_counts = build_colors_and_hover(fig_sunburst, skill_scores, subdomain_color_map)
                
                fig_sunburst.update_traces(
                    marker=dict(colors=custom_colors, line=dict(color='white', width=0.5)),
                    customdata=list(zip(hover_scores, hover_counts)),
                    hovertemplate='<b>%{label}</b><br>' +
                                  'Strength: %{customdata[0]:.0%}<br>' +
                                  'Skill Signals: %{customdata[1]}<br>' +
                                  '<extra></extra>'
                )
                
                fig_sunburst.update_layout(
                    margin=dict(t=10, l=0, r=0, b=10),
                    height=650,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#f0f0f5', size=10)
                )
                
                st.plotly_chart(fig_sunburst, use_container_width=True)
                
                # Legend for subdomain colors
                st.markdown("**Subdomain Colors** *(saturation = strength)*:")
                legend_cols = st.columns(min(4, len(unique_subdomains)))
                for i, sd in enumerate(unique_subdomains):
                    with legend_cols[i % 4]:
                        display_name = f'{sd[:30]}...' if len(sd) > 30 else sd
                        st.markdown(f'<span style="color:{subdomain_color_map[sd]}; font-size: 20px;">●</span> {display_name}', unsafe_allow_html=True)
            else:
                st.info("No skill data available for sunburst chart.")
        else:
            st.info("No skill data available for sunburst chart.")
        
    
    # ─────────────────────────────────────────────────────────────────
    # Content Recommendations (via Skills Search API)
    # ─────────────────────────────────────────────────────────────────
    with st.expander("Recommended Content", expanded=False):
        # Use the JWT token stored in session state
        skills_jwt = st.session_state.get('jwt_token')
        
        if not skills_jwt:
            st.info("JWT token required for content recommendations. Re-fetch data if needed.")
        elif len(attempts_df) == 0:
            st.warning("No user data available for recommendations.")
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(0, 245, 155, 0.15) 0%, rgba(0, 245, 155, 0.05) 100%); 
                        border: 1px solid rgba(0, 245, 155, 0.3); border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;">
                <span style="color: #00f59b;">Content Recommendations:</span> 
                <span style="color: #8888a0;">Based on weak skills, matched to Udacity content catalog.</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Fetch recommendations for all users
            if st.button("Generate Recommendations", use_container_width=True):
                with st.spinner("Fetching recommendations from Skills API..."):
                    from collections import Counter
                    import json
                    
                    all_recommendations = []
                    learner_results = []
                    
                    progress_bar = st.progress(0)
                    
                    for idx, (_, row) in enumerate(attempts_df.iterrows()):
                        weak_skills = row['needs_improvement_skills']
                        if weak_skills and len(weak_skills) > 0:
                            recs = fetch_skills_recommendations(weak_skills, skills_jwt)
                            
                            # Store for this learner
                            if recs:
                                # Extract parent_keys and find most common
                                parent_keys = [
                                    item['search']['metadata'].get('parent_key')
                                    for item in recs if item.get('search')
                                ]
                                
                                parent_key_counts = Counter(parent_keys)
                                most_common_key, count = parent_key_counts.most_common(1)[0] if parent_key_counts else (None, 0)
                                
                                # Get parent title for most common key
                                parent_title = None
                                for item in recs:
                                    if item.get('search') and item['search']['metadata'].get('parent_key') == most_common_key:
                                        parent_title = item['search']['metadata'].get('parent_title')
                                        break
                                
                                learner_results.append({
                                    'User': row['workera_user_email'],
                                    'Score': row['score'],
                                    'Weak Skills': len(weak_skills),
                                    'Top Recommendation': parent_title or most_common_key,
                                    'Recommendation Key': most_common_key,
                                    'Match Count': count
                                })
                                
                                # Collect all content recommendations
                                for item in recs:
                                    if item.get('search'):
                                        all_recommendations.append({
                                            'parent_key': item['search']['metadata'].get('parent_key'),
                                            'parent_title': item['search']['metadata'].get('parent_title'),
                                            'lesson_title': item['search']['metadata'].get('lesson_title'),
                                            'skill': item['search'].get('content'),
                                            'user': row['workera_user_email']
                                        })
                        
                        progress_bar.progress((idx + 1) / len(attempts_df))
                    
                    progress_bar.empty()
                    
                    if all_recommendations:
                        st.session_state['all_recommendations'] = all_recommendations
                        st.session_state['learner_results'] = learner_results
                        st.success(f"Generated recommendations for {len(learner_results)} users!")
                    else:
                        st.warning("No recommendations returned. Check your JWT token or try again.")
            
            # Display recommendations if we have them
            if 'all_recommendations' in st.session_state and st.session_state['all_recommendations']:
                all_recommendations = st.session_state['all_recommendations']
                learner_results = st.session_state['learner_results']
                
                st.markdown("---")
                
                # ─────────────────────────────────────────────────────
                # Table 1: Top Recommended Courses/Programs
                # ─────────────────────────────────────────────────────
                st.markdown("#### Top Recommended Courses/Programs")
                st.markdown("*Most frequently matched content across all learners' skill gaps.*")
                
                recs_df = pd.DataFrame(all_recommendations)
                
                # Group by parent_key and parent_title
                course_counts = recs_df.groupby(['parent_key', 'parent_title']).agg(
                    total_matches=('skill', 'count'),
                    unique_users=('user', 'nunique'),
                    skills_list=('skill', lambda x: ', '.join(sorted(set(x))[:5]))  # First 5 unique skills
                ).reset_index()
                course_counts = course_counts.sort_values('total_matches', ascending=False)
                course_counts['% of Cohort'] = (course_counts['unique_users'] / len(attempts_df) * 100).round(1)
                
                # Rename for display
                course_display = course_counts[['parent_title', 'parent_key', 'total_matches', 'unique_users', '% of Cohort', 'skills_list']].head(15)
                course_display.columns = ['Course/Program', 'Key', 'Total Matches', 'Users', '% of Cohort', 'Skills Addressed']
                
                st.dataframe(course_display, hide_index=True, use_container_width=True)
                
                st.markdown("---")
                
                # ─────────────────────────────────────────────────────
                # Table 2: Recommended Lessons by Course
                # ─────────────────────────────────────────────────────
                st.markdown("#### Recommended Lessons")
                st.markdown("*Specific lessons that address skill gaps.*")
                
                lesson_counts = recs_df.groupby(['parent_title', 'lesson_title']).agg(
                    matches=('skill', 'count'),
                    skills=('skill', lambda x: ', '.join(sorted(set(x))[:3]))  # First 3 skills
                ).reset_index()
                lesson_counts = lesson_counts.sort_values('matches', ascending=False)
                lesson_counts.columns = ['Course/Program', 'Lesson', 'Matches', 'Skills Addressed']
                
                st.dataframe(lesson_counts.head(25), hide_index=True, use_container_width=True)
                
                st.markdown("---")
                
                # ─────────────────────────────────────────────────────
                # Table 3: Per-Learner Recommendations
                # ─────────────────────────────────────────────────────
                st.markdown("#### Per-Learner Top Recommendation")
                st.markdown("*Each learner's most-matched course based on their specific skill gaps.*")
                
                learner_df = pd.DataFrame(learner_results)
                learner_df = learner_df.sort_values('Match Count', ascending=False)
                
                st.dataframe(learner_df, hide_index=True, use_container_width=True)
                
                st.markdown("---")
                
                # ─────────────────────────────────────────────────────
                # Table 4: Average Score by Recommended Course
                # ─────────────────────────────────────────────────────
                st.markdown("#### Average Score by Recommended Course")
                st.markdown("*Which courses are recommended for lower vs. higher scorers?*")
                
                score_by_course = learner_df.groupby(['Top Recommendation', 'Recommendation Key']).agg(
                    avg_score=('Score', 'mean'),
                    num_users=('User', 'count')
                ).reset_index()
                score_by_course['avg_score'] = score_by_course['avg_score'].round(0)
                score_by_course = score_by_course.sort_values('avg_score')
                score_by_course.columns = ['Recommended Course', 'Key', 'Avg Score', 'Users']
                
                st.dataframe(score_by_course, hide_index=True, use_container_width=True)
                
                st.markdown("---")
                
                # Export recommendations
                st.markdown("#### Export Recommendations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_courses = course_counts.to_csv(index=False)
                    st.download_button(
                        label="Course Recommendations CSV",
                        data=csv_courses,
                        file_name=f"{selected_assessment.replace(' ', '_')}_course_recommendations.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    csv_learners = learner_df.to_csv(index=False)
                    st.download_button(
                        label="Learner Recommendations CSV",
                        data=csv_learners,
                        file_name=f"{selected_assessment.replace(' ', '_')}_learner_recommendations.csv",
                        mime="text/csv"
                    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # AI Assistant Chat Interface (using Streamlit chat components)
    # ─────────────────────────────────────────────────────────────────────────────
    with st.expander("AI Assistant", expanded=False):
        if not OPENAI_API_KEY:
            st.warning("Add your OpenAI API key to settings.py to enable the AI assistant.")
        else:
            # Initialize chat history with welcome message
            if 'chat_history' not in st.session_state:
                st.session_state['chat_history'] = [
                    {
                        'role': 'assistant',
                        'content': "Hi! I can help you analyze this skills assessment data. Ask me things like: *How would you summarize insights for the client?*, *What should this organization learn next?*, or *Create a narrative of strengths and gaps*."
                    }
                ]
            
            # Clear chat button
            if st.button("Clear Chat", key="clear_chat"):
                st.session_state['chat_history'] = [
                    {
                        'role': 'assistant',
                        'content': "Hi! I can help you analyze this skills assessment data. Ask me things like: *How would you summarize insights for the client?*, *What should this organization learn next?*, or *Create a narrative of strengths and gaps*."
                    }
                ]
                st.rerun()
            
            # Display chat messages
            chat_container = st.container(height=450)
            with chat_container:
                for msg in st.session_state['chat_history']:
                    with st.chat_message(msg['role']):
                        st.markdown(msg['content'])
            
            # Chat input
            if prompt := st.chat_input("Ask about your data...", key="chat_input"):
                # Add user message to history
                st.session_state['chat_history'].append({
                    'role': 'user',
                    'content': prompt
                })
                
                # Display user message immediately
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                
                # Prepare session data for agent
                session_data = {
                    'attempts_df': st.session_state.get('current_attempts_df'),
                    'weak_df': st.session_state.get('current_weak_df'),
                    'strong_df': st.session_state.get('current_strong_df'),
                    'taxonomy_skills': st.session_state.get('current_taxonomy'),
                    'selected_assessment': st.session_state.get('current_assessment'),
                    'all_recommendations': st.session_state.get('all_recommendations', [])
                }
                
                # Get AI response with streaming display
                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = run_conversation(
                                prompt,
                                session_data,
                                st.session_state['chat_history'][:-1]
                            )
                        st.markdown(response)
                
                # Add assistant response to history
                st.session_state['chat_history'].append({
                    'role': 'assistant',
                    'content': response
                })
                
                st.rerun()


if __name__ == "__main__":
    main()
