# =============================================================================
# AUTHENTICATION
# =============================================================================

# Note: JWT and Workera API key are now fetched dynamically via the app's authentication flow
# The workera_api_key is obtained by calling the Udacity Workera API Keys endpoint
# with a valid JWT token and company ID

# Approved company IDs (only these are allowed)
approved_company_ids = [33, 343]

# Default company ID
default_company_id = 33

# Udacity API for fetching Workera API keys
udacity_workera_api_keys_url = 'https://api.udacity.com/api/workera/api_keys'

# OpenAI API Key (for chatbot) - loaded from Streamlit secrets
# To configure: Add OPENAI_API_KEY in Streamlit Cloud secrets or .streamlit/secrets.toml
import streamlit as st
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    openai_api_key = None  # Will be handled gracefully in chat_ui.py

# =============================================================================
# WORKERA API ENDPOINTS
# =============================================================================

workera_url_domains = 'https://skills.workera.ai/api/v1/domains'
workera_url_signals_cat = 'https://skills.workera.ai/api/v1/signals/cat'


# =============================================================================
# UDACITY API ENDPOINTS
# =============================================================================

# EMC Content API
emc_content_api_url = 'https://api.udacity.com/api/emc/gql/query/public'

# Assessments API
assessments_api_url = 'https://api.udacity.com/api/assessments/graphql'

# Classroom Content API
classroom_content_api_url = 'https://api.udacity.com/api/classroom-content/v1/graphql'

# Taxonomy API
utaxonomy_api_url = 'https://api.udacity.com/api/taxonomy/v1/graphql'

# Skills APIs
skills_api_url = 'https://skills.udacity.com/api/skills/nodes/related'
skills_search_api_url = 'https://api.udacity.com/api/skills/search/scoped/bundles'
udacity_skills_api_url = 'https://skills.udacity.com/api/skills/search'

# Progress API (used for learning activity/frequency data)
learning_activity_api_url = 'https://api.udacity.com/api/progress/graphql'
