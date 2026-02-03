import streamlit as st
# API Configuration Settings
# These URLs can be modified for different environments (dev/staging/prod)

# OpenAI API Key (for AI Assistant)
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Workera API
WORKERA_API_BASE = "https://skills.workera.ai/api/v1"
WORKERA_DOMAINS_URL = f"{WORKERA_API_BASE}/domains"
WORKERA_BENCHMARKS_URL = f"{WORKERA_API_BASE}/benchmarks"
WORKERA_SIGNALS_URL = f"{WORKERA_API_BASE}/signals/cat"
# Scores-based flow (list scores, then scores/{id} for skill details)
WORKERA_SCORES_URL = f"{WORKERA_API_BASE}/scores"
WORKERA_SCORES_DETAIL_URL = f"{WORKERA_API_BASE}/scores/{{id}}"
WORKERA_DOMAINS_SAGE_URL = f"{WORKERA_API_BASE}/domains/sage"

# Udacity API
UDACITY_API_BASE = "https://api.udacity.com/api"
UDACITY_WORKERA_API_KEYS_URL = f"{UDACITY_API_BASE}/workera/api_keys"

# Udacity Skills API
SKILLS_API_BASE = "https://skills.udacity.com/api"
SKILLS_SEARCH_URL = f"{SKILLS_API_BASE}/skills/search"

# Udacity Skills → Taxonomy mapping (Workera skill → Udacity skill name → subject/domain)
SKILLS_SEARCH_SCOPED_URL = "https://api.udacity.com/api/skills/search/scoped/bundles"
UTAXONOMY_GRAPHQL_URL = "https://api.udacity.com/api/taxonomy/v1/graphql"
SKILLS_NODES_RELATED_URL = "https://skills.udacity.com/api/skills/nodes/related"

