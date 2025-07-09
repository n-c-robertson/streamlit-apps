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

#========================================
#SETTINGS
#========================================

ENVIRONMENT = 'production'

# Nathan's JWT key. But it will expire every 3 weeks.
UDACITY_JWT = st.secrets['jwt_token']

def production_headers():
    STAFF_HEADERS = {
        'Authorization': f'Bearer {UDACITY_JWT}',
        'Content-Type': 'application/json'
    }
    return STAFF_HEADERS

ASSESSMENTS_API_URL = st.secrets['assessments_api_url']
CLASSROOM_CONTENT_API_URL = st.secrets['classroom_content_api_url']

openai_client = OpenAI(
    api_key = st.secrets['openai_api_key']
)

CHAT_COMPLETIONS_MODEL = 'gpt-4o'
CHAT_COMPLETIONS_TEMPERATURE = 0.2
CHAT_COMPLETIONS_RESPONSE_FORMAT = {
    'type': 'json_object'
}
