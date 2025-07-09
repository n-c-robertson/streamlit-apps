#========================================
# IMPORT PACKAGES
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
import settings


#========================================
# SETTINGS
#========================================

CLASSROOM_CONTENT_API_URL = st.secrets['classroom_content_api_url']

#========================================
# QUERIES
#========================================

def query_component(key, locale="en-us"):
    payload = {
        "query": """
        query AssessmentsAPINotebooks_ComponentQuery($key: String!, $locale: String!) {
          component(key: $key, locale: $locale) {
            latest_release {
              major
              minor
              patch
              root_node_id
              type
              component {
                metadata {
                  difficulty_level {
                    name
                    uri
                  }
                  teaches_skills {
                    name
                    uri
                  }
                }
              }
              root_node {
                id
                key
                locale
                version
                title
              }
            }
          }
        }
        """,
        "variables": {"key": key, "locale": locale}
    }

    try:
        resp = requests.post(
            CLASSROOM_CONTENT_API_URL,
            headers=settings.production_headers(),
            json=payload
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", {}) \
                   .get("component", {}) \
                   .get("latest_release")
    except Exception as e:
        print(f"\n\nERROR querying node for key {key}: {e}")
        return None


def query_node(node_id):
    payload = {
        "query": """
        query AssessmentsAPINotebooks_NodeQuery($id: Int!) {
          node(id: $id) {
            ...on Nanodegree {
              key
              locale
              version
              semantic_type
              title
              parts {
                key
                locale
                version
                semantic_type
                title
                modules {
                  key
                  locale
                  version
                  semantic_type
                  title
                  lessons {
                    key
                    locale
                    version
                    semantic_type
                    title
                    concepts {
                      key
                      locale
                      version
                      semantic_type
                      title
                      progress_key
                      atoms {
                        ...on TextAtom {
                          key
                          locale
                          version
                          semantic_type
                          title
                          text
                        }
                        ...on VideoAtom {
                          key
                          locale
                          version
                          semantic_type
                          title
                          video {
                            vtt_url
                          }
                        }
                        ... on RadioQuizAtom {
                          semantic_type
                          question {
                            prompt
                            answers {
                              is_correct
                              text
                            }
                          }
                        }
                        ... on CheckboxQuizAtom {
                          semantic_type
                          question {
                            prompt
                            correct_feedback
                            answers {
                              is_correct
                              text
                            }
                          }
                        }
                        ... on MatchingQuizAtom {
                          semantic_type
                          question {
                            answers_label
                            concepts_label
                            concepts {
                              text
                              correct_answer {
                                text
                              }
                            }
                            complex_prompt { text }
                            answers { text }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            ...on Part {
              key
              locale
              version
              semantic_type
              title
              modules {
                key
                locale
                version
                semantic_type
                title
                lessons {
                  key
                  locale
                  version
                  semantic_type
                  title
                  concepts {
                    key
                    locale
                    version
                    semantic_type
                    title
                    progress_key
                    atoms {
                      ...on TextAtom {
                        key
                        locale
                        version
                        semantic_type
                        title
                        text
                      }
                      ...on VideoAtom {
                        key
                        locale
                        version
                        semantic_type
                        title
                        video {
                          vtt_url
                        }
                      }
                      ... on RadioQuizAtom {
                        semantic_type
                        question {
                          prompt
                          answers {
                            is_correct
                            text
                          }
                        }
                      }
                      ... on CheckboxQuizAtom {
                        semantic_type
                        question {
                          prompt
                          correct_feedback
                          answers {
                            is_correct
                            text
                          }
                        }
                      }
                      ... on MatchingQuizAtom {
                        semantic_type
                        question {
                          answers_label
                          concepts_label
                          concepts {
                            text
                            correct_answer {
                              text
                            }
                          }
                          complex_prompt { text }
                          answers { text }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """,
        "variables": {
            "id": node_id
        }
    }

    try:
        response = requests.post(
            CLASSROOM_CONTENT_API_URL,
            headers=settings.production_headers(),
            json=payload
        )
        response.raise_for_status()
        return response.json()['data']['node']

    except Exception as e:
        print(f"\n\nERROR querying node {node_id}: {e}")
        return None