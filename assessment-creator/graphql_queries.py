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
                  prerequisite_skills {
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
            settings.CLASSROOM_CONTENT_API_URL,
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


def query_nd_parts_by_key(nd_key):
    """Crosswalk an `nd*` key to its part `cd*` keys in a single round-trip.

    Uses the dedicated `nanodegree(key:)` root resolver rather than going
    through component(key, locale:). The component-based path requires an
    exact locale match and silently returns null for NDs whose only release
    is in a non-en-us locale (e.g. enterprise variants like
    `nd029-ent-vfgermany`). `nanodegree(key:)` resolves by key alone and
    picks the latest available version regardless of locale.

    Returns the Nanodegree dict (with title + parts[]) or None on failure.
    The caller is responsible for asserting semantic_type == 'Nanodegree' and
    extracting parts[].key.
    """
    payload = {
        "query": """
        query AssessmentsAPI_NDPartsByKeyQuery($key: String!) {
          nanodegree(key: $key) {
            key
            title
            semantic_type
            parts {
              key
              title
              semantic_type
            }
          }
        }
        """,
        "variables": {"key": nd_key}
    }

    try:
        response = requests.post(
            settings.CLASSROOM_CONTENT_API_URL,
            headers=settings.production_headers(),
            json=payload
        )
        response.raise_for_status()
        data = response.json().get('data') or {}
        return data.get('nanodegree')

    except Exception as e:
        print(f"\n\nERROR querying nanodegree parts for key {nd_key}: {e}")
        return None


def query_node(node_id):
    payload = {
        "query": """
        query AssessmentsAPINotebooks_NodeQuery($id: Int!) {
  node(id: $id) {
    ... on Nanodegree {
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
                ... on TextAtom {
                  key
                  locale
                  version
                  semantic_type
                  title
                  text
                }
                ... on VideoAtom {
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
                    complex_prompt {
                      text
                    }
                    answers {
                      text
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    ... on Part {
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
              ... on TextAtom {
                key
                locale
                version
                semantic_type
                title
                text
              }
              ... on VideoAtom {
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
                  complex_prompt {
                    text
                  }
                  answers {
                    text
                  }
                }
              }
            }
          }
        }
      }
    }
    ... on Lesson {
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
          ... on TextAtom {
            key
            locale
            version
            semantic_type
            title
            text
          }
          ... on VideoAtom {
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
              complex_prompt {
                text
              }
              answers {
                text
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
            settings.CLASSROOM_CONTENT_API_URL,
            headers=settings.production_headers(),
            json=payload
        )
        response.raise_for_status()
        return response.json()['data']['node']

    except Exception as e:
        print(f"\n\nERROR querying node {node_id}: {e}")
        return None
