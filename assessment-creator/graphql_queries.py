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


def query_nd_full(nd_key, locale="en-us"):
    """Fetch an ND's full part data (metadata + content tree) in a single round-trip.

    Replaces the previous ND crosswalk (which only returned `parts[].key` and
    then required an N+1 fan-out of `query_component` + `query_node` calls per
    part) with a single GraphQL request that traverses
    `parts -> branch -> component -> metadata` and `parts -> modules -> lessons
    -> concepts -> atoms` in one shot.

    Why `branch.component.metadata` and not the Part node directly: in the
    classroom-content schema, `Part.metadata` is `[MetadataTag]` (software /
    hardware / third_party_tool tags). The skills-and-difficulty metadata that
    assessment generation needs (`difficulty_level`, `teaches_skills`,
    `prerequisite_skills`) lives on `Component.metadata: ComponentMetadata`,
    which is reachable from a Part node via `branch.component.metadata`.

    Locale: the underlying `nanodegree(key:)` resolver defaults to `'en-us'`
    when no locale is provided and does NOT fall back across locales. Pass an
    explicit `locale` for non-en-us NDs (e.g. `'de-de'` for vfgermany variants).

    Returns the Nanodegree dict (with `title`, `parts[]` containing inline
    `metadata`, `id`, `modules`, etc.) or None on failure. The caller is
    responsible for asserting semantic_type == 'Nanodegree' and iterating
    parts[].
    """
    payload = {
        "query": """
        query AssessmentsAPI_NDFullQuery($key: String!, $locale: String) {
          nanodegree(key: $key, locale: $locale) {
            id
            key
            locale
            version
            title
            semantic_type
            parts {
              id
              key
              locale
              version
              title
              semantic_type
              branch {
                component {
                  metadata {
                    difficulty_level { name uri }
                    teaches_skills { name uri }
                    prerequisite_skills { name uri }
                  }
                }
              }
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
        }
        """,
        "variables": {"key": nd_key, "locale": locale}
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
        print(f"\n\nERROR querying nanodegree full content for key {nd_key} (locale={locale}): {e}")
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
