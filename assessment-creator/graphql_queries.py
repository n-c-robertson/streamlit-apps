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
    """Fetch an ND's part list + full content tree (modules > lessons > concepts
    > atoms) in a single round-trip.

    Does NOT fetch per-part Component metadata (difficulty_level /
    teaches_skills / prerequisite_skills) in the same query. The previous
    revision tried to traverse `Part.branch.component.metadata`, but
    classroom-content does not register a field resolver for `Part.branch`,
    so that path returns null in production (confirmed against the schema
    in udacity-codebase: only `Query.branch(id:)` and `Dependent.branch` have
    resolvers; `PartResolvers.Part` has no `branch` entry). Per-part metadata
    is fetched separately by add_program_data via `query_component(key,
    locale=part_locale)`, which IS proven to work in the cd-key path.

    Locale: the underlying `nanodegree(key:)` resolver defaults to `'en-us'`
    when no locale is provided and does NOT fall back across locales. Pass an
    explicit `locale` for non-en-us NDs.

    Returns the Nanodegree dict (with `title`, `parts[]` containing `id`,
    `key`, `locale`, `version`, and a full `modules` tree) or None on failure.
    On failure, prints a structured diagnostic (HTTP status, GraphQL errors,
    short response preview) so we can actually debug what went wrong instead
    of getting a silent None.
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
            json=payload,
            timeout=60,
        )
    except Exception as e:
        print(
            f"\n\nERROR [query_nd_full] {nd_key} locale={locale!r}: "
            f"network/request failure: {type(e).__name__}: {e}"
        )
        return None

    if response.status_code != 200:
        preview = (response.text or "")[:500]
        print(
            f"\n\nERROR [query_nd_full] {nd_key} locale={locale!r}: "
            f"HTTP {response.status_code} from classroom-content. "
            f"Response preview: {preview!r}"
        )
        return None

    try:
        body = response.json()
    except Exception as e:
        preview = (response.text or "")[:500]
        print(
            f"\n\nERROR [query_nd_full] {nd_key} locale={locale!r}: "
            f"non-JSON response ({type(e).__name__}: {e}). Preview: {preview!r}"
        )
        return None

    errors = body.get("errors")
    if errors:
        # GraphQL execution / validation errors. Print them in full because
        # this is the path that was silently swallowing "data: null, errors:
        # [...]" responses before.
        try:
            errs_dump = json.dumps(errors, indent=2)[:2000]
        except Exception:
            errs_dump = str(errors)[:2000]
        print(
            f"\n\nERROR [query_nd_full] {nd_key} locale={locale!r}: "
            f"GraphQL returned {len(errors)} error(s):\n{errs_dump}"
        )

    data = body.get("data") or {}
    nd = data.get("nanodegree")
    if nd is None and not errors:
        # No exception, no GraphQL errors, but still no nanodegree. Most
        # commonly this is the locale gate (`nanodegree(key:, locale:)` does
        # not fall back across locales) or a visibility check silently nulling
        # the result inside fetchNodeWithAuth.
        print(
            f"\n[query_nd_full] {nd_key} locale={locale!r}: "
            "data.nanodegree was null (no GraphQL errors). Most likely causes: "
            "(1) the ND has no release in this locale, "
            "(2) the JWT lacks visibility on this ND, or "
            "(3) the key is not a Nanodegree."
        )
    return nd


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
