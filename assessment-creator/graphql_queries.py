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

def query_components_by_key(key):
    """Cross-locale enumeration of Component rows for a given key.

    `components(key:)` is NOT locale-gated, so this returns every release
    of the key across every locale we can see with the current JWT. Useful
    as the first step in resolving an ND key whose locale we don't know
    ahead of time (e.g. `nd029-ent-vfgermany`).

    Returns a list of component dicts (id, key, locale, type, deprecated,
    latest_release.root_node_id, ...) or [] on failure. On failure, prints
    structured diagnostics.
    """
    payload = {
        "query": """
        query AssessmentsAPI_ComponentsByKey($key: String!) {
          components(key: $key, count: 50) {
            id
            key
            locale
            type
            deprecated
            latest_release {
              root_node_id
              major
              minor
              patch
            }
          }
        }
        """,
        "variables": {"key": key}
    }

    try:
        resp = requests.post(
            settings.CLASSROOM_CONTENT_API_URL,
            headers=settings.production_headers(),
            json=payload,
            timeout=60,
        )
    except Exception as e:
        print(
            f"\n\nERROR [query_components_by_key] {key}: network/request "
            f"failure: {type(e).__name__}: {e}"
        )
        return []

    if resp.status_code != 200:
        preview = (resp.text or "")[:500]
        print(
            f"\n\nERROR [query_components_by_key] {key}: HTTP "
            f"{resp.status_code} from classroom-content. Preview: {preview!r}"
        )
        return []

    try:
        body = resp.json()
    except Exception as e:
        preview = (resp.text or "")[:500]
        print(
            f"\n\nERROR [query_components_by_key] {key}: non-JSON response "
            f"({type(e).__name__}: {e}). Preview: {preview!r}"
        )
        return []

    errors = body.get("errors")
    if errors:
        try:
            errs_dump = json.dumps(errors, indent=2)[:2000]
        except Exception:
            errs_dump = str(errors)[:2000]
        print(
            f"\n\nERROR [query_components_by_key] {key}: GraphQL returned "
            f"{len(errors)} error(s):\n{errs_dump}"
        )

    return ((body.get("data") or {}).get("components")) or []


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
        return ((data.get("data") or {}).get("component") or {}).get("latest_release")
    except Exception as e:
        print(f"\n\nERROR querying node for key {key}: {e}")
        return None


def query_construction_release(key, locale="en-us"):
    """Fallback resolver for keys that have no published RELEASE branch.

    classroom-content is branch-based (RELEASE / BUGFIX / CONSTRUCTION).
    `component(key:).latest_release` only returns the latest RELEASE branch,
    so it is null for a program that has never been published. The unreleased
    work lives on the CONSTRUCTION branch.

    Two root fields in a single query:
      - `node(key:, locale:, version:"construction")` — resolveRef maps the
        magic version string "construction" to the latest CONSTRUCTION branch
        ref for that key, returning the root node (id + title). This is the
        proven path for getting the content tree pointer for draft programs.
      - `component(key:, locale:).metadata` — top-level Component metadata
        (difficulty_level / teaches_skills / prerequisite_skills) is populated
        regardless of release state, so this is the right place to read skills
        and difficulty even when latest_release is null.

    The node-level `branch` field has no resolver in classroom-content (same
    quirk documented for `Part.branch` in query_nd_full), so
    `node(...).branch.component.metadata` always yields null — we deliberately
    do NOT use that path.

    Returns a dict shaped exactly like `query_component`'s `latest_release`
    return (so callers can use it without branching), plus an `_unreleased`
    marker so the UI can flag draft content:

        {
          "root_node_id": <int>,
          "root_node": {"id": <int>, "title": <str>},
          "component": {"metadata": {...} | None},
          "_unreleased": True,
        }

    Returns None on failure (no construction branch, auth gate, or transport
    error), printing structured diagnostics in the same style as the other
    queries here.
    """
    payload = {
        "query": """
        query AssessmentsAPI_ConstructionQuery($key: String!, $locale: String!) {
          node(key: $key, locale: $locale, version: "construction") {
            ... on Nanodegree { id title }
            ... on Course { id title }
            ... on Part { id title }
          }
          component(key: $key, locale: $locale) {
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
        }
        """,
        "variables": {"key": key, "locale": locale},
    }

    try:
        resp = requests.post(
            settings.CLASSROOM_CONTENT_API_URL,
            headers=settings.production_headers(),
            json=payload,
            timeout=60,
        )
    except Exception as e:
        print(
            f"\n\nERROR [query_construction_release] {key} locale={locale!r}: "
            f"network/request failure: {type(e).__name__}: {e}"
        )
        return None

    if resp.status_code != 200:
        preview = (resp.text or "")[:500]
        print(
            f"\n\nERROR [query_construction_release] {key} locale={locale!r}: "
            f"HTTP {resp.status_code} from classroom-content. Preview: {preview!r}"
        )
        return None

    try:
        body = resp.json()
    except Exception as e:
        preview = (resp.text or "")[:500]
        print(
            f"\n\nERROR [query_construction_release] {key} locale={locale!r}: "
            f"non-JSON response ({type(e).__name__}: {e}). Preview: {preview!r}"
        )
        return None

    errors = body.get("errors")
    if errors:
        try:
            errs_dump = json.dumps(errors, indent=2)[:2000]
        except Exception:
            errs_dump = str(errors)[:2000]
        print(
            f"\n\nERROR [query_construction_release] {key} locale={locale!r}: "
            f"GraphQL returned {len(errors)} error(s):\n{errs_dump}"
        )

    data_field = body.get("data") or {}
    node = data_field.get("node")
    component = data_field.get("component")

    if not node or not node.get("id"):
        print(
            f"\n[query_construction_release] {key} locale={locale!r}: "
            "node(key:, version:'construction') returned null or has no id. "
            "Either there is no CONSTRUCTION branch in this locale, the JWT "
            "cannot read it, or the key does not exist."
        )
        return None

    return {
        "root_node_id": node.get("id"),
        "root_node": {"id": node.get("id"), "title": node.get("title")},
        "component": {"metadata": (component or {}).get("metadata")},
        "_unreleased": True,
    }


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
      id
      key
      locale
      version
      semantic_type
      title
      parts {
        id
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
      id
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
