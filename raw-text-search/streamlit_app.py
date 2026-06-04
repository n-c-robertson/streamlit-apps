import streamlit as st
import requests
import pandas as pd
import re
import dataclasses
from collections.abc import Mapping, Iterable
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configuration
st.set_page_config(
    page_title="Udacity Catalog Text Search",
    page_icon="🔍",
    layout="wide"
)

# App secrets.
JWT = st.secrets['JWT']
CLASSROOM_CONTENT_API_URL = st.secrets['CLASSROOM_CONTENT_API_URL']
CATALOG_API_URL = st.secrets['CATALOG_API_URL']


headers = {
    'content-type': 'application/json',
    'Authorization': f'Bearer {JWT}',
    'Accept': 'application/json'
}

# Helper functions from the original script
SENTINEL = object()

def _is_primitive(x):
    return isinstance(x, (str, int, float, bool, type(None)))

def _to_iterable(obj):
    """Turn dataclasses and objects with __dict__ into mappable/iterable forms."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return obj.__dict__
    return SENTINEL

def _should_exclude(raw_key_path, key, exclude_keys, exclude_pred):
    """
    raw_key_path: tuple[str] of ancestor raw keys/indices (e.g., ('data','node','lessons','[0]','concepts'))
    key: str for the *next* child key we're considering
    """
    if exclude_pred and exclude_pred(raw_key_path, key):
        return True
    if not exclude_keys:
        return False

    # Build dotted raw path like "data.node.lessons.[0].concepts.key"
    dotted = ".".join([*raw_key_path, key]).lower()
    lowered = {ek.lower() for ek in exclude_keys if isinstance(ek, str)}

    return key.lower() in lowered or dotted in lowered

def _copy_mapping_like(obj):
    """Best-effort shallow copy of a mapping-like object for context payloads."""
    if isinstance(obj, Mapping):
        try:
            return {k: obj[k] for k in obj.keys()}
        except Exception:
            pass
        try:
            return dict(obj)
        except Exception:
            return str(obj)
    return obj

def _select_label(value, preferred_fields):
    """Pick a human-friendly label from a mapping 'value' using the given preferred_fields."""
    if not isinstance(value, Mapping):
        return None
    for f in preferred_fields:
        if f in value and value[f] not in (None, "", []):
            return str(value[f])
    return None

def _singularize(word: str) -> str:
    # Simple heuristic: "lessons" -> "lesson", "concepts" -> "concept", "atoms" -> "atom"
    if not word:
        return word
    if word.endswith('s') and len(word) > 1:
        return word[:-1]
    return word

def _make_path_entry(role_key, value, *, label_fields_per_key, default_label_fields):
    """
    Return a {role: label} style entry.
    - role_key: parent container key (e.g., 'node', 'lessons', 'concepts', 'atoms')
    - value: the child value we are descending into
    """
    role = _singularize((role_key or "item").strip())
    preferred = label_fields_per_key.get(role.lower(), default_label_fields)
    label = _select_label(value, preferred) if isinstance(value, Mapping) else None
    return {role: (label if label else role_key)}

def _iter_nodes(
    root,
    *,
    exclude_keys,
    exclude_pred=None,
    label_fields_per_key=None,
    default_label_fields=None,
):
    """
    Depth-first walk that yields:
    - disp_path: tuple[dict] -> structured display path [{role: label}, ...]
    - raw_path:  tuple[str]  -> raw keys/indices for internal logic
    - key:       current key (str) if inside a mapping, else None
    - value:     the current value
    - parents:   list of (container, raw_path_to_container, disp_path_to_container)
    """
    if label_fields_per_key is None:
        label_fields_per_key = {}
    if default_label_fields is None:
        default_label_fields = ["title", "name", "semantic_type", "id", "key", "slug"]

    # stack holds (value, raw_path, disp_path, key, parents)
    stack = [(root, (), (), None, [])]
    seen = set()

    while stack:
        value, raw_path, disp_path, key, parents = stack.pop()

        try:
            obj_id = id(value)
            if obj_id in seen:
                continue
            seen.add(obj_id)
        except Exception:
            pass

        yield disp_path, raw_path, key, value, parents

        # normalize dataclasses / plain objects
        as_iterable = _to_iterable(value)
        if as_iterable is not SENTINEL:
            value = as_iterable

        if isinstance(value, Mapping):
            for k, v in value.items():
                k_str = str(k)
                if _should_exclude(raw_path, k_str, exclude_keys, exclude_pred):
                    continue
                entry = _make_path_entry(
                    k_str, v,
                    label_fields_per_key=label_fields_per_key,
                    default_label_fields=default_label_fields,
                )
                stack.append(
                    (
                        v,
                        (*raw_path, k_str),
                        (*disp_path, entry),
                        k_str,
                        [*parents, (value, raw_path, disp_path)],
                    )
                )

        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            # For list elements, reuse the parent role (the last raw key)
            parent_role = raw_path[-1] if raw_path else "item"
            for idx, v in enumerate(value):
                idx_token = f"[{idx}]"
                # Exclusion works on raw keys; list elements don't have their own key, so we skip that check here
                entry = _make_path_entry(
                    parent_role, v,
                    label_fields_per_key=label_fields_per_key,
                    default_label_fields=default_label_fields,
                )
                stack.append(
                    (
                        v,
                        (*raw_path, idx_token),
                        (*disp_path, entry),
                        None,
                        [*parents, (value, raw_path, disp_path)],
                    )
                )

def _match_value(val_str, keyword, *, use_regex, case_sensitive):
    if use_regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        return re.search(keyword, val_str, flags) is not None
    else:
        return (keyword in val_str) if case_sensitive else (keyword.casefold() in val_str.casefold())

def search_dict_traces(
    data,
    keyword,
    *,
    exclude_keys=None,
    exclude_pred=None,
    case_sensitive=False,
    use_regex=False,
    context="nearest_mapping",  # or "self"
    # label behavior
    label_fields_per_key=None,
    default_label_fields=None,
    summarize=False,  # if True, return {"found", "count", "traces"}
):
    """
    Recursively search any nested dict/list/etc. for keyword matches.
    Returns detailed traces with structured semantic paths: [{'node': '...'}, {'lesson':'...'}, {'concept':'...'}, {'atom':'...'}]
    """
    exclude_keys = set(exclude_keys or [])

    # Sensible defaults for semantic labeling, including 'node'
    if label_fields_per_key is None:
        label_fields_per_key = {
            "node": ["title", "name", "semantic_type", "id", "key", "slug"],
            "lesson": ["title", "name"],
            "concept": ["title", "name"],
            "atom": ["semantic_type", "title", "name"],
        }
    if default_label_fields is None:
        default_label_fields = ["title", "name", "semantic_type", "id", "key", "slug"]

    traces = []

    for disp_path, raw_path, key, value, parents in _iter_nodes(
        data,
        exclude_keys=exclude_keys,
        exclude_pred=exclude_pred,
        label_fields_per_key=label_fields_per_key,
        default_label_fields=default_label_fields,
    ):
        # Only match on primitive values (stringify non-strings)
        if _is_primitive(value):
            val_str = str(value)
            if not _match_value(val_str, keyword, use_regex=use_regex, case_sensitive=case_sensitive):
                continue

            # choose context node and its display path
            if context == "self":
                ctx_node, ctx_disp_path = value, disp_path
            else:
                ctx_node, ctx_disp_path = None, None
                for container, parent_raw, parent_disp in reversed(parents):
                    if isinstance(container, Mapping):
                        ctx_node, ctx_disp_path = container, parent_disp
                        break
                if ctx_node is None:
                    if parents:
                        ctx_node, _, ctx_disp_path = parents[-1]
                    else:
                        ctx_node, ctx_disp_path = value, disp_path

            # matched key/value info
            if key is not None:
                matched_key = key
            else:
                # fall back to the role name of the last display entry
                if disp_path:
                    last = disp_path[-1]
                    matched_key = next(iter(last.keys()))
                else:
                    matched_key = "value"

            matched = {"key": matched_key, "value": val_str}
            ctx_payload = _copy_mapping_like(ctx_node)

            traces.append(
                {
                    "path": list(ctx_disp_path),
                    "matched": matched,
                }
            )

    if summarize:
        return {"found": bool(traces), "count": len(traces), "traces": traces}
    return traces

# API functions
class ClassroomContentError(Exception):
    """Structured error from a classroom-content GraphQL call.

    Carries the HTTP status, GraphQL errors[] (if any), and a body preview
    so the caller can render an actually-useful diagnostic. Replaces the
    old pattern of `response.json()['data']['node']['id']` blowing up with
    `KeyError: 'data'` whenever the API returned an auth error / gateway
    error / non-GraphQL payload.
    """

    def __init__(self, message, *, status=None, gql_errors=None, body_preview=None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.gql_errors = gql_errors
        self.body_preview = body_preview

    def __str__(self):
        bits = [self.message]
        if self.status is not None:
            bits.append(f"http={self.status}")
        if self.gql_errors:
            # Compress GraphQL errors to "first error message" + count.
            first = (self.gql_errors[0] or {}).get('message') if self.gql_errors else None
            bits.append(f"gql_errors={len(self.gql_errors)}({first!r})")
        if self.body_preview:
            bits.append(f"body={self.body_preview!r}")
        return " | ".join(bits)


def _post_gql(query, variables, *, op_label):
    """Run a GraphQL POST and return the parsed `data` dict.

    Raises ClassroomContentError with full HTTP/GraphQL context on any
    failure mode (network exception, non-200, non-JSON body,
    GraphQL errors[], or data=null).
    """
    try:
        resp = requests.post(
            CLASSROOM_CONTENT_API_URL,
            headers=headers,
            json={"query": query, "variables": variables},
            timeout=60,
        )
    except Exception as e:
        raise ClassroomContentError(
            f"{op_label}: network/request failure: {type(e).__name__}: {e}"
        ) from e

    body_preview = (resp.text or "")[:500]

    if resp.status_code != 200:
        raise ClassroomContentError(
            f"{op_label}: non-200 from classroom-content",
            status=resp.status_code,
            body_preview=body_preview,
        )

    try:
        body = resp.json()
    except Exception as e:
        raise ClassroomContentError(
            f"{op_label}: non-JSON response ({type(e).__name__}: {e})",
            status=resp.status_code,
            body_preview=body_preview,
        ) from e

    gql_errors = body.get("errors") if isinstance(body, dict) else None
    data = body.get("data") if isinstance(body, dict) else None

    if data is None:
        raise ClassroomContentError(
            f"{op_label}: response has no 'data' field",
            status=resp.status_code,
            gql_errors=gql_errors,
            body_preview=body_preview,
        )

    return data, gql_errors


def fetchNodeId(key):
    """Resolve a Component/Node key to a node id.

    Tries `Query.node(key:)` first (works for legacy/root-aligned keys).
    If that returns null, falls back to `Query.component(key:, locale:)`
    and uses `latest_release.root_node_id` - the same path the
    assessment-creator uses, which works for modern Component keys whose
    node-level key is a UUID rather than the cd*/ud* string.
    """
    data, _ = _post_gql(
        query="""
        query keyToNode($key: String!) {
          node(key: $key) {
            id
          }
        }
        """,
        variables={"key": key},
        op_label=f"node(key:{key!r})",
    )

    node = data.get("node")
    if node and node.get("id") is not None:
        return node["id"]

    # Fallback: resolve via Component -> latest_release.root_node_id.
    data2, gql_errors2 = _post_gql(
        query="""
        query keyToComponentRootNode($key: String!, $locale: String!) {
          component(key: $key, locale: $locale) {
            latest_release {
              root_node_id
            }
          }
        }
        """,
        variables={"key": key, "locale": "en-us"},
        op_label=f"component(key:{key!r}, locale:'en-us')",
    )

    component = data2.get("component")
    if not component:
        raise ClassroomContentError(
            f"key={key!r}: node(key:) returned null AND "
            f"component(key:, locale:'en-us') returned null. "
            "The key is unknown to classroom-content in en-us, or the "
            "JWT can't see it.",
            gql_errors=gql_errors2,
        )

    release = component.get("latest_release") or {}
    root_node_id = release.get("root_node_id")
    if root_node_id is None:
        raise ClassroomContentError(
            f"key={key!r}: component resolved but latest_release."
            "root_node_id is null. The Component exists but has no "
            "published release in en-us.",
            gql_errors=gql_errors2,
        )
    return root_node_id

def fetchClassroomContent(node_id):
    payload = {
        "query": """
        query classroomContent($id: Int!) {
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
            ... on Course {
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
          }
        }
                """,
        "variables": {
            "id": node_id
        }
    }
    data, gql_errors = _post_gql(
        query=payload["query"],
        variables=payload["variables"],
        op_label=f"node(id:{node_id})",
    )
    if (data.get("node") is None) and gql_errors:
        # Surface GraphQL errors as a structured exception so the caller
        # logs the actual reason instead of silently rendering an empty
        # search result.
        raise ClassroomContentError(
            f"node(id:{node_id}): data.node is null and GraphQL returned errors",
            gql_errors=gql_errors,
        )
    # Preserve the original `{"data": {...}}` envelope the downstream
    # search_dict_traces walker expects (format_path explicitly skips the
    # "data" key when rendering result paths).
    return {"data": data}

@st.cache_data
def load_catalog():
    """Load the Udacity catalog data"""
    catalog_url = CATALOG_API_URL
    catalog_data = requests.post(catalog_url, json={'pageSize': 1000}, headers=headers).json()
    catalog_results = [r for r in catalog_data['searchResult']['hits'] if r['is_offered_to_public']]
    return catalog_results

@st.cache_data
def load_content(selected_keys):
    """Load content for selected catalog items using parallel processing"""
    content_data = []
    
    # Thread-safe progress tracking
    progress_lock = threading.Lock()
    completed_count = [0]  # Use list for mutable reference
    
    def fetch_single_content(key):
        """Fetch content for a single key"""
        try:
            node_id = fetchNodeId(key)
            content = fetchClassroomContent(node_id)
            with progress_lock:
                completed_count[0] += 1
            return {
                'key': key,
                'node_id': node_id,
                'content': content,
                'success': True,
                'error': None,
                'error_detail': None,
            }
        except ClassroomContentError as e:
            with progress_lock:
                completed_count[0] += 1
            return {
                'key': key,
                'node_id': None,
                'content': None,
                'success': False,
                'error': str(e),
                'error_detail': {
                    'type': 'ClassroomContentError',
                    'message': e.message,
                    'http_status': e.status,
                    'gql_errors': e.gql_errors,
                    'body_preview': e.body_preview,
                },
            }
        except Exception as e:
            with progress_lock:
                completed_count[0] += 1
            return {
                'key': key,
                'node_id': None,
                'content': None,
                'success': False,
                'error': f"{type(e).__name__}: {e}",
                'error_detail': {
                    'type': type(e).__name__,
                    'message': str(e),
                },
            }
    
    # Create progress bar and status
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Use ThreadPoolExecutor for parallel processing
    max_workers = min(10, len(selected_keys))  # Limit concurrent requests
    
    failures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_key = {executor.submit(fetch_single_content, key): key for key in selected_keys}

        # Process completed tasks
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            result = future.result()

            if result['success']:
                content_data.append({
                    'key': result['key'],
                    'node_id': result['node_id'],
                    'content': result['content']
                })
            else:
                failures.append(result)

            # Update progress
            progress = len([f for f in future_to_key if f.done()]) / len(selected_keys)
            progress_bar.progress(progress)
            status_text.text(f'Loading content... ({len([f for f in future_to_key if f.done()])}/{len(selected_keys)} completed)')

    status_text.empty()
    progress_bar.empty()

    if failures:
        # Bucket failures by the leading part of the error so 700 catalog
        # entries failing for the same reason collapse to one row instead
        # of 700 red banners.
        from collections import Counter as _Counter
        buckets = _Counter()
        for f in failures:
            detail = f.get('error_detail') or {}
            http = detail.get('http_status')
            gql = detail.get('gql_errors') or []
            first_gql_msg = (gql[0] or {}).get('message') if gql else None
            bucket_key = (
                detail.get('type') or 'Unknown',
                http,
                first_gql_msg,
            )
            buckets[bucket_key] += 1

        st.warning(
            f"{len(failures)} of {len(selected_keys)} key(s) failed to load - "
            "expand below for details. The downstream search will only run "
            "against the successful ones."
        )
        with st.expander("Per-key failure diagnostics", expanded=True):
            st.markdown("**Failure buckets:**")
            for (etype, http, first_gql), count in buckets.most_common():
                st.write(
                    f"- `{count}x` {etype} | http={http} | "
                    f"first gql error: {first_gql!r}"
                )

            st.markdown("**First 5 failed keys (full detail):**")
            for f in failures[:5]:
                st.code(
                    f"key={f['key']!r}\n"
                    f"error={f['error']}\n"
                    f"detail={f.get('error_detail')!r}",
                    language="text",
                )

    return content_data

def format_path(path):
    """Format the path for display"""
    path_parts = []
    for item in path:
        for role, label in item.items():
            # Skip if role is "data"
            if role.lower() == "data":
                continue
            # Skip if role and label are the same (e.g., "lesson:lesson", "concept:concept")
            if role.lower() == label.lower():
                continue
            # Skip if role and label are singular/plural of each other
            role_lower = role.lower()
            label_lower = label.lower()
            if (role_lower + 's' == label_lower) or (label_lower + 's' == role_lower):
                continue
            path_parts.append(f"{role}: {label}")
    return " → ".join(path_parts)

def main():
    st.title("🔍 Udacity Catalog Text Search")
    st.markdown("Search for text strings across the Udacity catalog content")
    
    # Sidebar for configuration
    st.sidebar.header("Search Configuration")
    
    # Load catalog
    if 'catalog' not in st.session_state:
        with st.spinner('Loading Udacity catalog...'):
            st.session_state.catalog = load_catalog()
    
    catalog = st.session_state.catalog
    
    # Catalog selection
    st.sidebar.subheader("Select Courses/Nanodegrees")
    catalog_options = {item['title']: item['key'] for item in catalog}
    
    selected_titles = st.sidebar.multiselect(
        "Choose items to search (leave empty to search all):",
        options=list(catalog_options.keys()),
        default=[]
    )
    
    # If nothing selected, search all
    if not selected_titles:
        selected_keys = list(catalog_options.values())
        st.sidebar.info(f"Will search all {len(selected_keys)} courses/nanodegrees")
    else:
        selected_keys = [catalog_options[title] for title in selected_titles]
    
    # Search options
    st.sidebar.subheader("Search Options")
    case_sensitive = st.sidebar.checkbox("Case sensitive search")
    use_regex = st.sidebar.checkbox("Use regular expressions")
    
    # Main search interface
    st.header("Search")
    search_term = st.text_input("Enter text to search for:", placeholder="e.g., machine learning, python, neural network")
    
    if st.button("🔍 Search", type="primary", key="search_button") and search_term:
        # Load content
        with st.spinner('Loading content...'):
            content_data = load_content(selected_keys)
        
        # Perform search
        all_results = []
        # Always exclude metadata and titles
        exclude_keys = {"metadata", "title"}
        
        with st.spinner('Searching...'):
            for item in content_data:
                try:
                    results = search_dict_traces(
                        item['content'],
                        search_term,
                        exclude_keys=exclude_keys,
                        case_sensitive=case_sensitive,
                        use_regex=use_regex,
                        summarize=True
                    )
                    
                    if results['found']:
                        for trace in results['traces']:
                            trace['source_key'] = item['key']
                            trace['source_title'] = next(
                                (cat['title'] for cat in catalog if cat['key'] == item['key']), 
                                item['key']
                            )
                        all_results.extend(results['traces'])
                except Exception as e:
                    st.error(f"Error searching in {item['key']}: {str(e)}")
        
        # Display results
        if all_results:
            st.success(f"Found {len(all_results)} matches across {len(set(r['source_key'] for r in all_results))} courses/nanodegrees")
            
            # Group results by source
            results_by_source = {}
            for result in all_results:
                source = result['source_title']
                if source not in results_by_source:
                    results_by_source[source] = []
                results_by_source[source].append(result)
            
            # Display results grouped by source
            for source_title, source_results in results_by_source.items():
                with st.expander(f"📚 {source_title} ({len(source_results)} matches)", expanded=True):
                    for i, result in enumerate(source_results):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**Path:** {format_path(result['path'])}")
                            
                            # Show matched value with highlighting using st.code for better display
                            matched_value = result['matched']['value']
                            st.markdown("**Content:**")
                            
                            # Create highlighted text
                            if not use_regex and matched_value:
                                # Split the text around the search term for highlighting
                                if case_sensitive:
                                    parts = matched_value.split(search_term)
                                    if len(parts) > 1:
                                        # Use colored text with markdown
                                        highlighted_parts = []
                                        for j, part in enumerate(parts):
                                            if j < len(parts) - 1:  # Not the last part
                                                highlighted_parts.append(part)
                                                highlighted_parts.append(f":red[**{search_term}**]")
                                            else:
                                                highlighted_parts.append(part)
                                        st.markdown("".join(highlighted_parts))
                                    else:
                                        st.markdown(matched_value)
                                else:
                                    # Case insensitive highlighting
                                    import re as regex_module
                                    pattern = regex_module.compile(regex_module.escape(search_term), regex_module.IGNORECASE)
                                    matches = list(pattern.finditer(matched_value))
                                    if matches:
                                        result_parts = []
                                        last_end = 0
                                        for match in matches:
                                            # Add text before match
                                            result_parts.append(matched_value[last_end:match.start()])
                                            # Add highlighted match
                                            result_parts.append(f":red[**{matched_value[match.start():match.end()]}**]")
                                            last_end = match.end()
                                        # Add remaining text
                                        result_parts.append(matched_value[last_end:])
                                        st.markdown("".join(result_parts))
                                    else:
                                        st.markdown(matched_value)
                            else:
                                # For regex or if no highlighting needed, render as markdown
                                st.markdown(matched_value)
                        
                        with col2:
                            st.markdown(f"*Match {i+1}*")
                        
                        if i < len(source_results) - 1:
                            st.divider()
            
            # Export results
            if st.button("📥 Export Results as JSON", key="export_button"):
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(all_results, indent=2),
                    file_name=f"udacity_search_results_{search_term.replace(' ', '_')}.json",
                    mime="application/json"
                )
        
        else:
            st.warning("No matches found. Try adjusting your search term or selecting different courses.")
    
    # Display catalog info
    if catalog:
        st.sidebar.markdown(f"**Total available:** {len(catalog)} courses/nanodegrees")
        st.sidebar.markdown(f"**Selected:** {len(selected_keys)} items")

if __name__ == "__main__":
    main() 