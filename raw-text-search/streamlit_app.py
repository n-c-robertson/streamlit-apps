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
def fetchNodeId(key):
    payload = {
        "query": """
        query keyToNode($key: String!) {
          node(key: $key ) {
            id
          }
        }
      """,
        "variables": {
            "key": key
        }
    }

    response = requests.post(
        CLASSROOM_CONTENT_API_URL,
        headers=headers,
        json=payload
    )
    
    return response.json()['data']['node']['id']

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
    response = requests.post(
        CLASSROOM_CONTENT_API_URL,
        headers=headers,
        json=payload
    )
    
    return response.json()

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
            
            # Update progress thread-safely
            with progress_lock:
                completed_count[0] += 1
                progress = completed_count[0] / len(selected_keys)
                # Note: We can't update Streamlit components from threads, 
                # so we'll handle progress display differently
            
            return {
                'key': key,
                'node_id': node_id,
                'content': content,
                'success': True,
                'error': None
            }
        except Exception as e:
            with progress_lock:
                completed_count[0] += 1
            return {
                'key': key,
                'node_id': None,
                'content': None,
                'success': False,
                'error': str(e)
            }
    
    # Create progress bar and status
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Use ThreadPoolExecutor for parallel processing
    max_workers = min(10, len(selected_keys))  # Limit concurrent requests
    
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
                st.error(f"Error loading content for {result['key']}: {result['error']}")
            
            # Update progress
            progress = len([f for f in future_to_key if f.done()]) / len(selected_keys)
            progress_bar.progress(progress)
            status_text.text(f'Loading content... ({len([f for f in future_to_key if f.done()])}/{len(selected_keys)} completed)')
    
    status_text.empty()
    progress_bar.empty()
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