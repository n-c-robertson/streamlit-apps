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

# API key for service account.
# The JWT is no longer stored in Streamlit secrets (it expired every ~3 weeks
# and required redeploying the app). Instead, each staff user pastes their own
# Udacity staff JWT into the sidebar on first use; it is kept in
# ``st.session_state`` for the lifetime of the browser session. See
# ``render_jwt_sidebar`` / ``get_udacity_jwt`` below.
SESSION_STATE_JWT_KEY = 'udacity_staff_jwt'

# Module-level mirror of the session JWT. ``st.session_state`` is thread-local
# in Streamlit (bound to the script-run context), so worker threads spawned via
# ``concurrent.futures.ThreadPoolExecutor`` cannot read it — they get no
# context and ``st.session_state.get(...)`` raises. To keep every authenticated
# call site working regardless of which thread it runs on, the main thread
# writes the JWT here whenever it (re)reads session state, and worker threads
# fall back to this cache. A single Python string reference is read/written
# atomically under the GIL, so no lock is needed for this one-writer/many-reader
# pattern. Cleared in lock-step with the session value (see ``render_jwt_sidebar``
# and ``clear_udacity_jwt``).
_JWT_CACHE = None


def get_udacity_jwt():
    """Return the staff JWT for the current session, or ``None`` if unset.

    Prefers ``st.session_state`` (the authoritative store, only readable on the
    main Streamlit thread) and mirrors the value into ``_JWT_CACHE`` so worker
    threads — which have no script-run context and cannot access
    ``st.session_state`` — can still retrieve the token via the cache.
    """
    global _JWT_CACHE
    try:
        jwt = st.session_state.get(SESSION_STATE_JWT_KEY) or None
        if jwt:
            # Keep the thread-readable cache in sync for worker threads.
            _JWT_CACHE = jwt
        return jwt
    except Exception:
        # ``st.session_state`` is only available inside a Streamlit run; module
        # imports outside Streamlit (e.g. unit tests) should not crash here.
        # Also reached inside worker threads that have no script-run context —
        # fall back to the cache populated by the main thread.
        return _JWT_CACHE


def clear_udacity_jwt():
    """Forget the JWT from both session state and the thread-readable cache.

    Called by ``render_jwt_sidebar`` when the user clears the token, so a
    cleared token is immediately invisible to in-flight worker threads too.
    """
    global _JWT_CACHE
    _JWT_CACHE = None
    try:
        st.session_state.pop(SESSION_STATE_JWT_KEY, None)
    except Exception:
        pass


def is_jwt_set():
    """True if a staff JWT has been entered for this session."""
    return bool(get_udacity_jwt())


def production_headers():
    """Build authenticated request headers using the session's staff JWT.

    Raises a ``RuntimeError`` with a helpful message if no JWT has been
    entered yet, so callers fail loudly instead of sending unauthenticated
    requests that silently 401 against Udacity's GraphQL APIs.
    """
    jwt = get_udacity_jwt()
    if not jwt:
        raise RuntimeError(
            "No Udacity staff JWT found for this session. Enter your JWT in "
            "the sidebar (\"Udacity staff JWT\") before running this action."
        )
    return {
        'Authorization': f'Bearer {jwt}',
        'Content-Type': 'application/json',
    }


def _jwt_fingerprint(jwt_value):
    """Short non-reversible hash of a JWT so we can show which token is in
    use without ever displaying the token itself."""
    if not jwt_value:
        return 'empty'
    return hashlib.sha256(jwt_value.encode('utf-8')).hexdigest()[:10]


# Minimal read-only GraphQL query used purely to validate that a candidate
# JWT is accepted by Udacity's assessments API. `difficultyLevels` is
# argument-less and cheap, so it's a good liveness/auth probe. Mirrors the
# query already used by `fetch_difficulty_levels` in the upload page.
_JWT_PROBE_QUERY = """
query {
  difficultyLevels {
    id
  }
}
"""


def verify_jwt(token, timeout=15):
    """Validate a candidate staff JWT by making one authenticated read.

    Sends the tiny ``difficultyLevels`` query to the assessments API with the
    candidate token as the Bearer header (NOT the session token, so we can
    validate before persisting). Returns ``(ok, message)`` where ``ok`` is
    True only when the API accepts the token and returns data.

    Used by ``render_jwt_sidebar`` to reject expired/invalid tokens at save
    time instead of letting the user discover the failure on the next action.
    """
    token = (token or '').strip()
    if not token:
        return False, "JWT is empty."
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
    }
    try:
        resp = requests.post(
            ASSESSMENTS_API_URL,
            headers=headers,
            json={"query": _JWT_PROBE_QUERY},
            timeout=timeout,
        )
    except requests.exceptions.RequestException as e:
        # Network failure — don't block the save, but warn. The token may be
        # fine and the API may simply be unreachable right now.
        return False, f"Could not reach the assessments API to verify the JWT: {e}"
    if resp.status_code in (401, 403):
        return False, (
            f"The assessments API rejected this JWT (HTTP {resp.status_code}). "
            "It is likely expired or not a staff token — grab a fresh one and "
            "try again."
        )
    if resp.status_code != 200:
        preview = (resp.text or "")[:200]
        return False, (
            f"Unexpected HTTP {resp.status_code} from the assessments API while "
            f"verifying the JWT. Preview: {preview!r}"
        )
    try:
        body = resp.json()
    except Exception:
        return False, "The assessments API returned a non-JSON response while verifying the JWT."
    if body.get("errors"):
        return False, (
            "Error: JWT token not accepted. Check to see if you are using an "
            "expired JWT token."
        )
    return True, "JWT verified against the assessments API."


def render_jwt_sidebar():
    """Sidebar widget that lets a staff user paste their own Udacity JWT.

    Call this once per page that needs authenticated API access. The token is
    stored in ``st.session_state`` so it persists across pages for the rest
    of the browser session without re-entry. Shows a fingerprint + email
    hint once set, and a Clear button to wipe it.
    """
    with st.sidebar:
        st.markdown("### Udacity staff JWT")
        current = get_udacity_jwt()
        if current:
            st.success(f"JWT set (sha256[:10] `{_jwt_fingerprint(current)}`)")
            # Try to surface the user's email from the JWT payload (no
            # signature verification — purely informational).
            try:
                payload_b64 = current.split('.')[1]
                # Pad base64url to a multiple of 4 for decode.
                payload_b64 += '=' * (-len(payload_b64) % 4)
                payload = json.loads(
                    __import__('base64').urlsafe_b64decode(payload_b64).decode('utf-8')
                )
                email = payload.get('email') or payload.get('sub')
                if email:
                    st.caption(f"Token subject: `{email}`")
            except Exception:
                pass
            if st.button("Clear JWT", use_container_width=True, help="Forget the JWT for this session."):
                clear_udacity_jwt()
                st.rerun()
        else:
            st.info(
                "Paste your **Udacity staff JWT** below. It is stored only in "
                "this browser session (Streamlit session state) and is never "
                "written to disk or secrets."
            )
            new_jwt = st.text_input(
                "Udacity staff JWT",
                value='',
                key="udacity_staff_jwt_input",
                type="password",
                help=(
                    "Your personal Udacity staff JWT. Used as the Bearer token "
                    "for all GraphQL calls against the assessments / "
                    "classroom-content / skills APIs. Expires with your staff "
                    "session, so re-paste when it expires."
                ),
            )
            if st.button("Save JWT", use_container_width=True):
                token = (new_jwt or '').strip()
                if not token:
                    st.error("Please paste a non-empty JWT.")
                else:
                    # Validate the candidate token against the assessments API
                    # before persisting it, so we never store an expired/invalid
                    # JWT and the user gets immediate feedback.
                    with st.spinner("Verifying JWT against the assessments API…"):
                        ok, message = verify_jwt(token)
                    if ok:
                        st.session_state[SESSION_STATE_JWT_KEY] = token
                        st.success("JWT verified and saved for this session.")
                        st.rerun()
                    else:
                        st.error(message)

ASSESSMENTS_API_URL = st.secrets['assessments_api_url']
CLASSROOM_CONTENT_API_URL = st.secrets['classroom_content_api_url']
SKILLS_API_URL = st.secrets['skills_api_url']

PASSWORD = st.secrets['password']

openai_client = OpenAI(
    api_key = st.secrets['openai_api_key']
)

CHAT_COMPLETIONS_MODEL = 'gpt-4o'
CHAT_COMPLETIONS_TEMPERATURE = 0.2
CHAT_COMPLETIONS_RESPONSE_FORMAT = {
    'type': 'json_object'
}

#========================================
#CODING QUESTION TYPE
#========================================

# Assessments API QuestionCategory enum value for coding questions. CODING
# questions are created via a dedicated `createCodingQuestion` mutation (no
# `category` field — it is implicitly CODING), and carry `codingDetails` +
# `testCases` instead of choices.
CODING_CATEGORY = "CODING"

# The only question categories this app supports end-to-end (generate -> review
# -> upload). The assessments API's QuestionCategory enum also has TRUE_FALSE,
# but this app does not author it, so it is excluded. This is the single source
# of truth — import from here, do not redefine.
VALID_QUESTION_CATEGORIES = {"SINGLE_CHOICE", "MULTIPLE_CHOICE", "SHORT_ANSWER", "CODING"}


def infer_category(has_coding=False, has_rubric=False, choices_correct_count=None, has_choices=False):
    """Infer the correct QuestionCategory enum from a question's data structure.

    Used to RECOVER questions whose `category` field is invalid/sloppy (e.g.
    "HARD", "Code", "Deployment") instead of dropping or aborting. The
    structure is authoritative — a question with codingDetails/testCases is
    CODING regardless of what the LLM labelled it; a rubric means SHORT_ANSWER;
    choices with >1 correct answer means MULTIPLE_CHOICE; otherwise SINGLE_CHOICE.

    Args:
        has_coding: question carries codingDetails + testCases.
        has_rubric: question carries a rubric.
        choices_correct_count: number of correct choices (None if no choices).
        has_choices: question has a (possibly empty) choices array.

    Returns one of VALID_QUESTION_CATEGORIES, or None if it cannot be inferred
    (no distinguishing structure at all) — callers should treat None as a hard
    error since there is genuinely nothing to recover from.
    """
    if has_coding:
        return CODING_CATEGORY
    if has_rubric:
        return "SHORT_ANSWER"
    if has_choices or choices_correct_count is not None:
        if choices_correct_count is not None and choices_correct_count > 1:
            return "MULTIPLE_CHOICE"
        return "SINGLE_CHOICE"
    return None


def normalize_category(category, has_coding=False, has_rubric=False, choices_correct_count=None, has_choices=False):
    """Canonical normalizer: return a valid UPPERCASE QuestionCategory enum, or None.

    LITERAL RULE (structure is authoritative, to remove non-determinism):
      1. If the question carries coding fields (codingDetails + testCases) the
         category is CODING — period, even if the label is a valid-but-wrong
         enum like SINGLE_CHOICE.
      2. Else if it carries a rubric, the category is SHORT_ANSWER.
      3. Else if the label is a valid enum (case-insensitive), return it UPPERCASED.
      4. Else infer from the choices structure (>1 correct -> MULTIPLE_CHOICE,
         else SINGLE_CHOICE).
      5. Else None (hard error — nothing to recover from).

    Always returns an uppercase enum string when it returns a string.
    """
    if has_coding:
        return CODING_CATEGORY
    if has_rubric:
        return "SHORT_ANSWER"
    if isinstance(category, str) and category.strip().upper() in VALID_QUESTION_CATEGORIES:
        return category.strip().upper()
    if has_choices or choices_correct_count is not None:
        if choices_correct_count is not None and choices_correct_count > 1:
            return "MULTIPLE_CHOICE"
        return "SINGLE_CHOICE"
    return None


def normalize_question_category(question, choices=None):
    """Convenience wrapper: normalize the category of a question dict.

    Extracts the structure signals (codingDetails/testCases, rubric, choices
    correct-count) from the question/choices and delegates to normalize_category.
    Returns a valid uppercase enum or None.
    """
    if not isinstance(question, dict):
        return None
    has_coding = isinstance(question.get('codingDetails'), dict) and isinstance(question.get('testCases'), list)
    has_rubric = isinstance(question.get('rubric'), dict)
    correct_count = None
    if isinstance(choices, list):
        try:
            correct_count = int(sum(1 for c in choices if c and c.get('isCorrect')))
        except Exception:
            correct_count = 0
    has_choices = isinstance(choices, list) and len(choices) > 0
    return normalize_category(
        question.get('category'),
        has_coding=has_coding,
        has_rubric=has_rubric,
        choices_correct_count=correct_count,
        has_choices=has_choices,
    )

# UI label -> Assessments API `CodingLanguage` enum. Only these 5 are wired up
# for execution in assessments-api (internal/services/code_execution); the
# other enum values (TYPESCRIPT, HTML, CSS, SHELL) are authoring-only.
CODING_LANGUAGE_OPTIONS = {
    "Python": "PYTHON",
    "Sql": "SQL",
    "Java": "JAVA",
    "Javascript": "JAVASCRIPT",
    "C++": "CPP",
}

# Reverse lookup: API enum -> UI label (for normalizing imported/LLM values).
CODING_LANGUAGE_ENUM_TO_LABEL = {v: k for k, v in CODING_LANGUAGE_OPTIONS.items()}

# Default execution limits. The API accepts timeLimitMs in [100, 30000] and
# memoryLimitMb in [16, 512]; these match the API's own defaults.
CODING_DEFAULT_TIME_LIMIT_MS = 5000
CODING_DEFAULT_MEMORY_LIMIT_MB = 256

# Minimum number of test cases per CODING question. The API accepts >=1, but we
# require >=3 at generation time (matching the prompt's "at least 3 test cases"
# rule) so the dry-run has meaningful coverage and thin suites can't slip through.
CODING_MIN_TEST_CASES = 3

#========================================
#CONTEXT MANAGEMENT UTILITIES
#========================================

# GPT-4o context limits.
GPT4O_MAX_TOKENS = 128000
# Tokens we reserve for the model's response so a large input can't crowd out
# the completion. gpt-4o supports up to ~16k output tokens.
RESERVED_OUTPUT_TOKENS = 16000
# Small buffer for per-message structural overhead.
SAFETY_BUFFER_TOKENS = 1000
# Maximum input tokens we will send in a single request.
GPT4O_MAX_INPUT_TOKENS = GPT4O_MAX_TOKENS - RESERVED_OUTPUT_TOKENS - SAFETY_BUFFER_TOKENS

# Lazily-initialised tiktoken encoder so we count tokens accurately instead of
# relying on a characters-per-token heuristic (which badly underestimates dense
# content like code, JSON and VTT transcripts).
_TOKEN_ENCODER = None

def _get_token_encoder():
    global _TOKEN_ENCODER
    if _TOKEN_ENCODER is not None:
        return _TOKEN_ENCODER
    try:
        import tiktoken
        try:
            _TOKEN_ENCODER = tiktoken.encoding_for_model(CHAT_COMPLETIONS_MODEL)
        except Exception:
            # o200k_base is the encoding used by gpt-4o / gpt-4o-mini.
            _TOKEN_ENCODER = tiktoken.get_encoding('o200k_base')
    except Exception as e:
        print(f"tiktoken unavailable, falling back to heuristic token counting: {e}")
        _TOKEN_ENCODER = False  # Sentinel: use heuristic.
    return _TOKEN_ENCODER

def estimate_tokens(text):
    """
    Count the number of tokens in a text string.

    Uses tiktoken for an accurate count when available, and falls back to a
    conservative ~3 characters/token heuristic otherwise.
    """
    if not text:
        return 0
    encoder = _get_token_encoder()
    if encoder:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass
    # Conservative fallback (dense content can be well under 4 chars/token).
    return len(text) // 3 + 1

def estimate_messages_tokens(messages):
    """
    Estimate the total number of tokens in a messages array.
    Includes overhead for message formatting.
    """
    total_tokens = 0
    for message in messages:
        # Base tokens for message structure
        total_tokens += 4
        # Content tokens
        content = message.get('content', '')
        total_tokens += estimate_tokens(content)
        # Role tokens
        role = message.get('role', '')
        total_tokens += estimate_tokens(role)
        # Additional overhead
        total_tokens += 2
    return total_tokens

def chunk_content(content, max_chunk_tokens, overlap=200):
    """
    Split large content into chunks that each fit within a token budget.

    Splitting is performed on real token boundaries (via tiktoken when
    available) so every chunk is guaranteed to be at or below
    ``max_chunk_tokens``. When tiktoken is unavailable we fall back to a
    character-based split using the same conservative ratio as
    ``estimate_tokens``.

    Args:
        content: The content to chunk
        max_chunk_tokens: Maximum tokens allowed per chunk
        overlap: Number of tokens to overlap between consecutive chunks

    Returns:
        List of content chunks
    """
    if not content:
        return ['']

    max_chunk_tokens = max(1, int(max_chunk_tokens))
    overlap = max(0, min(int(overlap), max_chunk_tokens - 1))

    encoder = _get_token_encoder()

    if encoder:
        tokens = encoder.encode(content)
        if len(tokens) <= max_chunk_tokens:
            return [content]
        chunks = []
        start = 0
        step = max_chunk_tokens - overlap
        while start < len(tokens):
            window = tokens[start:start + max_chunk_tokens]
            chunk = encoder.decode(window).strip()
            if chunk:
                chunks.append(chunk)
            if start + max_chunk_tokens >= len(tokens):
                break
            start += step
        return chunks if chunks else [content]

    # Heuristic fallback: approximate the token budget in characters.
    max_chars = max_chunk_tokens * 3
    overlap_chars = overlap * 3

    if len(content) <= max_chars:
        return [content]

    chunks = []
    start = 0
    while start < len(content):
        end = start + max_chars
        if end < len(content):
            search_start = max(start, end - 1000)
            sentence_end = content.rfind('. ', search_start, end)
            if sentence_end > start:
                end = sentence_end + 1
        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(content):
            break
        start = end - overlap_chars

    return chunks

def create_chunked_messages(system_content, user_content, max_input_tokens=GPT4O_MAX_INPUT_TOKENS):
    """
    Create messages array with chunked content if needed.
    
    Args:
        system_content: The system message content
        user_content: The user message content
        max_input_tokens: Maximum input tokens allowed
    
    Returns:
        List of message arrays (one for each chunk if chunking is needed)
    """
    # Create base messages
    base_messages = [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': user_content}
    ]
    
    # Check if we need to chunk
    estimated_tokens = estimate_messages_tokens(base_messages)
    
    if estimated_tokens <= max_input_tokens:
        return [base_messages]
    
    # We need to chunk the user content
    print(f"Content too large ({estimated_tokens} tokens), chunking into smaller pieces...")
    
    # Estimate tokens for system message
    system_tokens = estimate_tokens(system_content) + 10  # Add overhead
    
    # Calculate available tokens for user content (budget is in TOKENS).
    available_tokens = max_input_tokens - system_tokens - 20  # Buffer for message structure
    if available_tokens < 1000:
        # System prompt alone is large; keep a minimal floor so we still split.
        available_tokens = 1000
    
    # Chunk the user content on real token boundaries.
    chunks = chunk_content(user_content, available_tokens, overlap=200)
    
    # Create message arrays for each chunk
    message_arrays = []
    for i, chunk in enumerate(chunks):
        chunk_messages = [
            {'role': 'system', 'content': system_content},
            {'role': 'user', 'content': chunk}
        ]
        message_arrays.append(chunk_messages)
        print(f"Created chunk {i+1}/{len(chunks)} with ~{estimate_messages_tokens(chunk_messages)} tokens")
    
    return message_arrays

def call_openai_with_context_management(messages, **kwargs):
    """
    Call OpenAI API with automatic context management for GPT-4o.
    
    Args:
        messages: List of message dictionaries
        **kwargs: Additional arguments for the OpenAI API call
    
    Returns:
        Combined response from all chunks if chunking was needed
    """
    # Extract system and user content from messages
    system_message = next((msg for msg in messages if msg['role'] == 'system'), None)
    user_message = next((msg for msg in messages if msg['role'] == 'user'), None)
    
    if not system_message or not user_message:
        # Fall back to direct API call if message structure is unexpected
        return openai_client.chat.completions.create(messages=messages, **kwargs)
    
    # Create chunked messages if needed
    message_arrays = create_chunked_messages(
        system_message['content'], 
        user_message['content']
    )
    
    if len(message_arrays) == 1:
        # No chunking needed, make direct API call
        return openai_client.chat.completions.create(messages=message_arrays[0], **kwargs)
    
    # Multiple chunks needed - process each chunk and combine results
    print(f"Processing {len(message_arrays)} chunks...")
    
    all_responses = []
    for i, chunk_messages in enumerate(message_arrays):
        print(f"Processing chunk {i+1}/{len(message_arrays)}...")
        try:
            response = openai_client.chat.completions.create(messages=chunk_messages, **kwargs)
            all_responses.append(response)
            # Add small delay between chunks to avoid rate limiting
            if i < len(message_arrays) - 1:
                time.sleep(0.5)
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            # Continue with other chunks
            continue
    
    if not all_responses:
        raise Exception("All chunks failed to process")
    
    if len(all_responses) == 1:
        return all_responses[0]
    
    # Merge the JSON payloads from every chunk into a single response so callers
    # see the combined result (e.g. all learning objectives / all questions).
    contents = []
    for response in all_responses:
        try:
            contents.append(response.choices[0].message.content)
        except Exception:
            continue
    
    merged_content = _merge_json_contents(contents)
    print(f"Merged {len(all_responses)} chunk responses into a single result.")
    return _build_synthetic_response(merged_content)

def _merge_json_contents(contents):
    """
    Merge the JSON string payloads returned for each content chunk.

    List-valued keys (e.g. ``objectives``, ``questions_choices``) are
    concatenated across chunks; identical string entries are de-duplicated while
    preserving order. Scalar keys (e.g. ``title``) take the first non-empty
    value. If nothing parses as JSON, the first raw payload is returned.
    """
    parsed = []
    for content in contents:
        if not content:
            continue
        cleaned = content.replace('```json', '').replace('```', '').strip()
        try:
            parsed.append(json.loads(cleaned))
        except Exception:
            continue
    
    if not parsed:
        return contents[0] if contents else '{}'
    if len(parsed) == 1:
        return json.dumps(parsed[0])
    
    merged = {}
    for obj in parsed:
        if not isinstance(obj, dict):
            continue
        for key, value in obj.items():
            if isinstance(value, list):
                bucket = merged.setdefault(key, [])
                if isinstance(bucket, list):
                    bucket.extend(value)
                else:
                    merged[key] = value
            elif key not in merged or merged[key] in (None, '', [], {}):
                merged[key] = value
    
    # De-duplicate plain string lists (e.g. learning objectives) in place.
    for key, value in merged.items():
        if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
            seen = set()
            deduped = []
            for item in value:
                if item not in seen:
                    seen.add(item)
                    deduped.append(item)
            merged[key] = deduped
    
    return json.dumps(merged)

def _build_synthetic_response(content):
    """
    Wrap a content string in an object that mimics the parts of the OpenAI
    chat-completion response that callers in this codebase actually read
    (``response.choices[0].message.content``).
    """
    from types import SimpleNamespace
    message = SimpleNamespace(content=content, role='assistant', tool_calls=None, function_call=None)
    choice = SimpleNamespace(message=message, finish_reason='stop', index=0, logprobs=None)
    return SimpleNamespace(choices=[choice], usage=None, id='chunked-merged-response', object='chat.completion')

def call_openai_with_fallback(messages, **kwargs):
    """
    Call OpenAI API with fallback strategies for handling large contexts.
    
    This function implements a more sophisticated approach that can:
    1. Try the original request
    2. If it fails due to context length, chunk the content
    3. If chunking fails, try with reduced content
    4. If all else fails, raise an exception
    
    Args:
        messages: List of message dictionaries
        **kwargs: Additional arguments for the OpenAI API call
    
    Returns:
        OpenAI API response
    """
    try:
        # First, try the original request
        return openai_client.chat.completions.create(messages=messages, **kwargs)
    except Exception as e:
        error_message = str(e).lower()
        
        # Check if it's a context length error
        if any(keyword in error_message for keyword in [
            'context_length', 'context length', 'maximum context',
            'token_limit', 'too many tokens', 'reduce the length',
        ]):
            print(f"Context length exceeded, attempting to chunk content: {e}")
            
            try:
                # Try with context management
                return call_openai_with_context_management(messages, **kwargs)
            except Exception as chunk_error:
                print(f"Chunking failed: {chunk_error}")
                
                # As a last resort, try with truncated content
                try:
                    return _call_openai_with_truncated_content(messages, **kwargs)
                except Exception as truncate_error:
                    print(f"Truncation also failed: {truncate_error}")
                    raise e  # Re-raise the original error
        else:
            # Not a context length error, re-raise
            raise e

def _truncate_text_to_tokens(text, max_tokens):
    """Truncate text to at most ``max_tokens`` tokens, on real token boundaries."""
    encoder = _get_token_encoder()
    if encoder:
        tokens = encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return encoder.decode(tokens[:max_tokens])
    # Heuristic fallback mirrors estimate_tokens (~3 chars/token).
    return text[:max_tokens * 3]

def _call_openai_with_truncated_content(messages, max_tokens_per_message=None, **kwargs):
    """
    Last-resort fallback that truncates content to fit within token limits.

    Truncation loses information, so this only runs after chunking has failed.
    The budget is derived from the model's input limit minus the other (e.g.
    system) messages so the request is guaranteed to fit.
    
    Args:
        messages: List of message dictionaries
        max_tokens_per_message: Optional explicit cap for the largest message
        **kwargs: Additional arguments for the OpenAI API call
    
    Returns:
        OpenAI API response
    """
    # Tokens consumed by everything except the single largest message.
    non_max_tokens = estimate_messages_tokens(messages)
    largest_idx = None
    largest_tokens = -1
    for idx, message in enumerate(messages):
        tokens = estimate_tokens(message.get('content', ''))
        if tokens > largest_tokens:
            largest_tokens = tokens
            largest_idx = idx
    
    if largest_idx is not None:
        non_max_tokens -= largest_tokens
    
    budget = GPT4O_MAX_INPUT_TOKENS - max(0, non_max_tokens)
    if max_tokens_per_message is not None:
        budget = min(budget, max_tokens_per_message)
    budget = max(500, budget)
    
    truncated_messages = []
    for idx, message in enumerate(messages):
        content = message.get('content', '')
        estimated = estimate_tokens(content)
        if idx == largest_idx and estimated > budget:
            truncated_content = _truncate_text_to_tokens(content, budget) + "\n\n[Content truncated due to length...]"
            truncated_messages.append({'role': message['role'], 'content': truncated_content})
            print(f"Truncated {message['role']} message from {estimated} to ~{budget} tokens")
        else:
            truncated_messages.append(message)
    
    return openai_client.chat.completions.create(messages=truncated_messages, **kwargs)
