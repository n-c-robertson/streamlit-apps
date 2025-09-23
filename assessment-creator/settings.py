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
UDACITY_JWT = st.secrets['jwt_token']

def production_headers():
    STAFF_HEADERS = {
        'Authorization': f'Bearer {UDACITY_JWT}',
        'Content-Type': 'application/json'
    }
    return STAFF_HEADERS

ASSESSMENTS_API_URL = st.secrets['assessments_api_url']
CLASSROOM_CONTENT_API_URL = st.secrets['classroom_content_api_url']
SKILLS_API_URL = st.secrets['skills_api_url']

PASSWORD = st.secrets['password']

openai_client = OpenAI(
    api_key = st.secrets['openai_api_key']
)

CHAT_COMPLETIONS_MODEL = 'gpt-5'
CHAT_COMPLETIONS_TEMPERATURE = 0.2
CHAT_COMPLETIONS_RESPONSE_FORMAT = {
    'type': 'json_object'
}

#========================================
#CONTEXT MANAGEMENT UTILITIES
#========================================

# GPT-4o context limits (approximate)
GPT4O_MAX_TOKENS = 128000
GPT4O_MAX_INPUT_TOKENS = 100000  # Conservative estimate for input tokens

def estimate_tokens(text):
    """
    Estimate the number of tokens in a text string.
    This is a rough approximation: ~4 characters per token for English text.
    """
    if not text:
        return 0
    return len(text) // 4

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

def chunk_content(content, max_chunk_size=80000, overlap=1000):
    """
    Split large content into overlapping chunks.
    
    Args:
        content: The content to chunk
        max_chunk_size: Maximum tokens per chunk
        overlap: Number of tokens to overlap between chunks
    
    Returns:
        List of content chunks
    """
    if not content:
        return ['']
    
    # Convert token limit to character limit (rough approximation)
    max_chars = max_chunk_size * 4
    overlap_chars = overlap * 4
    
    if len(content) <= max_chars:
        return [content]
    
    chunks = []
    start = 0
    
    while start < len(content):
        end = start + max_chars
        
        # If this isn't the last chunk, try to break at a sentence boundary
        if end < len(content):
            # Look for sentence endings within the last 1000 characters
            search_start = max(start, end - 1000)
            sentence_end = content.rfind('. ', search_start, end)
            if sentence_end > start:
                end = sentence_end + 1
        
        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position, accounting for overlap
        start = end - overlap_chars
        if start >= len(content):
            break
    
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
    
    # Calculate available tokens for user content
    available_tokens = max_input_tokens - system_tokens - 20  # Buffer for message structure
    
    # Convert to character limit
    max_chars = available_tokens * 4
    
    # Chunk the user content
    chunks = chunk_content(user_content, max_chars, overlap=1000)
    
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
    
    # For now, return the first successful response
    # In a more sophisticated implementation, you might want to combine results
    # based on the specific use case (e.g., merging learning objectives, combining questions)
    return all_responses[0]

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
        if any(keyword in error_message for keyword in ['context_length', 'token_limit', 'too many tokens']):
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

def _call_openai_with_truncated_content(messages, max_tokens_per_message=50000, **kwargs):
    """
    Fallback method that truncates content to fit within token limits.
    
    Args:
        messages: List of message dictionaries
        max_tokens_per_message: Maximum tokens to allow per message
        **kwargs: Additional arguments for the OpenAI API call
    
    Returns:
        OpenAI API response
    """
    truncated_messages = []
    
    for message in messages:
        content = message.get('content', '')
        estimated_tokens = estimate_tokens(content)
        
        if estimated_tokens > max_tokens_per_message:
            # Truncate content
            max_chars = max_tokens_per_message * 4
            truncated_content = content[:max_chars] + "\n\n[Content truncated due to length...]"
            truncated_messages.append({
                'role': message['role'],
                'content': truncated_content
            })
            print(f"Truncated {message['role']} message from {estimated_tokens} to ~{max_tokens_per_message} tokens")
        else:
            truncated_messages.append(message)
    
    return openai_client.chat.completions.create(messages=truncated_messages, **kwargs)
