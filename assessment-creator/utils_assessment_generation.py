#========================================
#IMPORT PACKAGES
#========================================

import ast
import concurrent.futures
import hashlib
import json
import os
import pickle
import random
import re
import requests
import sys
import time
import traceback
from collections import Counter
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import graphql_queries
import settings
import prompts

#========================================
# FUNCTIONS
#========================================

def format_exception_details(e):
    """
    Format exception details for display in Streamlit.
    Returns a formatted string with file, line, and traceback information.
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    
    # Get the most recent frame (where the exception occurred)
    tb = traceback.extract_tb(exc_traceback)
    if tb:
        filename = tb[-1].filename
        line_number = tb[-1].lineno
        function_name = tb[-1].name
        line_content = tb[-1].line
    else:
        filename = "Unknown"
        line_number = "Unknown"
        function_name = "Unknown"
        line_content = "Unknown"
    
    details = f"""
**Exception Details:**
- **Type:** {type(e).__name__}
- **Message:** {str(e)}
- **File:** {filename}
- **Line:** {line_number}
- **Function:** {function_name}
- **Code:** `{line_content}`

**Full Traceback:**
```
{traceback.format_exc()}
```
"""
    return details

def prep_program_keys(PROGRAM_KEYS):

    # 1. Split into lines and filter out any blank lines:
    keys = PROGRAM_KEYS.replace(' ', '').split(',')

    # 2. Build the desired structure with empty titles:
    section_content_definitions = [
        {
        'title': '',
        'content_keys': [key],
        'content_ids': []
        }
        for key in keys
    ]

    return section_content_definitions


def fetch_readiness_lessons_from_skills_api(prerequisite_skills):
    """
    Fetch lessons from Skills API based on prerequisite skills for readiness assessment.
    
    Args:
        prerequisite_skills: List of skill names to search for
        
    Returns:
        List of lesson IDs to use for question generation
    """
    # Skills API endpoint - use the correct endpoint for search (not vector search)
    url = "https://skills.udacity.com/api/skills/search"
    
    # Headers for API call
    headers = {
        'content-type': 'application/json',
        'Authorization': f'Bearer {settings.UDACITY_JWT}',
        'Accept': 'application/json'
    }
    
    # JSON payload
    payload = {
        'search': prerequisite_skills,
        'searchField': "knowledge_component_names",
        'filter': {
            'Difficulty': {'$in': ['Beginner']}
        }
    }
    
    print(f"\n=== SKILLS API DEBUG ===")
    print(f"URL: {settings.SKILLS_API_URL}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        # Send POST request
        response = requests.post(settings.SKILLS_API_URL, headers=headers, data=json.dumps(payload))
        print(f"Response status: {response.status_code}")
        
        response.raise_for_status()
        
        api_response = response.json()
        print(f"API response type: {type(api_response)}")
        print(f"API response length: {len(api_response) if isinstance(api_response, list) else 'not a list'}")
        
        if isinstance(api_response, list) and len(api_response) > 0:
            print(f"First item keys: {list(api_response[0].keys()) if api_response[0] else 'empty item'}")
        
        # Extract lesson IDs from the response
        lesson_ids = []
        for item in api_response:
            if 'lesson' in item and 'content' in item['lesson'] and 'id' in item['lesson']['content']:
                lesson_id = item['lesson']['content']['id']
                lesson_ids.append(lesson_id)
                print(f"Found lesson ID: {lesson_id}")
        
        print(f"Total lesson IDs found: {len(lesson_ids)}")
        print(f"=== END SKILLS API DEBUG ===\n")
        
        return lesson_ids
        
    except requests.RequestException as e:
        print(f"Skills API request error: {e}")
        return []
    except Exception as e:
        print(f"Skills API processing error: {e}")
        return []

def add_program_data(section_content_definitions, assessment_type="placement"):

    missing_prerequisite_skills = []
    
    for section in section_content_definitions:
        section['difficulty_level'] = {}
        section['skills']           = {}
        section['readiness_lesson_ids'] = []  # New field for readiness lessons

        for key in section['content_keys']:
            release = graphql_queries.query_component(key)
            if not release:
                continue

            # 1) root_node_id may not exist — fall back to root_node.id
            root_id = release.get('root_node_id') \
                or release.get('root_node', {}).get('id')
            
            # 2) extract the title once, from the root_node key
            if not section['title']:
                section['title'] = release.get('root_node', {}) \
                                        .get('title', '')

            # 3) pull metadata safely
            comp = release.get('component')
            meta = (comp or {}).get('metadata')
            if not meta:
                continue

            section['difficulty_level'][key] = meta.get('difficulty_level')
            
            # Handle assessment type logic
            if assessment_type.lower() == "readiness":
                # For readiness assessment, check for prerequisite skills
                prerequisite_skills = meta.get('prerequisite_skills', [])
                
                print(f"\n=== SKILL TRACKING: Program Key {key} ===")
                print(f"Assessment Type: {assessment_type}")
                print(f"Raw prerequisite_skills from metadata: {prerequisite_skills}")
                
                if not prerequisite_skills:
                    # Return error if no prerequisite skills found
                    raise ValueError(f"No prerequisite skills found for key {key}. Please go to Studio and add prerequisite skills to generate a readiness assessment.")
                
                # Extract skill names from prerequisite skills
                skill_names = [skill.get('name', '') for skill in prerequisite_skills if skill.get('name')]
                
                print(f"Extracted skill names: {skill_names}")
                print(f"Full prerequisite skill objects:")
                for i, skill in enumerate(prerequisite_skills):
                    print(f"  {i+1}. name: '{skill.get('name', 'MISSING')}', uri: '{skill.get('uri', 'MISSING')}'")
                
                if not skill_names:
                    raise ValueError(f"No valid prerequisite skill names found for key {key}. Please check the prerequisite skills in Studio.")
                
                # Fetch lessons from Skills API based on prerequisite skills
                lesson_ids = fetch_readiness_lessons_from_skills_api(skill_names)
                
                # Store the lesson IDs and prerequisite skills
                section['readiness_lesson_ids'].extend(lesson_ids)
                section['skills'][key] = prerequisite_skills
                
            else:
                # For placement assessment, use original logic
                if root_id is not None:
                    section['content_ids'].append(root_id)
                
                teaches_skills = meta.get('teaches_skills')
                print(f"\n=== SKILL TRACKING: Program Key {key} (Placement) ===")
                print(f"Assessment Type: {assessment_type}")
                print(f"Raw teaches_skills from metadata: {teaches_skills}")
                
                if teaches_skills:
                    print(f"Teaches skill objects:")
                    for i, skill in enumerate(teaches_skills):
                        print(f"  {i+1}. name: '{skill.get('name', 'MISSING')}', uri: '{skill.get('uri', 'MISSING')}'")
                
                section['skills'][key] = teaches_skills

    return section_content_definitions, missing_prerequisite_skills

def add_node_data(section_content_definitions, assessment_type="placement"):
    
    for section in section_content_definitions:
        section['nodes'] = {}
        
        if assessment_type.lower() == "readiness":
            # For readiness assessment, process lesson IDs from Skills API
            for lesson_id in section.get('readiness_lesson_ids', []):
                try:
                    node_data = graphql_queries.query_node(lesson_id)
                    if node_data is not None:
                        # Use lesson_id as key for readiness lessons
                        section['nodes'][str(lesson_id)] = node_data
                except Exception:
                    continue
        else:
            # For placement assessment, use original logic
            # assume section['content_ids'] and section['content_keys'] line up one-to-one
            for key, node_id in zip(section['content_keys'], section['content_ids']):
                node_data = graphql_queries.query_node(node_id)
                if node_data is not None:
                    section['nodes'][key] = node_data
    
    return section_content_definitions

def detect_coding_content(text):
    """
    Detect if text contains coding content by looking for common programming patterns.
    Returns True if coding content is detected, False otherwise.
    """
    if not text:
        return False
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Common programming keywords and patterns
    programming_keywords = [
        'function', 'def ', 'class ', 'import ', 'from ', 'return', 'if ', 'else:', 'elif ',
        'for ', 'while ', 'try:', 'except:', 'finally:', 'with ', 'as ', 'in ',
        'var ', 'let ', 'const ', 'function(', '=>', 'console.log', 'print(',
        'public ', 'private ', 'protected ', 'static ', 'void ', 'int ', 'string ',
        'boolean', 'array', 'list', 'dict', 'object', 'null', 'undefined',
        'true', 'false', 'this', 'super', 'new ', 'extends', 'implements'
    ]
    
    # Code block patterns
    code_patterns = [
        r'```\w*',  # Code blocks
        r'`[^`]+`',  # Inline code
        r'<code[^>]*>',  # HTML code tags
        r'console\.',  # Console methods
        r'\.log\(',  # Logging
        r'\.get\(',  # HTTP methods
        r'\.post\(',  # HTTP methods
        r'\.put\(',  # HTTP methods
        r'\.delete\(',  # HTTP methods
        r'\.query\(',  # Database queries
        r'\.execute\(',  # Execution methods
        r'\.render\(',  # Rendering methods
        r'\.map\(',  # Array methods
        r'\.filter\(',  # Array methods
        r'\.reduce\(',  # Array methods
        r'\.forEach\(',  # Array methods
        r'\.push\(',  # Array methods
        r'\.pop\(',  # Array methods
        r'\.shift\(',  # Array methods
        r'\.unshift\(',  # Array methods
        r'\.slice\(',  # Array methods
        r'\.splice\(',  # Array methods
        r'\.sort\(',  # Array methods
        r'\.reverse\(',  # Array methods
        r'\.join\(',  # Array methods
        r'\.split\(',  # String methods
        r'\.replace\(',  # String methods
        r'\.toLowerCase\(',  # String methods
        r'\.toUpperCase\(',  # String methods
        r'\.trim\(',  # String methods
        r'\.substring\(',  # String methods
        r'\.indexOf\(',  # String methods
        r'\.includes\(',  # String methods
        r'\.startsWith\(',  # String methods
        r'\.endsWith\(',  # String methods
        r'\.charAt\(',  # String methods
        r'\.charCodeAt\(',  # String methods
        r'\.parseInt\(',  # Number methods
        r'\.parseFloat\(',  # Number methods
        r'\.toString\(',  # Object methods
        r'\.hasOwnProperty\(',  # Object methods
        r'\.keys\(',  # Object methods
        r'\.values\(',  # Object methods
        r'\.entries\(',  # Object methods
        r'\.assign\(',  # Object methods
        r'\.create\(',  # Object methods
        r'\.freeze\(',  # Object methods
        r'\.seal\(',  # Object methods
        r'\.preventExtensions\(',  # Object methods
        r'\.isFrozen\(',  # Object methods
        r'\.isSealed\(',  # Object methods
        r'\.isExtensible\(',  # Object methods
        r'\.getPrototypeOf\(',  # Object methods
        r'\.setPrototypeOf\(',  # Object methods
        r'\.defineProperty\(',  # Object methods
        r'\.defineProperties\(',  # Object methods
        r'\.getOwnPropertyDescriptor\(',  # Object methods
        r'\.getOwnPropertyNames\(',  # Object methods
        r'\.getOwnPropertySymbols\(',  # Object methods
        r'\.is\(',  # Object methods
        r'\.preventExtensions\(',  # Object methods
        r'\.seal\(',  # Object methods
        r'\.freeze\(',  # Object methods
        r'\.create\(',  # Object methods
        r'\.assign\(',  # Object methods
        r'\.entries\(',  # Object methods
        r'\.values\(',  # Object methods
        r'\.keys\(',  # Object methods
        r'\.hasOwnProperty\(',  # Object methods
        r'\.toString\(',  # Object methods
        r'\.parseFloat\(',  # Number methods
        r'\.parseInt\(',  # Number methods
        r'\.charCodeAt\(',  # String methods
        r'\.charAt\(',  # String methods
        r'\.endsWith\(',  # String methods
        r'\.startsWith\(',  # String methods
        r'\.includes\(',  # String methods
        r'\.indexOf\(',  # String methods
        r'\.substring\(',  # String methods
        r'\.trim\(',  # String methods
        r'\.toUpperCase\(',  # String methods
        r'\.toLowerCase\(',  # String methods
        r'\.replace\(',  # String methods
        r'\.split\(',  # String methods
        r'\.join\(',  # Array methods
        r'\.reverse\(',  # Array methods
        r'\.sort\(',  # Array methods
        r'\.splice\(',  # Array methods
        r'\.slice\(',  # Array methods
        r'\.unshift\(',  # Array methods
        r'\.shift\(',  # Array methods
        r'\.pop\(',  # Array methods
        r'\.push\(',  # Array methods
        r'\.forEach\(',  # Array methods
        r'\.reduce\(',  # Array methods
        r'\.filter\(',  # Array methods
        r'\.map\(',  # Array methods
        r'\.render\(',  # Rendering methods
        r'\.execute\(',  # Execution methods
        r'\.query\(',  # Database queries
        r'\.delete\(',  # HTTP methods
        r'\.put\(',  # HTTP methods
        r'\.post\(',  # HTTP methods
        r'\.get\(',  # HTTP methods
        r'\.log\(',  # Logging
        r'console\.',  # Console methods
        r'<code[^>]*>',  # HTML code tags
        r'`[^`]+`',  # Inline code
        r'```\w*'  # Code blocks
    ]
    
    # Check for programming keywords
    for keyword in programming_keywords:
        if keyword in text_lower:
            return True
    
    # Check for code patterns using regex
    for pattern in code_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    # Check for common programming file extensions mentioned
    file_extensions = ['.py', '.js', '.java', '.cpp', '.c', '.html', '.css', '.php', '.rb', '.go', '.rs', '.swift', '.kt']
    for ext in file_extensions:
        if ext in text_lower:
            return True
    
    return False 

def process_node(node):
    """
    Recursively traverse a node (or list of nodes) to fetch VTT content for each VideoAtom.
    This function updates the node in place and marks nodes with coding content.
    """

    def remove_timestamps(text):
        # Regular expression to match time stamps of the format "00:00:09.525 --> 00:00:12.850"
        timestamp_pattern = r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}'
        cleaned_text = re.sub(timestamp_pattern, '', text)
        cleaned_text = re.sub(r'\n+', '\n', cleaned_text).strip()
        return cleaned_text

    if isinstance(node, dict):
        # Initialize coding content flag for this node
        node['has_coding_content'] = False
        
        for key, value in node.items():
            if key == "atoms":
                for atom in value:
                    if atom and atom.get("semantic_type") == "VideoAtom" and "video" in atom:
                        video = atom["video"]
                        if "vtt_url" in video:
                            vtt_url = video["vtt_url"]
                            if vtt_url.startswith("//"):
                                vtt_url = "https:" + vtt_url
                            try:
                                response = requests.get(vtt_url)
                                response.raise_for_status()
                                vtt_json = response.json()
                                vtt_file_url = vtt_json.get("en-us")
                                if vtt_file_url:
                                    if vtt_file_url.startswith("//"):
                                        vtt_file_url = "https:" + vtt_file_url
                                    vtt_response = requests.get(vtt_file_url)
                                    vtt_response.raise_for_status()
                                    vtt_content = remove_timestamps(vtt_response.text)
                                    video["vtt"] = vtt_content
                                    
                                    # Check for coding content in VTT
                                    if detect_coding_content(vtt_content):
                                        node['has_coding_content'] = True
                                        atom['has_coding_content'] = True
                            except requests.RequestException:
                                continue
                    elif atom and atom.get("semantic_type") == "TextAtom":
                        # Check for coding content in text atoms
                        text_content = atom.get("text", "")
                        if detect_coding_content(text_content):
                            node['has_coding_content'] = True
                            atom['has_coding_content'] = True
            elif isinstance(value, (dict, list)):
                process_node(value)
    elif isinstance(node, list):
        for item in node:
            process_node(item)

def process_nodes(section_content_definitions, assessment_type="placement"):
    # Create a list of nodes to process based on assessment type
    all_nodes = []
    for section in section_content_definitions:
        if assessment_type.lower() == "readiness":
            # For readiness, only process lesson nodes from Skills API
            for lesson_key, node in section.get('nodes', {}).items():
                if node:
                    all_nodes.append(node)
        else:
            # For placement, process original program key nodes
            for key, node in section.get('nodes', {}).items():
                if node:
                    all_nodes.append(node)

    # Process all nodes concurrently.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_node, node) for node in all_nodes]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception:
                continue
    
    return all_nodes

def extract_content_helper(node):
    """
    Traverse a part node and extract lesson titles, concept titles,
    content from text and video atoms, plus all quizzes.
    
    Parameters:
      node (dict): A dictionary representing a Part with 'modules'.
    
    Returns:
      str: A concatenated string of all extracted information.
    """
    parts = []
    for module in node.get("modules", []):
        module_title = module.get("title", "Unnamed Module")
        parts.append(f"Module: {module_title}")
        for lesson in module.get("lessons", []):
            lesson_title = lesson.get("title", "Unnamed Lesson")
            parts.append(f"  Lesson: {lesson_title}")
            for concept in lesson.get("concepts", []):
                concept_title = concept.get("title", "Unnamed Concept")
                parts.append(f"    Concept: {concept_title}")
                for atom in concept.get("atoms", []):
                    stype = atom.get("semantic_type", "")
                    # TextAtom
                    if stype == "TextAtom":
                        text = atom.get('text', "").strip()
                        if text:
                            parts.append(f"      Text: {text}")
                    # VideoAtom
                    elif stype == "VideoAtom":
                        if atom['title']:
                            vtitle = atom.get("title", "").strip()
                            vtt = atom.get("video", {}).get("vtt_url", "").strip()
                            if vtitle:
                                parts.append(f"      Video Title: {vtitle}")
                            if vtt:
                                parts.append(f"      Video Captions: {vtt}")
                    # RadioQuizAtom
                    elif stype == "RadioQuizAtom":
                        q = atom["question"]
                        parts.append(f"      [RadioQuiz] Prompt: {q.get('prompt')}")
                        for ans in q.get("answers", []):
                            parts.append(f"        - ({'✔' if ans['is_correct'] else '✘'}) {ans['text']}")
                    # CheckboxQuizAtom
                    elif stype == "CheckboxQuizAtom":
                        q = atom["question"]
                        parts.append(f"      [CheckboxQuiz] Prompt: {q.get('prompt')}")
                        parts.append(f"        * Feedback: {q.get('correct_feedback')}")
                        for ans in q.get("answers", []):
                            parts.append(f"        - ({'✔' if ans['is_correct'] else '✘'}) {ans['text']}")
                    # MatchingQuizAtom
                    elif stype == "MatchingQuizAtom":
                        q = atom["question"]
                        parts.append(f"      [MatchingQuiz] {q.get('complex_prompt', {}).get('text')}")
                        parts.append(f"        Answers Label: {q.get('answers_label')}")
                        parts.append(f"        Concepts Label: {q.get('concepts_label')}")
                        parts.append("        Pairs:")
                        for c in q.get("concepts", []):
                            ca = c.get("correct_answer", {}).get("text")
                            parts.append(f"          • {c['text']}  →  {ca}")
                        parts.append("        Available answers:")
                        for a in q.get("answers", []):
                            parts.append(f"          - {a.get('text')}")
    return "\n".join(parts)

def extract_content(section_content_definitions):
    for section in section_content_definitions:
        section['content'] = dict()
        for key in section['nodes']:
            node = section['nodes'][key] 
            section['content'][key] = extract_content_helper(node)
    return section_content_definitions

def learning_objective_generator(section_content_definitions):

    for section in section_content_definitions:

        # Aggregate all content, skills, and difficulty levels for this section.
        all_content = ""
        all_skills = set()
        difficulty_levels = set()

        # For readiness assessments, we have different keys for content vs original program keys
        content_keys = list(section.get('content', {}).keys())  # These are the actual node keys
        
        # Get content from all available nodes
        for content_key in content_keys:
            if content_key in section.get('content', {}):
                all_content += "\n\n" + section['content'][content_key]
        
        # Get skills and difficulty from original program keys
        for program_key in section['content_keys']:
            if program_key in section.get('skills', {}):
                for skill in section['skills'][program_key]:
                    all_skills.add(skill.get('name', ''))
            if program_key in section.get('difficulty_level', {}):
                difficulty = section['difficulty_level'][program_key]
                if difficulty:
                    difficulty_levels.add(difficulty.get('name', ''))

        # Convert sets to sorted lists for clarity.
        difficulties = sorted(list(difficulty_levels))
        skills = sorted(list(all_skills))
        aggregated_content = all_content.strip()
        
        start_completion_time = time.perf_counter()
        
        # Use context management for OpenAI API call
        messages = prompts.get_learning_objectives_prompt(skills, difficulties, aggregated_content)
        
        try:
            chat_completion_response = settings.call_openai_with_fallback(
                model=settings.CHAT_COMPLETIONS_MODEL,
                response_format=settings.CHAT_COMPLETIONS_RESPONSE_FORMAT,
                #temperature=settings.CHAT_COMPLETIONS_TEMPERATURE, #not supported in gpt 5
                messages=messages
                                )
        except Exception as e:
            print(f"Error in learning objectives generation: {e}")
            print(f"Messages: {messages}")
            print(f"Skills: {skills}")
            print(f"Difficulties: {difficulties}")
            print(f"Aggregated content: {aggregated_content}")
            raise e
                            
        section['learning_objectives'] = json.loads(chat_completion_response.choices[0].message.content)
    return section_content_definitions

# Helper function: generate a unique ID for a question based on its content and other fields.
def get_qc_id(qc):
    # Combine question content with skill and difficulty fields to create a unique string.
    # Adjust this as needed.
    unique_str = qc['question']['content']
    unique_str += qc['question'].get('skillId', '')
    unique_str += qc['question'].get('difficultyLevelId', '')
    return hashlib.md5(unique_str.encode('utf-8')).hexdigest()

def process_concept(sectionId, node, lesson, concept, difficulty_level, difficulty_level_uri, skills, learning_objectives, number_questions_per_concept, question_types, customized_difficulty, customized_prompt_instructions, assessment_type="placement"):
    # Build atom content from concept atoms.
    atom_content = ""
    quiz_atoms = []
    for atom in concept.get('atoms', []):
        if not atom:
            continue
        stype = atom.get('semantic_type', '')
        if stype == "TextAtom":
            atom_content += "\nTEXT:\n" + atom.get('text', '')
        elif stype == "VideoAtom":
            atom_content += "\nVIDEO VTT:\n" + atom.get("video", {}).get("vtt_url", '')
        elif stype == "RadioQuizAtom":
            q = atom['question']
            quiz_atoms.append({
                'type': stype,
                'prompt': q.get('prompt'),
                'answers': q.get('answers', [])
            })
        elif stype == "CheckboxQuizAtom":
            q = atom['question']
            quiz_atoms.append({
                'type': stype,
                'prompt': q.get('prompt'),
                'correct_feedback': q.get('correct_feedback'),
                'answers': q.get('answers', [])
            })
        elif stype == "MatchingQuizAtom":
            q = atom['question']
            quiz_atoms.append({
                'type': stype,
                'complex_prompt': q.get('complex_prompt', {}).get('text'),
                'answers_label': q.get('answers_label'),
                'concepts_label': q.get('concepts_label'),
                'concepts': q.get('concepts', []),
                'answers': q.get('answers', [])
            })

    # Build the content dictionary with metadata.
    content = {
        'partTitle': node['title'],
        'partKey': node['key'],
        'partLocale': node['locale'],
        'partVersion': node['version'],
        'lesson': {
            'title': lesson['title'],
            'key': lesson['key'],
            'locale': lesson['locale'],
            'version': lesson['version'],
            'concept': {
                'title': concept['title'],
                'key': concept['key'],
                'locale': concept['locale'],
                'version': concept['version'],
                'atom_content': atom_content,
                'quiz_atoms': quiz_atoms
            }
        }
    }
    
    print(f"\n=== PROCESS_CONCEPT: {concept['title']} ===")
    print(f"Assessment type: {assessment_type}")
    print(f"Skills being passed to AI: {skills}")
    print(f"Concept: {concept['title']} (Key: {concept['key']})")
    print(f"Lesson: {lesson['title']} (Key: {lesson['key']})")
    print(f"Node: {node['title']} (Key: {node['key']})")
    
    # Choose the appropriate prompt based on assessment type
    if assessment_type.lower() == "readiness":
        prompt_messages = prompts.get_readiness_assessment_questions_prompt(
            number_questions_per_concept,
            difficulty_level,
            skills,
            question_types,
            learning_objectives,
            content,
            customized_difficulty,
            customized_prompt_instructions
        )
    else:
        # Default to placement assessment
        prompt_messages = prompts.get_assessment_questions_prompt(
            number_questions_per_concept,
            difficulty_level,
            skills,
            question_types,
            learning_objectives,
            content,
            customized_difficulty,
            customized_prompt_instructions
        )
    
    # Use context management for OpenAI API call
    chat_completion_response = settings.call_openai_with_fallback(
        model=settings.CHAT_COMPLETIONS_MODEL,
        response_format=settings.CHAT_COMPLETIONS_RESPONSE_FORMAT,
        #temperature=settings.CHAT_COMPLETIONS_TEMPERATURE, # not supported in gpt 5.
        messages=prompt_messages
    )
    
    # Remove any JSON code fences and parse the result.
    chat_completion = json.loads(
        chat_completion_response.choices[0].message.content.replace('```json','').replace('```','')
    )


    # Check if this concept has coding content
    has_coding_content_in_concept = False
    
    # Check atom content for coding
    if detect_coding_content(atom_content):
        has_coding_content_in_concept = True
    
    # Check concept title for coding
    if detect_coding_content(concept['title']):
        has_coding_content_in_concept = True
    
    # Check lesson title for coding
    if detect_coding_content(lesson['title']):
        has_coding_content_in_concept = True
    
    # Check node title for coding
    if detect_coding_content(node['title']):
        has_coding_content_in_concept = True
    
    # Process each question choice in the response.
    valid_questions = []
    for i, qc in enumerate(chat_completion['questions_choices']):
        # Validate response.
        if 'question' not in qc or 'choices' not in qc:
            print(f"  Question {i+1}: INVALID - missing 'question' or 'choices' key")
            continue
            
        question_skill = qc['question'].get('skillId', '')
        
        print(f"  Question {i+1}: skillId='{question_skill}' (Expected one of: {skills})")
        
        if question_skill not in skills:
            print(f"  Question {i+1}: REJECTED - skillId '{question_skill}' not in expected skills {skills}")
            continue
        else:
            print(f"  Question {i+1}: ACCEPTED - skillId '{question_skill}' matches expected skills")
            
        valid_questions.append(qc)
    
    print(f"Valid questions after filtering: {len(valid_questions)} out of {len(chat_completion.get('questions_choices', []))}")
    
    # Process valid questions
    for qc in valid_questions:
        # Attach metadata.
        qc['question']['sectionId'] = sectionId
        qc['question']['difficultyLevelUri'] = difficulty_level_uri  # Set the difficulty URI
        qc['question']['source'] = {
            'partTitle': node['title'],
            'partKey': node['key'],
            'partLocale': node['locale'],
            'partVersion': node['version'],
            'lessonTitle': lesson['title'],
            'lessonKey': lesson['key'],
            'lessonLocale': lesson['locale'],
            'lessonVersion': lesson['version'],
            'conceptTitle': concept['title'],
            'conceptKey': concept['key'],
            'conceptLocale': concept['locale'],
            'conceptVersion': concept['version'],
            'uri': f"https://learn.udacity.com/{node['key']}?progressKey={concept.get('progress_key', 'N/A')}"
        }
        qc['question']['sourceCategory'] = 'UDACITY'
        
        # Add coding content metadata to each question
        qc['question']['hasCodingContent'] = has_coding_content_in_concept
        
    return valid_questions


def process_concepts(section_content_definitions, number_questions_per_concept, question_types, customized_difficulty, customized_prompt_instructions, assessment_type="placement"):
    
    for section in section_content_definitions:
        section['questions_choices'] = []
        learning_objectives = section.get('learning_objectives')

        if assessment_type.lower() == "readiness":
            # For readiness assessment, collect all prerequisite skills from all program keys
            original_prerequisite_skills = set()
            for key in section['content_keys']:
                skills_for_key = section['skills'].get(key, [])
                for skill in skills_for_key:
                    skill_name = skill.get('name', '')
                    original_prerequisite_skills.add(skill_name)
            
            print(f"\n=== READINESS SKILLS VALIDATION ===")
            print(f"Original prerequisite skills collected: {sorted(list(original_prerequisite_skills))}")
            
            # Process each lesson node from Skills API
            for lesson_key, node in section['nodes'].items():
                if not node:
                    continue
                
                # For readiness, we use the first available difficulty and skills from the original program keys
                # Get difficulty from first available key
                difficulty_level = None
                difficulty_level_uri = None
                for key in section['content_keys']:
                    if section['difficulty_level'].get(key):
                        difficulty_level = section['difficulty_level'][key]['name']
                        difficulty_level_uri = section['difficulty_level'][key].get('uri', '')
                        break
                
                if not difficulty_level:
                    continue
                
                # Use original prerequisite skills for question generation
                skills = list(original_prerequisite_skills)
                
                print(f"\nProcessing lesson node {lesson_key}")
                print(f"Difficulty level: {difficulty_level} (URI: {difficulty_level_uri})")
                print(f"Skills: {skills}")
                
                futures = []
                concept_count = 0
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Handle different node types
                    if node.get('semantic_type') == 'Lesson':
                        # Skills API returns Lesson nodes that have concepts directly
                        for concept in node.get('concepts', []):
                            concept_count += 1
                            futures.append(
                                executor.submit(
                                    process_concept,
                                    section['content_keys'][0],  # Use first original program key as sectionId
                                    node, node, concept,  # For lesson nodes, pass node as both node and lesson
                                    difficulty_level, difficulty_level_uri, skills, learning_objectives,
                                    number_questions_per_concept, question_types, customized_difficulty, customized_prompt_instructions,
                                    assessment_type
                                )
                            )
                    else:
                        # Handle Part/Nanodegree nodes that have modules > lessons > concepts hierarchy
                        for module in node.get('modules', []):
                            for lesson in module.get('lessons', []):
                                for concept in lesson.get('concepts', []):
                                    concept_count += 1
                                    futures.append(
                                        executor.submit(
                                            process_concept,
                                            section['content_keys'][0],  # Use first original program key as sectionId
                                            node, lesson, concept,
                                            difficulty_level, difficulty_level_uri, skills, learning_objectives,
                                            number_questions_per_concept, question_types, customized_difficulty, customized_prompt_instructions,
                                            assessment_type
                                        )
                                    )
                    
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            qcs = future.result()
                            
                            # Filter questions to only include those tagged with original prerequisite skills
                            filtered_qcs = []
                            for qc in qcs:
                                question_skill = qc.get('question', {}).get('skillId', '')
                                
                                print(f"Question generated with skillId: '{question_skill}' - Valid: {question_skill in original_prerequisite_skills}")
                                
                                if question_skill in original_prerequisite_skills:
                                    filtered_qcs.append(qc)
                                else:
                                    print(f"WARNING: Question skillId '{question_skill}' not in original prerequisite skills {sorted(list(original_prerequisite_skills))}")
                            
                            section['questions_choices'].extend(filtered_qcs)
                        except Exception:
                            continue
                            
        else:
            # Original placement logic
            for key in section['content_keys']:
                difficulty_level = section['difficulty_level'][key]['name']
                difficulty_level_uri = section['difficulty_level'][key].get('uri', '')
                skills = [s['name'] for s in section['skills'][key]]
                node = section['nodes'][key]
                
                print(f"\n=== PLACEMENT SKILLS VALIDATION ===")
                print(f"Program key: {key}")
                print(f"Difficulty level: {difficulty_level} (URI: {difficulty_level_uri})")
                print(f"Skills for question generation: {skills}")

                futures = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    for module in node.get('modules', []):
                        for lesson in module.get('lessons', []):
                            for concept in lesson.get('concepts', []):
                                futures.append(
                                    executor.submit(
                                        process_concept,
                                        key, node, lesson, concept,
                                        difficulty_level, difficulty_level_uri, skills, learning_objectives,
                                        number_questions_per_concept, question_types, customized_difficulty, customized_prompt_instructions,
                                        assessment_type
                                    )
                                )
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            qcs = future.result()
                            
                            # Log generated questions for placement
                            for qc in qcs:
                                question_skill = qc.get('question', {}).get('skillId', '')
                                print(f"Placement question generated with skillId: '{question_skill}' - Valid: {question_skill in skills}")
                                if question_skill not in skills:
                                    print(f"WARNING: Placement question skillId '{question_skill}' not in expected skills {skills}")
                            
                            section['questions_choices'].extend(qcs)
                        except Exception:
                            continue

        # Deduplicate questions using the hash-based approach.
        unique_qcs = {}
        for qc in section['questions_choices']:
            qc_id = get_qc_id(qc)
            if qc_id not in unique_qcs:
                unique_qcs[qc_id] = qc

        section['questions_choices'] = list(unique_qcs.values())

    return section_content_definitions

def redistribute_order_indices(section_content_definitions):
    for section in section_content_definitions:
        for qc in section.get('questions_choices', []):
            choices = qc.get('choices', [])
            # make a slot for each choice (0..n-1), shuffle, then pop one per choice
            slots = list(range(len(choices)))
            random.shuffle(slots)

            for choice in choices:
                choice['orderIndex'] = slots.pop()
    return section_content_definitions

def evaluate_question(learning_objectives, qc, question_types):
    """
    Evaluate a single question choice (qc) against the provided learning objectives.
    Returns a tuple of (qc, evaluation_result).
    """
    try:
        start_time = time.perf_counter()
        
        # Use the new prompt template from prompts.py
        messages = prompts.get_question_evaluation_prompt(learning_objectives, qc, question_types)
        
        response = settings.call_openai_with_fallback(
            model=settings.CHAT_COMPLETIONS_MODEL,
            response_format=settings.CHAT_COMPLETIONS_RESPONSE_FORMAT,
            #temperature=settings.CHAT_COMPLETIONS_TEMPERATURE, # not supported in gpt 5.
            messages=messages
        )
        evaluation_result = json.loads(response.choices[0].message.content)
        elapsed = (time.perf_counter() - start_time) / 60.0
        print(f"Evaluation for one qc took {round(elapsed, 1)} mins")
        return qc, evaluation_result
    except Exception as e:
        print(f"Error evaluating qc: {e}")
        return qc, None

def evaluate_questions(section_content_definitions, question_types):

    for section in section_content_definitions:
        # Use the aggregated learning objectives at the section level.
        learning_objectives = section.get('learning_objectives')
        questions_choices = section.get('questions_choices', [])
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(evaluate_question, learning_objectives, qc, question_types)
                for qc in questions_choices
            ]
            for future in concurrent.futures.as_completed(futures, timeout=100000):  # Optional timeout per future.
                try:
                    qc, eval_result = future.result(timeout=100000)  # Timeout for each future's result.
                    if eval_result is not None:
                        qc['eval'] = eval_result  # Attach the evaluation result to the question object.
                except Exception as e:
                    print(f"Error in future result: {e}")
    
    return section_content_definitions

def json_to_dataframe(section_content_definitions):
    """
    Convert the nested section content definitions to a flat DataFrame.
    """
    
    # Collect all skills from the content definitions for URI mapping
    skill_to_id = {}
    for section in section_content_definitions:
        for key in section['content_keys']:
            skills_for_key = section['skills'].get(key, [])
            for skill in skills_for_key:
                skill_name = skill.get('name', '')
                skill_uri = skill.get('uri', '')
                if skill_name and skill_uri:
                    skill_to_id[skill_name] = skill_uri
    
    print(f"\n=== DATAFRAME SKILL MAPPING ===")
    print(f"Skill name -> URI mappings collected from metadata:")
    for skill_name, skill_uri in skill_to_id.items():
        print(f"  '{skill_name}' -> '{skill_uri}'")
    
    rows = []
    for section in section_content_definitions:
        for qc in section['questions_choices']:
            question = qc['question']
            choices = qc['choices']
            
            # Get the skill name from the question
            question_skill_name = question.get('skillId', '')
            # Map to URI using our collected mapping
            question_skill_uri = skill_to_id.get(question_skill_name, '')
            
            print(f"\nProcessing question with skillId: '{question_skill_name}'")
            print(f"  Mapped to skillUri: '{question_skill_uri}'")
            if not question_skill_uri and question_skill_name:
                print(f"  WARNING: No URI mapping found for skill '{question_skill_name}'")
            
            for choice in choices:
                row = {
                    'sectionId': question.get('sectionId'),
                    'difficultyLevelId': question.get('difficultyLevelId'),
                    'difficultyLevelUri': question.get('difficultyLevelUri'),
                    'skillId': question_skill_name,  # This holds the skill name initially
                    'skillUri': question_skill_uri,  # This holds the taxonomy URI
                    'category': question.get('category'),
                    'question_status': question.get('status'),
                    'question_content': question.get('content'),
                    'sourceCategory': question.get('sourceCategory'),
                    'source': question.get('source'),
                    'choice_status': choice.get('status'),
                    'choice_content': choice.get('content'),
                    'choice_isCorrect': choice.get('isCorrect'),
                    'choice_orderIndex': choice.get('orderIndex'),
                    'questionEvaluation': qc.get('eval', {}).get('questionEvaluation'),
                    'relevanceAndClarity': qc.get('eval', {}).get('relevanceAndClarity'),
                    'questionTypeDiversity': qc.get('eval', {}).get('questionTypeDiversity'),
                    'choiceQuality': qc.get('eval', {}).get('choiceQuality'),
                    'generalAdherence': qc.get('eval', {}).get('generalAdherence')
                }
                rows.append(row)
    
    print(f"\n=== DATAFRAME CREATION SUMMARY ===")
    print(f"Total rows created: {len(rows)}")
    if rows:
        unique_skill_names = set(row['skillId'] for row in rows if row['skillId'])
        unique_skill_uris = set(row['skillUri'] for row in rows if row['skillUri'])
        print(f"Unique skill names in DataFrame: {sorted(list(unique_skill_names))}")
        print(f"Unique skill URIs in DataFrame: {sorted(list(unique_skill_uris))}")
        
        # Check for mismatches
        rows_without_uri = [row for row in rows if row['skillId'] and not row['skillUri']]
        if rows_without_uri:
            print(f"WARNING: {len(rows_without_uri)} rows have skillId but no skillUri:")
            skill_names_without_uri = set(row['skillId'] for row in rows_without_uri)
            for skill_name in sorted(skill_names_without_uri):
                print(f"  - '{skill_name}'")
    
    return pd.DataFrame(rows)

def dedupe_questions_keep_choices(
    df: pd.DataFrame,
    question_col: str = "question_content",
    similarity_threshold: float = 0.8
) -> tuple[pd.DataFrame, dict]:
    # --- Phase 1: Get unique questions (one row per question) for analysis ---
    # Group by question_content and take the first row of each group to get unique questions
    unique_questions_df = df.groupby(question_col).first().reset_index()
    
    # Track initial unique question count
    initial_unique_questions = len(unique_questions_df)
    
    # Check if we have any questions to process
    if initial_unique_questions == 0:
        print("WARNING: No questions found for deduplication")
        intermediate_counts = {
            'initial_unique_questions': 0,
            'after_tfidf_unique_questions': 0,
            'after_semantic_unique_questions': 0,
            'initial_choices': len(df),
            'after_tfidf_choices': len(df),
            'final_choices': len(df)
        }
        return df, intermediate_counts
    
    # Check for empty or very short question content
    valid_questions = unique_questions_df[unique_questions_df[question_col].str.len() > 10]
    if len(valid_questions) == 0:
        print("WARNING: No questions with sufficient content for TF-IDF analysis (all questions too short)")
        intermediate_counts = {
            'initial_unique_questions': initial_unique_questions,
            'after_tfidf_unique_questions': initial_unique_questions,
            'after_semantic_unique_questions': initial_unique_questions,
            'initial_choices': len(df),
            'after_tfidf_choices': len(df),
            'final_choices': len(df)
        }
        return df, intermediate_counts
    
    try:
        # --- Phase 2: TF–IDF + cosine on the unique questions only ---
        print(f"Running TF-IDF analysis on {len(unique_questions_df)} unique questions...")
        tfidf = TfidfVectorizer(
            stop_words='english',
            min_df=1,  # Include terms that appear at least once
            max_features=10000,  # Limit features to avoid memory issues
            ngram_range=(1, 2)  # Include unigrams and bigrams
        ).fit_transform(unique_questions_df[question_col].astype(str))
        sims = cosine_similarity(tfidf)
        
        # --- Phase 3: pick which question-rows to drop (in the unique list) ---
        n = sims.shape[0]
        to_drop = set()
        for i in range(n):
            if i in to_drop:
                continue
            for j in range(i+1, n):
                if sims[i, j] >= similarity_threshold:
                    to_drop.add(j)
        
        # these are the *positions* in unique_questions_df that we want to KEEP
        keep_positions = [i for i in range(n) if i not in to_drop]
        keep_questions = unique_questions_df.loc[keep_positions, question_col]
        
        # Track unique questions after TF-IDF
        after_tfidf_unique_questions = len(keep_questions)
        
        print(f"TF-IDF deduplication: {initial_unique_questions} -> {after_tfidf_unique_questions} questions")
        
    except ValueError as e:
        if "empty vocabulary" in str(e).lower():
            print("WARNING: TF-IDF failed due to empty vocabulary (questions may contain only stop words)")
            print("Skipping TF-IDF deduplication and proceeding with semantic analysis...")
            # Skip TF-IDF and use all questions
            keep_questions = unique_questions_df[question_col]
            after_tfidf_unique_questions = len(keep_questions)
        else:
            # Re-raise if it's a different ValueError
            raise e
    except Exception as e:
        print(f"WARNING: TF-IDF analysis failed with error: {e}")
        print("Skipping TF-IDF deduplication and proceeding with semantic analysis...")
        # Skip TF-IDF and use all questions
        keep_questions = unique_questions_df[question_col]
        after_tfidf_unique_questions = len(keep_questions)
    
    # --- Phase 4: filter the *original* df to only those questions (keeping all choices) ---
    df_after_tfidf = (
        df[df[question_col].isin(keep_questions)]
          .reset_index(drop=True)
    )
    
    # --- Phase 5: OpenAI semantic similarity analysis by skillId ---
    # Group questions by skillId (using unique questions for analysis)
    unique_questions_after_tfidf = df_after_tfidf.groupby(question_col).first().reset_index()
    skill_groups = unique_questions_after_tfidf.groupby('skillId')
    
    questions_to_keep = set()
    
    print(f"Running semantic similarity analysis on {len(skill_groups)} skill groups...")
    
    for skill_id, skill_df in skill_groups:
        if len(skill_df) <= 1:
            # If only one question for this skill, keep it
            questions_to_keep.update(skill_df[question_col].tolist())
            continue
            
        # Get the questions for this skill
        questions_list = skill_df[question_col].tolist()
        
        try:
            # Call OpenAI for semantic similarity analysis
            messages = prompts.get_semantic_similarity_prompt(skill_id, questions_list)
            
            response = settings.call_openai_with_fallback(
                model=settings.CHAT_COMPLETIONS_MODEL,
                response_format=settings.CHAT_COMPLETIONS_RESPONSE_FORMAT,
                #temperature=settings.CHAT_COMPLETIONS_TEMPERATURE, #not supported in gpt 5.
                messages=messages
            )
            
            # Parse the response
            analysis_result = json.loads(response.choices[0].message.content)
            
            # Get the indices of questions to keep for this skill
            keep_indices = analysis_result.get('questions_to_keep', [])
            
            # Convert to actual question content
            for keep_idx in keep_indices:
                if 0 <= keep_idx < len(questions_list):
                    questions_to_keep.add(questions_list[keep_idx])
                    
        except Exception as e:
            print(f"Error in semantic similarity analysis for skill {skill_id}: {e}")
            # If there's an error, keep all questions for this skill
            questions_to_keep.update(questions_list)
    
    # Track unique questions after semantic analysis
    after_semantic_unique_questions = len(questions_to_keep)
    
    print(f"Semantic deduplication: {after_tfidf_unique_questions} -> {after_semantic_unique_questions} questions")
    
    # --- Phase 6: filter the dataframe to only keep the questions identified by OpenAI (keeping all choices) ---
    final_df = df_after_tfidf[df_after_tfidf[question_col].isin(questions_to_keep)].reset_index(drop=True)
    
    # Return both the final dataframe and intermediate counts (tracking unique questions, not choices)
    intermediate_counts = {
        'initial_unique_questions': initial_unique_questions,
        'after_tfidf_unique_questions': after_tfidf_unique_questions,
        'after_semantic_unique_questions': after_semantic_unique_questions,
        'initial_choices': len(df),
        'after_tfidf_choices': len(df_after_tfidf),
        'final_choices': len(final_df)
    }
    
    return final_df, intermediate_counts

def filter_content_specific_questions(
    df: pd.DataFrame,
    question_col: str = "question_content",
    batch_size: int = 50
) -> tuple[pd.DataFrame, dict]:
    """
    Filter out questions that are too specific to the course content.
    
    Args:
        df: DataFrame containing questions and choices
        question_col: Column name containing question content
        batch_size: Number of questions to process in each batch
    
    Returns:
        tuple: (filtered_dataframe, filtering_stats)
    """
    # Get unique questions for analysis
    unique_questions_df = df.groupby(question_col).first().reset_index()
    initial_unique_questions = len(unique_questions_df)
    
    questions_to_keep = set()
    
    # Process questions in batches to avoid token limits
    for i in range(0, len(unique_questions_df), batch_size):
        batch_df = unique_questions_df.iloc[i:i+batch_size]
        questions_list = batch_df[question_col].tolist()
        
        try:
            # Call OpenAI for content specificity analysis
            messages = prompts.get_content_specificity_prompt(questions_list)
            
            response = settings.call_openai_with_fallback(
                model=settings.CHAT_COMPLETIONS_MODEL,
                response_format=settings.CHAT_COMPLETIONS_RESPONSE_FORMAT,
                #temperature=settings.CHAT_COMPLETIONS_TEMPERATURE, #not supported in gpt 5
                messages=messages
            )
            
            # Parse the response
            analysis_result = json.loads(response.choices[0].message.content)
            
            # Get the indices of questions to keep for this batch
            keep_indices = analysis_result.get('questions_to_keep', [])
            
            # Convert to actual question content
            for keep_idx in keep_indices:
                if 0 <= keep_idx < len(questions_list):
                    questions_to_keep.add(questions_list[keep_idx])
                    
        except Exception as e:
            print(f"Error in content specificity analysis for batch {i//batch_size + 1}: {e}")
            # If there's an error, keep all questions in this batch
            questions_to_keep.update(questions_list)
    
    # Filter the dataframe to only keep the questions identified as not too specific
    filtered_df = df[df[question_col].isin(questions_to_keep)].reset_index(drop=True)
    
    # Calculate statistics
    after_filtering_unique_questions = len(questions_to_keep)
    filtered_out_questions = initial_unique_questions - after_filtering_unique_questions
    
    filtering_stats = {
        'initial_unique_questions': initial_unique_questions,
        'after_filtering_unique_questions': after_filtering_unique_questions,
        'filtered_out_questions': filtered_out_questions,
        'filtering_percentage': (filtered_out_questions / initial_unique_questions * 100) if initial_unique_questions > 0 else 0,
        'initial_choices': len(df),
        'final_choices': len(filtered_df)
    }
    
    return filtered_df, filtering_stats

def convert_questions_to_case_studies(
    df: pd.DataFrame,
    question_col: str = "question_content",
    conversion_percentage: float = 0.15,
    batch_size: int = 50,
    customized_prompt_instructions: str = ""
) -> tuple[pd.DataFrame, dict]:
    """
    Convert a percentage of questions to case study format for increased complexity.
    
    Args:
        df: DataFrame containing questions and choices
        question_col: Column name containing question content
        conversion_percentage: Percentage of questions to convert (default 15%)
        batch_size: Number of questions to process in each batch
        customized_prompt_instructions: Custom instructions to incorporate into case studies
    
    Returns:
        tuple: (converted_dataframe, conversion_stats)
    """
    # Get unique questions for analysis
    unique_questions_df = df.groupby(question_col).first().reset_index()
    initial_unique_questions = len(unique_questions_df)
    
    # Calculate how many questions to convert
    questions_to_convert = max(1, int(initial_unique_questions * conversion_percentage))
    
    # Randomly select questions to convert
    selected_indices = random.sample(range(len(unique_questions_df)), min(questions_to_convert, len(unique_questions_df)))
    selected_questions = unique_questions_df.iloc[selected_indices]
    
    converted_questions = []
    questions_that_were_converted = set()  # Track which questions were actually converted
    
    # Process selected questions in batches
    for i in range(0, len(selected_questions), batch_size):
        batch_df = selected_questions.iloc[i:i+batch_size]
        
        # Group by skillId for better context
        skill_groups = batch_df.groupby('skillId')
        
        for skill_id, skill_df in skill_groups:
            questions_list = skill_df[question_col].tolist()
            
            # Create question objects in the format expected by the prompt
            question_objects = []
            for i, question_content in enumerate(questions_list):
                question_objects.append({
                    'question': {
                        'content': question_content
                    }
                })
            
            try:
                # Call OpenAI for case study conversion
                messages = prompts.get_case_study_conversion_prompt(skill_id, question_objects, customized_prompt_instructions)
                
                response = settings.call_openai_with_fallback(
                    model=settings.CHAT_COMPLETIONS_MODEL,
                    response_format=settings.CHAT_COMPLETIONS_RESPONSE_FORMAT,
                    #temperature=settings.CHAT_COMPLETIONS_TEMPERATURE, # not supported in gpt 5.
                    messages=messages
                )
                
                # Parse the response
                conversion_result = json.loads(response.choices[0].message.content)
                
                # Process converted questions
                for converted_item in conversion_result.get('converted_questions', []):
                    original_index = converted_item.get('original_question_index')
                    case_study_question = converted_item.get('case_study_question', {})
                    choices = converted_item.get('choices', [])
                    
                    if original_index is not None and 0 <= original_index < len(questions_list):
                        # Find the original question in the dataframe
                        original_question = questions_list[original_index]
                        original_rows = df[df[question_col] == original_question]
                        
                        if not original_rows.empty:
                            # Track that this question was converted
                            questions_that_were_converted.add(original_question)
                            
                            # Create new rows for the case study question
                            for _, original_row in original_rows.iterrows():
                                try:
                                    new_row = original_row.copy()
                                    new_row[question_col] = case_study_question.get('content', original_question)
                                    new_row['difficultyLevelId'] = case_study_question.get('difficultyLevelId', original_row['difficultyLevelId'])
                                    new_row['category'] = case_study_question.get('category', original_row['category'])
                                    new_row['question_status'] = case_study_question.get('status', original_row['question_status'])
                                    
                                    # Update choice content if available
                                    choice_index = original_row.get('choice_orderIndex', 0)
                                    for choice in choices:
                                        if choice.get('orderIndex') == choice_index:
                                            new_row['choice_content'] = choice.get('content', original_row['choice_content'])
                                            new_row['choice_isCorrect'] = choice.get('isCorrect', original_row['choice_isCorrect'])
                                            new_row['choice_orderIndex'] = choice.get('orderIndex', original_row['choice_orderIndex'])
                                            break
                                    
                                    converted_questions.append(new_row)
                                except Exception:
                                    continue
                    
            except Exception:
                # If there's an error, keep the original questions
                continue
    
    # Create new dataframe with converted questions
    if converted_questions:
        converted_df = pd.DataFrame(converted_questions)
        
        # Remove original questions that were converted and add the converted versions
        questions_to_remove = list(questions_that_were_converted)  # Use the set of actually converted questions
        df_without_converted = df[~df[question_col].isin(questions_to_remove)]
        
        # Combine original questions with converted questions
        final_df = pd.concat([df_without_converted, converted_df], ignore_index=True)
    else:
        final_df = df.copy()
    
    # Calculate statistics
    final_unique_questions = final_df[question_col].nunique()
    successfully_converted = len(questions_that_were_converted)  # Use the set count instead of estimation
    
    conversion_stats = {
        'initial_unique_questions': initial_unique_questions,
        'questions_selected_for_conversion': len(selected_questions),
        'successfully_converted': successfully_converted,
        'conversion_percentage': (successfully_converted / initial_unique_questions * 100) if initial_unique_questions > 0 else 0,
        'final_unique_questions': final_unique_questions,
        'initial_choices': len(df),
        'final_choices': len(final_df)
    }
    
    return final_df, conversion_stats

def filter_questions_by_evaluation(
    df: pd.DataFrame,
    evaluation_col: str = "questionEvaluation"
) -> tuple[pd.DataFrame, dict]:
    """
    Filter questions based on evaluation results (PASS/FAIL).
    
    Args:
        df: DataFrame containing questions and choices
        evaluation_col: Column name containing evaluation results
    
    Returns:
        tuple: (filtered_dataframe, filtering_stats)
    """
    # Get unique questions for analysis
    unique_questions_df = df.groupby('question_content').first().reset_index()
    initial_unique_questions = len(unique_questions_df)
    
    print(f"\n=== EVALUATION FILTERING DEBUG ===")
    print(f"Initial unique questions: {initial_unique_questions}")
    
    # Check evaluation values
    eval_values = unique_questions_df[evaluation_col].value_counts()
    print(f"Evaluation value counts:")
    for value, count in eval_values.items():
        print(f"  {value}: {count}")
    
    # Filter questions that PASS evaluation
    passed_questions = unique_questions_df[unique_questions_df[evaluation_col] == 'PASS']['question_content'].tolist()
    
    print(f"Questions that PASSED evaluation: {len(passed_questions)}")
    
    # If no questions pass, let's be less strict and keep questions with non-null evaluations
    if len(passed_questions) == 0:
        print("WARNING: No questions passed evaluation! Keeping questions with non-null evaluations...")
        non_null_questions = unique_questions_df[unique_questions_df[evaluation_col].notna()]['question_content'].tolist()
        print(f"Questions with non-null evaluations: {len(non_null_questions)}")
        
        if len(non_null_questions) > 0:
            passed_questions = non_null_questions
        else:
            print("WARNING: No questions have evaluation results! Keeping all questions...")
            passed_questions = unique_questions_df['question_content'].tolist()
    
    # Filter the dataframe to only keep questions that passed evaluation
    filtered_df = df[df['question_content'].isin(passed_questions)].reset_index(drop=True)
    
    # Calculate statistics
    after_evaluation_unique_questions = len(passed_questions)
    failed_questions = initial_unique_questions - after_evaluation_unique_questions
    
    print(f"Final questions after evaluation filtering: {after_evaluation_unique_questions}")
    print(f"=== END EVALUATION FILTERING DEBUG ===\n")
    
    evaluation_stats = {
        'initial_unique_questions': initial_unique_questions,
        'after_evaluation_unique_questions': after_evaluation_unique_questions,
        'failed_questions': failed_questions,
        'evaluation_failure_percentage': (failed_questions / initial_unique_questions * 100) if initial_unique_questions > 0 else 0,
        'initial_choices': len(df),
        'final_choices': len(filtered_df)
    }
    
    return filtered_df, evaluation_stats

def convert_questions_to_code_format_based_on_metadata(
    section_content_definitions,
    conversion_percentage: float = 0.30,
    customized_prompt_instructions: str = ""
) -> dict:
    """
    Convert questions to include code markdown based on hasCodingContent metadata.
    
    Parameters:
        section_content_definitions: The processed content definitions with questions
        conversion_percentage: Percentage of coding questions to convert (default 30%)
        customized_prompt_instructions: Additional instructions for the conversion
    
    Returns:
        dict: Conversion statistics
    """
    # Collect all questions with coding content metadata
    all_questions = []
    coding_questions = []
    
    for section in section_content_definitions:
        for qc in section.get('questions_choices', []):
            all_questions.append(qc)
            if qc['question'].get('hasCodingContent', False):
                coding_questions.append(qc)
    
    total_questions = len(all_questions)
    total_coding_questions = len(coding_questions)
    
    # If no questions are coding marked, skip this step
    if total_coding_questions == 0:
        return {
            "converted_questions": 0,
            "total_questions": total_questions,
            "total_coding_questions": 0,
            "conversion_percentage": 0,
            "reason": "No questions with coding content found"
        }
    
    # Calculate how many questions to convert
    target_conversion_count = int(total_questions * conversion_percentage)
    
    # If less than 30% of questions are coding marked, get all of them
    if total_coding_questions <= target_conversion_count:
        questions_to_convert = coding_questions
        actual_conversion_count = total_coding_questions
    else:
        # Sample 30% from coding questions
        questions_to_convert = random.sample(coding_questions, target_conversion_count)
        actual_conversion_count = target_conversion_count
    
    converted_count = 0
    failed_conversions = 0
    
    # Convert selected questions to include code markdown
    for qc in questions_to_convert:
        try:
            # Prepare the question data for conversion
            question_data = {
                "question": qc['question'],
                "choices": qc['choices']
            }
            
            # Convert to JSON string
            questions_json = json.dumps({"questions_choices": [question_data]}, ensure_ascii=False)
            
            # Get the conversion prompt
            prompt_messages = prompts.get_code_conversion_prompt(
                qc['question']['skillId'],
                questions_json,
                qc['question'].get('difficultyLevelId', ''),
                [qc['question']['skillId']],  # skills
                "",  # learning_objectives (not needed for conversion)
                "",  # question_types (not needed for conversion)
                "",  # content (not needed for conversion)
                "No Change",  # customized_difficulty
                customized_prompt_instructions
            )
            
            # Make API call using the existing client from settings
            response = settings.openai_client.chat.completions.create(
                model=settings.CHAT_COMPLETIONS_MODEL,
                messages=prompt_messages,
                #temperature=0.3, # not supported in gpt 5.
                max_tokens=4000
            )
            
            # Parse the response
            response_content = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            try:
                # Look for JSON in the response
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    converted_data = json.loads(json_match.group())
                    
                    if "questions_choices" in converted_data and len(converted_data["questions_choices"]) > 0:
                        converted_qc = converted_data["questions_choices"][0]
                        
                        # Update the question content and choices
                        qc['question']['content'] = converted_qc["question"]["content"]
                        qc['choices'] = converted_qc["choices"]
                        
                        converted_count += 1
                    else:
                        failed_conversions += 1
                else:
                    failed_conversions += 1
                    
            except json.JSONDecodeError:
                # If conversion fails, keep the original question
                failed_conversions += 1
                continue
                
        except Exception:
            # If conversion fails, keep the original question
            failed_conversions += 1
            continue
    
    return {
        "converted_questions": converted_count,
        "failed_conversions": failed_conversions,
        "total_questions": total_questions,
        "total_coding_questions": total_coding_questions,
        "conversion_percentage": (converted_count / total_questions * 100) if total_questions > 0 else 0,
        "reason": f"Converted {converted_count} out of {actual_conversion_count} selected questions"
    }

def tune_distractors(section_content_definitions, tuning_percentage=0.20):
    """
    Tune distractors for a percentage of questions to make them more challenging.
    
    Args:
        section_content_definitions: The processed content definitions with questions
        tuning_percentage: Percentage of questions to tune (default 20%)
    
    Returns:
        dict: Tuning statistics
    """
    # Collect all questions
    all_questions = []
    for section in section_content_definitions:
        for qc in section.get('questions_choices', []):
            all_questions.append(qc)
    
    total_questions = len(all_questions)
    
    if total_questions == 0:
        return {
            "tuned_questions": 0,
            "total_questions": 0,
            "tuning_percentage": 0,
            "reason": "No questions found to tune"
        }
    
    # Calculate how many questions to tune
    questions_to_tune_count = max(1, int(total_questions * tuning_percentage))
    
    # Randomly select questions to tune
    questions_to_tune = random.sample(all_questions, min(questions_to_tune_count, total_questions))
    
    tuned_count = 0
    failed_tuning = 0
    
    # Tune selected questions
    for qc in questions_to_tune:
        try:
            # Prepare the question data for tuning
            question_data = {
                "question": qc['question'],
                "choices": qc['choices']
            }
            
            # Get the tuning prompt
            prompt_messages = prompts.get_distractor_tuning_prompt(question_data)
            
            # Make API call
            response = settings.call_openai_with_fallback(
                model=settings.CHAT_COMPLETIONS_MODEL,
                response_format=settings.CHAT_COMPLETIONS_RESPONSE_FORMAT,
                #temperature=settings.CHAT_COMPLETIONS_TEMPERATURE, # not supported in gpt5
                messages=prompt_messages
            )
            
            # Parse the response
            response_content = response.choices[0].message.content.strip()
            
            # Clean and parse JSON response
            try:
                # Remove any JSON code fences
                cleaned_response = response_content.replace('```json', '').replace('```', '').strip()
                tuned_data = json.loads(cleaned_response)
                
                # Update the question with tuned distractors
                if "question" in tuned_data and "choices" in tuned_data:
                    qc['question'] = tuned_data["question"]
                    qc['choices'] = tuned_data["choices"]
                    tuned_count += 1
                else:
                    failed_tuning += 1
                    
            except json.JSONDecodeError:
                failed_tuning += 1
                continue
                
        except Exception:
            failed_tuning += 1
            continue
    
    return {
        "tuned_questions": tuned_count,
        "failed_tuning": failed_tuning,
        "total_questions": total_questions,
        "questions_selected_for_tuning": len(questions_to_tune),
        "tuning_percentage": (tuned_count / total_questions * 100) if total_questions > 0 else 0,
        "reason": f"Tuned {tuned_count} out of {len(questions_to_tune)} selected questions"
    }

def generate_assessments(
    PROGRAM_KEYS, 
    QUESTION_TYPES, 
    QUESTION_LIMIT, 
    CUSTOMIZED_DIFFICULTY, 
    CUSTOMIZED_PROMPT_INSTRUCTIONS, 
    TEMPERATURE, 
    ASSESSMENT_TYPE, 
    NUMBER_QUESTIONS_PER_CONCEPT, 
    progress_bar=None, 
    progress_text=None
):
    # Used to be defaulted to 1, now set to up to 5 to allow work around for Solutions Architects.
    NUMBER_QUESTIONS_PER_CONCEPT = NUMBER_QUESTIONS_PER_CONCEPT
    
    steps = [
        "Preparing program keys...",
        "Adding program data...",
        "Adding node data...",
        "Processing nodes...",
        "Extracting content...",
        "Generating learning objectives...",
        "Processing concepts...",
        "Redistributing order indices...",
        "Evaluating questions...",
        "Converting to DataFrame...",
        "Filtering failed evaluations...",
        "Deduplicating questions...",
        "Filtering content-specific questions...",
        "Converting 15 percent of questions to case studies...",
        "Converting 30 percent of coding questions to include code markdown...",
        "Tuning distractors for 20 percent of questions...",
        "Selecting best questions (if limit specified)...",
        "Finalizing results..."
    ]
    total_steps = len(steps)

    def update_progress(step_index):
        if progress_bar:
            progress_bar.progress((step_index + 1) / total_steps)
        if progress_text:
            progress_text.text(steps[step_index])

    # Track question counts at each step
    question_counts = []
    
    # Track debug outputs from each step
    debug_outputs = {}

    step = 0

    update_progress(step)
    section_content_definitions = prep_program_keys(PROGRAM_KEYS)
    debug_outputs['prep_program_keys'] = {
        'output': str(section_content_definitions)[:1000],  # Limit to 1000 chars
        'type': type(section_content_definitions).__name__
    }
    
    step += 1
    update_progress(step)
    section_content_definitions, missing_prerequisite_skills = add_program_data(section_content_definitions, assessment_type=ASSESSMENT_TYPE)
    debug_outputs['add_program_data'] = {
        'output': str(section_content_definitions)[:1000],
        'type': type(section_content_definitions).__name__,
        'missing_prerequisite_skills': missing_prerequisite_skills
    }

    step += 1
    update_progress(step)
    section_content_definitions = add_node_data(section_content_definitions, assessment_type=ASSESSMENT_TYPE)
    debug_outputs['add_node_data'] = {
        'output': str(section_content_definitions)[:1000],
        'type': type(section_content_definitions).__name__,
        'node_count': sum(len(s.get('nodes', {})) for s in section_content_definitions)
    }

    step += 1
    update_progress(step)
    all_nodes = process_nodes(section_content_definitions, assessment_type=ASSESSMENT_TYPE)
    debug_outputs['process_nodes'] = {
        'output': f"Processed {len(all_nodes)} nodes",
        'type': 'list',
        'node_count': len(all_nodes)
    }

    step += 1
    update_progress(step)
    section_content_definitions = extract_content(section_content_definitions)
    debug_outputs['extract_content'] = {
        'output': str(section_content_definitions)[:1000],
        'type': type(section_content_definitions).__name__
    }

    step += 1
    update_progress(step)
    try:
        section_content_definitions = learning_objective_generator(section_content_definitions)
        debug_outputs['learning_objective_generator'] = {
            'output': str([s.get('learning_objectives', {}) for s in section_content_definitions])[:1000],
            'type': 'list'
        }
    except Exception as e:
        print(f"Error in learning objectives generation: {e}")
        print(format_exception_details(e))
        print(f"Section content definitions: {section_content_definitions}")
        raise e

    step += 1
    update_progress(step)
    section_content_definitions = process_concepts(section_content_definitions, NUMBER_QUESTIONS_PER_CONCEPT, QUESTION_TYPES, CUSTOMIZED_DIFFICULTY, CUSTOMIZED_PROMPT_INSTRUCTIONS, ASSESSMENT_TYPE)
    debug_outputs['process_concepts'] = {
        'output': f"Generated questions for {len(section_content_definitions)} sections",
        'type': 'list',
        'total_questions': sum(len(s.get('questions_choices', [])) for s in section_content_definitions)
    }
    
    # Count questions after processing concepts
    total_questions = sum(len(section.get('questions_choices', [])) for section in section_content_definitions)
    # Estimate total choices (assuming average 4 choices per question)
    estimated_choices = total_questions * 4
    question_counts.append(("After Processing Concepts", total_questions, estimated_choices))
    
    # Add debugging for empty questions
    if total_questions == 0:
        print("\n" + "="*80)
        print("WARNING: No questions were generated during concept processing!")
        print("="*80)
        print("Debugging information:")
        for i, section in enumerate(section_content_definitions):
            print(f"\nSection {i+1}:")
            print(f"  Title: {section.get('title', 'N/A')}")
            print(f"  Content keys: {section.get('content_keys', [])}")
            print(f"  Nodes: {len(section.get('nodes', {}))}")
            print(f"  Skills: {section.get('skills', {})}")
            print(f"  Questions generated: {len(section.get('questions_choices', []))}")
            
            # Check if nodes were processed
            if not section.get('nodes'):
                print("  ERROR: No nodes found - check if program keys are valid")
            elif not section.get('skills'):
                print("  ERROR: No skills found - check program metadata")
            else:
                print("  Nodes and skills present but no questions generated")
        print("="*80)
        
        # Continue with empty dataset to avoid crashes
        questions_choices_df = pd.DataFrame(columns=[
            'sectionId', 'difficultyLevelId', 'difficultyLevelUri', 'skillId', 'skillUri',
            'category', 'question_status', 'question_content', 'sourceCategory', 'source',
            'choice_status', 'choice_content', 'choice_isCorrect', 'choice_orderIndex',
            'questionEvaluation', 'relevanceAndClarity', 'questionTypeDiversity', 
            'choiceQuality', 'generalAdherence'
        ])
        
        # Return early with empty results
        progress_data = {
            'question_counts': question_counts,
            'evaluation_stats': {'initial_unique_questions': 0, 'after_evaluation_unique_questions': 0, 'failed_questions': 0},
            'intermediate_counts': {'initial_unique_questions': 0, 'after_tfidf_unique_questions': 0, 'after_semantic_unique_questions': 0},
            'filtering_stats': {'initial_unique_questions': 0, 'after_filtering_unique_questions': 0, 'filtered_out_questions': 0},
            'selection_stats': {'selection_needed': False, 'reason': 'No questions generated'},
            'conversion_stats': {'initial_unique_questions': 0, 'successfully_converted': 0},
            'code_conversion_stats': {'converted_questions': 0, 'total_questions': 0},
            'tuning_stats': {'tuned_questions': 0, 'total_questions': 0},
            'missing_prerequisite_skills': missing_prerequisite_skills,
            'debug_outputs': debug_outputs
        }
        return questions_choices_df, progress_data

    step += 1
    update_progress(step)
    section_content_definitions = redistribute_order_indices(section_content_definitions)
    debug_outputs['redistribute_order_indices'] = {
        'output': 'Order indices redistributed',
        'type': 'list'
    }

    step += 1
    update_progress(step)
    section_content_definitions = evaluate_questions(section_content_definitions, QUESTION_TYPES)
    debug_outputs['evaluate_questions'] = {
        'output': f"Evaluated questions for {len(section_content_definitions)} sections",
        'type': 'list'
    }

    step += 1
    update_progress(step)
    questions_choices_df = json_to_dataframe(section_content_definitions)
    debug_outputs['json_to_dataframe'] = {
        'output': f"DataFrame with {len(questions_choices_df)} rows, {questions_choices_df['question_content'].nunique()} unique questions",
        'type': 'DataFrame',
        'shape': questions_choices_df.shape
    }
    
    # Count questions after converting to dataframe
    total_choices = len(questions_choices_df)
    unique_questions = questions_choices_df['question_content'].nunique()
    question_counts.append(("After Converting to DataFrame", unique_questions, total_choices))
    
    print(f"\n=== DATAFRAME CONVERSION DEBUG ===")
    print(f"Questions after dataframe conversion: {unique_questions} unique questions, {total_choices} total choices")
    if unique_questions == 0:
        print("ERROR: No questions in dataframe after conversion!")
    print(f"=== END DATAFRAME CONVERSION DEBUG ===\n")

    step += 1
    update_progress(step)
    questions_choices_df, evaluation_stats = filter_questions_by_evaluation(questions_choices_df, evaluation_col="questionEvaluation")
    debug_outputs['filter_questions_by_evaluation'] = {
        'output': f"Filtered to {questions_choices_df['question_content'].nunique()} unique questions",
        'type': 'DataFrame',
        'evaluation_stats': evaluation_stats
    }
    
    # Count questions after evaluation filtering
    total_choices_after_eval = len(questions_choices_df)
    unique_questions_after_eval = questions_choices_df['question_content'].nunique()
    question_counts.append(("After Evaluation Filtering", unique_questions_after_eval, total_choices_after_eval))

    step += 1
    update_progress(step)
    questions_choices_df, intermediate_counts = dedupe_questions_keep_choices(questions_choices_df, question_col="question_content", similarity_threshold=0.75)
    debug_outputs['dedupe_questions_keep_choices'] = {
        'output': f"Deduplicated to {questions_choices_df['question_content'].nunique()} unique questions",
        'type': 'DataFrame',
        'intermediate_counts': intermediate_counts
    }
    
    # Count questions after deduplication
    total_choices_final = len(questions_choices_df)
    unique_questions_final = questions_choices_df['question_content'].nunique()
    question_counts.append(("After Deduplication", unique_questions_final, total_choices_final))

    step += 1
    update_progress(step)
    questions_choices_df, filtering_stats = filter_content_specific_questions(questions_choices_df, question_col="question_content", batch_size=50)
    debug_outputs['filter_content_specific_questions'] = {
        'output': f"Filtered to {questions_choices_df['question_content'].nunique()} unique questions",
        'type': 'DataFrame',
        'filtering_stats': filtering_stats
    }
    
    # Count questions after filtering
    total_choices_final_filtered = len(questions_choices_df)
    unique_questions_final_filtered = questions_choices_df['question_content'].nunique()
    question_counts.append(("After Filtering", unique_questions_final_filtered, total_choices_final_filtered))

    step += 1
    update_progress(step)
    questions_choices_df, conversion_stats = convert_questions_to_case_studies(questions_choices_df, question_col="question_content", conversion_percentage=0.15, batch_size=50, customized_prompt_instructions=CUSTOMIZED_PROMPT_INSTRUCTIONS)
    debug_outputs['convert_questions_to_case_studies'] = {
        'output': f"Converted {conversion_stats.get('successfully_converted', 0)} questions to case studies",
        'type': 'DataFrame',
        'conversion_stats': conversion_stats
    }
    
    # Count questions after case study conversion
    total_choices_final_converted = len(questions_choices_df)
    unique_questions_final_converted = questions_choices_df['question_content'].nunique()
    question_counts.append(("After Case Study Conversion", unique_questions_final_converted, total_choices_final_converted))

    step += 1
    update_progress(step)
    # Convert code format questions directly on the filtered dataframe
    questions_choices_df, code_conversion_stats = convert_questions_to_code_format_dataframe(
        questions_choices_df, 
        conversion_percentage=0.30, 
        customized_prompt_instructions=CUSTOMIZED_PROMPT_INSTRUCTIONS
    )
    debug_outputs['convert_questions_to_code_format'] = {
        'output': f"Converted {code_conversion_stats.get('converted_questions', 0)} questions to code format",
        'type': 'DataFrame',
        'code_conversion_stats': code_conversion_stats
    }
    
    # Count questions after code conversion
    unique_questions_after_code = questions_choices_df['question_content'].nunique()
    total_choices_after_code = len(questions_choices_df)
    question_counts.append(("After Code Format Conversion", unique_questions_after_code, total_choices_after_code))

    step += 1
    update_progress(step)
    # Tune distractors directly on the filtered dataframe
    questions_choices_df, tuning_stats = tune_distractors_dataframe(
        questions_choices_df, 
        tuning_percentage=0.20
    )
    debug_outputs['tune_distractors'] = {
        'output': f"Tuned {tuning_stats.get('tuned_questions', 0)} questions",
        'type': 'DataFrame',
        'tuning_stats': tuning_stats
    }
    
    # Count questions after distractor tuning
    unique_questions_after_tuning = questions_choices_df['question_content'].nunique()
    total_choices_after_tuning = len(questions_choices_df)
    question_counts.append(("After Distractor Tuning", unique_questions_after_tuning, total_choices_after_tuning))

    step += 1
    update_progress(step)
    # Apply intelligent question selection if limit is specified and not "No Limit"
    if QUESTION_LIMIT != "No Limit":
        questions_choices_df, selection_stats = intelligent_question_selection(questions_choices_df, QUESTION_LIMIT, question_col="question_content")
        debug_outputs['intelligent_question_selection'] = {
            'output': f"Selected {questions_choices_df['question_content'].nunique()} questions",
            'type': 'DataFrame',
            'selection_stats': selection_stats
        }
        
        # Count questions after selection
        total_choices_after_selection = len(questions_choices_df)
        unique_questions_after_selection = questions_choices_df['question_content'].nunique()
        question_counts.append(("After Question Selection", unique_questions_after_selection, total_choices_after_selection))
    else:
        # No selection needed
        selection_stats = {
            'selection_needed': False,
            'reason': 'No question limit specified'
        }
        debug_outputs['intelligent_question_selection'] = {
            'output': 'Skipped - no question limit specified',
            'type': 'skipped',
            'selection_stats': selection_stats
        }

    step += 1
    update_progress(step)

    # Return both the dataframe and the progress data
    progress_data = {
        'question_counts': question_counts,
        'evaluation_stats': evaluation_stats,
        'intermediate_counts': intermediate_counts,
        'filtering_stats': filtering_stats,
        'selection_stats': selection_stats,
        'conversion_stats': conversion_stats,
        'code_conversion_stats': code_conversion_stats,
        'tuning_stats': tuning_stats,
        'missing_prerequisite_skills': missing_prerequisite_skills,
        'debug_outputs': debug_outputs
    }

    return questions_choices_df, progress_data

def intelligent_question_selection(questions_choices_df, question_limit, question_col="question_content"):
    """
    Use an AI agent to intelligently select the best questions up to the specified limit,
    cycling through skills to ensure balanced representation.
    
    Args:
        questions_choices_df: DataFrame containing questions and choices
        question_limit: Maximum number of questions to select
        question_col: Column name containing question content
    
    Returns:
        tuple: (filtered_dataframe, selection_stats)
    """
    # Get unique questions for analysis
    unique_questions_df = questions_choices_df.groupby(question_col).first().reset_index()
    total_unique_questions = len(unique_questions_df)
    
    if total_unique_questions <= question_limit:
        # If we have fewer questions than the limit, return all questions
        return questions_choices_df, {
            'total_questions': total_unique_questions,
            'selected_questions': total_unique_questions,
            'question_limit': question_limit,
            'selection_needed': False,
            'reason': f"Total questions ({total_unique_questions}) <= limit ({question_limit}), no selection needed"
        }
    
    # Group questions by skill
    skill_groups = unique_questions_df.groupby('skillId')
    selected_questions = set()
    skills_list = list(skill_groups.groups.keys())
    
    # Create a candidate pool for each skill (questions not yet selected)
    skill_candidates = {}
    for skill, group in skill_groups:
        skill_candidates[skill] = group[question_col].tolist()
    
    # Cycle through skills and select the best question from each skill's candidates
    cycle_count = 0
    max_cycles = 100  # Prevent infinite loops
    
    while len(selected_questions) < question_limit and cycle_count < max_cycles:
        cycle_count += 1
        
        for skill in skills_list:
            if len(selected_questions) >= question_limit:
                break
                
            # Get remaining candidates for this skill
            candidates = [q for q in skill_candidates[skill] if q not in selected_questions]
            
            if not candidates:
                continue  # No more candidates for this skill
                
            # If only one candidate, select it
            if len(candidates) == 1:
                selected_question = candidates[0]
            else:
                # Use AI agent to select the best question from candidates
                try:
                    selected_question = select_best_question_with_ai(skill, candidates)
                except Exception:
                    # Fallback to first candidate
                    selected_question = candidates[0]
            
            selected_questions.add(selected_question)
    
    # Filter the original dataframe to only include selected questions
    filtered_df = questions_choices_df[questions_choices_df[question_col].isin(selected_questions)].reset_index(drop=True)
    
    selection_stats = {
        'total_questions': total_unique_questions,
        'selected_questions': len(selected_questions),
        'question_limit': question_limit,
        'selection_needed': True,
        'cycles_completed': cycle_count,
        'skills_processed': len(skills_list),
        'reason': f"Selected {len(selected_questions)} best questions from {total_unique_questions} using AI agent"
    }
    
    return filtered_df, selection_stats

def select_best_question_with_ai(skill, candidate_questions):
    """
    Use OpenAI to select the best question from a list of candidates for a given skill.
    
    Args:
        skill: The skill name
        candidate_questions: List of question content strings
    
    Returns:
        str: The selected best question content
    """
    # Get the selection prompt
    prompt_messages = prompts.get_question_selection_prompt(skill, candidate_questions)
    
    # Make API call
    response = settings.call_openai_with_fallback(
        model=settings.CHAT_COMPLETIONS_MODEL,
        response_format=settings.CHAT_COMPLETIONS_RESPONSE_FORMAT,
        #temperature=0.1,  # Low temperature for consistent selection
        messages=prompt_messages
    )
    
    # Parse the response
    response_content = response.choices[0].message.content.strip()
    
    try:
        # Clean and parse JSON response
        cleaned_response = response_content.replace('```json', '').replace('```', '').strip()
        selection_result = json.loads(cleaned_response)
        
        # Get the selected question index
        selected_index = selection_result.get('selected_question_index', 0)
        
        # Validate index and return selected question
        if 0 <= selected_index < len(candidate_questions):
            return candidate_questions[selected_index]
        else:
            return candidate_questions[0]
            
    except json.JSONDecodeError:
        return candidate_questions[0]  # Fallback to first candidate

def convert_questions_to_code_format_dataframe(
    df: pd.DataFrame,
    conversion_percentage: float = 0.30,
    customized_prompt_instructions: str = ""
) -> tuple[pd.DataFrame, dict]:
    """
    Convert questions to include code markdown based on coding content detection in the dataframe.
    Works directly on the filtered dataframe to preserve filtering results.
    
    Parameters:
        df: The filtered dataframe containing questions and choices
        conversion_percentage: Percentage of coding questions to convert (default 30%)
        customized_prompt_instructions: Additional instructions for the conversion
    
    Returns:
        tuple: (converted_dataframe, conversion_stats)
    """
    # Get unique questions for analysis
    unique_questions_df = df.groupby('question_content').first().reset_index()
    
    # Detect coding content in questions (simple heuristic)
    coding_questions = []
    for _, row in unique_questions_df.iterrows():
        question_content = row['question_content']
        if detect_coding_content(question_content):
            coding_questions.append(question_content)
    
    total_questions = len(unique_questions_df)
    total_coding_questions = len(coding_questions)
    
    # If no questions are coding marked, skip this step
    if total_coding_questions == 0:
        return df, {
            "converted_questions": 0,
            "total_questions": total_questions,
            "total_coding_questions": 0,
            "conversion_percentage": 0,
            "reason": "No questions with coding content found"
        }
    
    # Calculate how many questions to convert
    target_conversion_count = int(total_questions * conversion_percentage)
    
    # If less than target percentage of questions are coding, get all of them
    if total_coding_questions <= target_conversion_count:
        questions_to_convert = coding_questions
        actual_conversion_count = total_coding_questions
    else:
        # Sample target percentage from coding questions
        questions_to_convert = random.sample(coding_questions, target_conversion_count)
        actual_conversion_count = target_conversion_count
    
    converted_count = 0
    failed_conversions = 0
    
    # Convert selected questions to include code markdown
    for question_content in questions_to_convert:
        try:
            # Get all rows for this question
            question_rows = df[df['question_content'] == question_content]
            if question_rows.empty:
                continue
                
            # Get the first row to extract question data
            first_row = question_rows.iloc[0]
            
            # Prepare the question data for conversion
            question_data = {
                "question": {
                    "content": question_content,
                    "skillId": first_row.get('skillId', ''),
                    "difficultyLevelId": first_row.get('difficultyLevelId', ''),
                    "category": first_row.get('category', ''),
                    "status": first_row.get('question_status', 'ACTIVE')
                },
                "choices": []
            }
            
            # Add choices from all rows for this question
            for _, row in question_rows.iterrows():
                question_data["choices"].append({
                    "content": row.get('choice_content', ''),
                    "isCorrect": row.get('choice_isCorrect', False),
                    "orderIndex": row.get('choice_orderIndex', 0),
                    "status": row.get('choice_status', 'ACTIVE')
                })
            
            # Convert to JSON string
            questions_json = json.dumps({"questions_choices": [question_data]}, ensure_ascii=False)
            
            # Get the conversion prompt
            prompt_messages = prompts.get_code_conversion_prompt(
                first_row.get('skillId', ''),
                questions_json,
                first_row.get('difficultyLevelId', ''),
                [first_row.get('skillId', '')],  # skills
                "",  # learning_objectives (not needed for conversion)
                "",  # question_types (not needed for conversion)
                "",  # content (not needed for conversion)
                "No Change",  # customized_difficulty
                customized_prompt_instructions
            )
            
            # Make API call using the existing client from settings
            response = settings.openai_client.chat.completions.create(
                model=settings.CHAT_COMPLETIONS_MODEL,
                messages=prompt_messages,
                #temperature=0.3,
                max_tokens=4000
            )
            
            # Parse the response
            response_content = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            try:
                # Look for JSON in the response
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    converted_data = json.loads(json_match.group())
                    
                    if "questions_choices" in converted_data and len(converted_data["questions_choices"]) > 0:
                        converted_qc = converted_data["questions_choices"][0]
                        
                        # Update the question content in the dataframe
                        df.loc[df['question_content'] == question_content, 'question_content'] = converted_qc["question"]["content"]
                        
                        # Update choices if available
                        converted_choices = converted_qc.get("choices", [])
                        for _, row_idx in enumerate(df[df['question_content'] == converted_qc["question"]["content"]].index):
                            choice_order = df.loc[row_idx, 'choice_orderIndex']
                            for choice in converted_choices:
                                if choice.get('orderIndex') == choice_order:
                                    df.loc[row_idx, 'choice_content'] = choice.get('content', df.loc[row_idx, 'choice_content'])
                                    df.loc[row_idx, 'choice_isCorrect'] = choice.get('isCorrect', df.loc[row_idx, 'choice_isCorrect'])
                                    break
                        
                        converted_count += 1
                    else:
                        failed_conversions += 1
                else:
                    failed_conversions += 1
                    
            except json.JSONDecodeError:
                # If conversion fails, keep the original question
                failed_conversions += 1
                continue
                
        except Exception:
            # If conversion fails, keep the original question
            failed_conversions += 1
            continue
    
    return df, {
        "converted_questions": converted_count,
        "failed_conversions": failed_conversions,
        "total_questions": total_questions,
        "total_coding_questions": total_coding_questions,
        "conversion_percentage": (converted_count / total_questions * 100) if total_questions > 0 else 0,
        "reason": f"Converted {converted_count} out of {actual_conversion_count} selected questions"
    }

def tune_distractors_dataframe(df: pd.DataFrame, tuning_percentage=0.20) -> tuple[pd.DataFrame, dict]:
    """
    Tune distractors for a percentage of questions to make them more challenging.
    Works directly on the filtered dataframe to preserve filtering results.
    
    Args:
        df: The filtered dataframe containing questions and choices
        tuning_percentage: Percentage of questions to tune (default 20%)
    
    Returns:
        tuple: (tuned_dataframe, tuning_stats)
    """
    # Get unique questions for analysis
    unique_questions_df = df.groupby('question_content').first().reset_index()
    total_questions = len(unique_questions_df)
    
    if total_questions == 0:
        return df, {
            "tuned_questions": 0,
            "total_questions": 0,
            "tuning_percentage": 0,
            "reason": "No questions found to tune"
        }
    
    # Calculate how many questions to tune
    questions_to_tune_count = max(1, int(total_questions * tuning_percentage))
    
    # Randomly select questions to tune
    questions_to_tune = random.sample(unique_questions_df['question_content'].tolist(), min(questions_to_tune_count, total_questions))
    
    tuned_count = 0
    failed_tuning = 0
    
    # Tune selected questions
    for question_content in questions_to_tune:
        try:
            # Get all rows for this question
            question_rows = df[df['question_content'] == question_content]
            if question_rows.empty:
                continue
                
            # Get the first row to extract question data
            first_row = question_rows.iloc[0]
            
            # Prepare the question data for tuning
            question_data = {
                "question": {
                    "content": question_content,
                    "skillId": first_row.get('skillId', ''),
                    "difficultyLevelId": first_row.get('difficultyLevelId', ''),
                    "category": first_row.get('category', ''),
                    "status": first_row.get('question_status', 'ACTIVE')
                },
                "choices": []
            }
            
            # Add choices from all rows for this question
            for _, row in question_rows.iterrows():
                question_data["choices"].append({
                    "content": row.get('choice_content', ''),
                    "isCorrect": row.get('choice_isCorrect', False),
                    "orderIndex": row.get('choice_orderIndex', 0),
                    "status": row.get('choice_status', 'ACTIVE')
                })
            
            # Get the tuning prompt
            prompt_messages = prompts.get_distractor_tuning_prompt(question_data)
            
            # Make API call
            response = settings.call_openai_with_fallback(
                model=settings.CHAT_COMPLETIONS_MODEL,
                response_format=settings.CHAT_COMPLETIONS_RESPONSE_FORMAT,
                #temperature=settings.CHAT_COMPLETIONS_TEMPERATURE,
                messages=prompt_messages
            )
            
            # Parse the response
            response_content = response.choices[0].message.content.strip()
            
            # Clean and parse JSON response
            try:
                # Remove any JSON code fences
                cleaned_response = response_content.replace('```json', '').replace('```', '').strip()
                tuned_data = json.loads(cleaned_response)
                
                # Update the question with tuned distractors
                if "question" in tuned_data and "choices" in tuned_data:
                    # Update question content if changed
                    new_question_content = tuned_data["question"].get("content", question_content)
                    df.loc[df['question_content'] == question_content, 'question_content'] = new_question_content
                    
                    # Update choices
                    tuned_choices = tuned_data.get("choices", [])
                    for _, row_idx in enumerate(df[df['question_content'] == new_question_content].index):
                        choice_order = df.loc[row_idx, 'choice_orderIndex']
                        for choice in tuned_choices:
                            if choice.get('orderIndex') == choice_order:
                                df.loc[row_idx, 'choice_content'] = choice.get('content', df.loc[row_idx, 'choice_content'])
                                df.loc[row_idx, 'choice_isCorrect'] = choice.get('isCorrect', df.loc[row_idx, 'choice_isCorrect'])
                                break
                    
                    tuned_count += 1
                else:
                    failed_tuning += 1
                    
            except json.JSONDecodeError:
                failed_tuning += 1
                continue
                
        except Exception:
            failed_tuning += 1
            continue
    
    return df, {
        "tuned_questions": tuned_count,
        "failed_tuning": failed_tuning,
        "total_questions": total_questions,
        "questions_selected_for_tuning": len(questions_to_tune),
        "tuning_percentage": (tuned_count / total_questions * 100) if total_questions > 0 else 0,
        "reason": f"Tuned {tuned_count} out of {len(questions_to_tune)} selected questions"
    }
