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


def add_program_data(section_content_definitions):

    for section in section_content_definitions:
        section['difficulty_level'] = {}
        section['skills']           = {}

        for key in section['content_keys']:
            release = graphql_queries.query_component(key)
            if not release:
                continue

            # 1) root_node_id may not exist — fall back to root_node.id
            root_id = release.get('root_node_id') \
                or release.get('root_node', {}).get('id')
            if root_id is not None:
                section['content_ids'].append(root_id)
            else:
                print(f"Warning: no root_node_id or root_node.id for key {key}")

            # 2) extract the title once, from the root_node key
            if not section['title']:
                section['title'] = release.get('root_node', {}) \
                                        .get('title', '')

            # 3) pull metadata safely
            comp = release.get('component')
            meta = (comp or {}).get('metadata')
            if not meta:
                print(f"Warning: no metadata for key {key}")
                continue

            section['difficulty_level'][key] = meta.get('difficulty_level')
            section['skills'][key]           = meta.get('teaches_skills')

    return section_content_definitions

def add_node_data(section_content_definitions):
    for section in section_content_definitions:
        section['nodes'] = {}
        # assume section['content_ids'] and section['content_keys'] line up one-to-one
        for key, node_id in zip(section['content_keys'], section['content_ids']):
            node_data = graphql_queries.query_node(node_id)
            if node_data is not None:
                section['nodes'][key] = node_data
    return section_content_definitions

def process_node(node):
    """
    Recursively traverse a node (or list of nodes) to fetch VTT content for each VideoAtom.
    This function updates the node in place.
    """

    def remove_timestamps(text):
        # Regular expression to match time stamps of the format "00:00:09.525 --> 00:00:12.850"
        timestamp_pattern = r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}'
        cleaned_text = re.sub(timestamp_pattern, '', text)
        cleaned_text = re.sub(r'\n+', '\n', cleaned_text).strip()
        return cleaned_text


    if isinstance(node, dict):
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
                                    video["vtt"] = remove_timestamps(vtt_response.text)
                                    print(f"Fetched VTT for {atom.get('title')}")
                                else:
                                    print(f"No VTT URL found for 'en-us' in {video.get('title')}")
                            except requests.RequestException as e:
                                print(f"Error fetching VTT content: {e}")
            elif isinstance(value, (dict, list)):
                process_node(value)
    elif isinstance(node, list):
        for item in node:
            process_node(item)

def process_nodes(section_content_definitions):
    # Create a list of all nodes (one per content key across sections).
    all_nodes = []
    for section in section_content_definitions:
        for key, node in section.get('nodes', {}).items():
            all_nodes.append(node)

    # Process all nodes concurrently.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_node, node) for node in all_nodes]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing a node: {e}")
    
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
                        text = atom.get("text", "").strip()
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

        for content_key in section['content_keys']:
            if content_key in section.get('content', {}):
                all_content += "\n\n" + section['content'][content_key]
            if content_key in section.get('skills', {}):
                for skill in section['skills'][content_key]:
                    all_skills.add(skill.get('name', ''))
            if content_key in section.get('difficulty_level', {}):
                difficulty = section['difficulty_level'][content_key]
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
                temperature=settings.CHAT_COMPLETIONS_TEMPERATURE,
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

def process_concept(section_key, node, lesson, concept, difficulty_level, skills, learning_objectives, number_questions_per_concept, question_types, customized_difficulty, customized_prompt_instructions):
    """
    Process one concept: extract atom content (including quizzes), build the content dictionary (with metadata),
    call the OpenAI chat completion API, process the result, and return the list of question choices.
    """

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
    
    # Use context management for OpenAI API call
    chat_completion_response = settings.call_openai_with_fallback(
        model=settings.CHAT_COMPLETIONS_MODEL,
        response_format=settings.CHAT_COMPLETIONS_RESPONSE_FORMAT,
        temperature=settings.CHAT_COMPLETIONS_TEMPERATURE,
        messages=prompts.get_assessment_questions_prompt(number_questions_per_concept,difficulty_level,skills,question_types,learning_objectives,content,customized_difficulty,customized_prompt_instructions)
    )
    
    # Remove any JSON code fences and parse the result.
    chat_completion = json.loads(
        chat_completion_response.choices[0].message.content.replace('```json','').replace('```','')
    )

    
    # Process each question choice in the response.
    for qc in chat_completion['questions_choices']:
        # Validate response.
        assert 'question' in qc and 'choices' in qc
        assert qc['question']['skillId'] in skills
        
        # Attach metadata.
        qc['question']['sectionId'] = section_key
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
    
    return chat_completion['questions_choices']


def process_concepts(section_content_definitions, number_questions_per_concept, question_types, customized_difficulty, customized_prompt_instructions):
    
    for section in section_content_definitions:
        section['questions_choices'] = []
        learning_objectives = section.get('learning_objectives')

        for key in section['content_keys']:
            difficulty_level = section['difficulty_level'][key]['name']
            skills = [s['name'] for s in section['skills'][key]]
            node = section['nodes'][key]

            futures = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for module in node.get('modules', []):
                    for lesson in module.get('lessons', []):
                        for concept in lesson.get('concepts', []):
                            futures.append(
                                executor.submit(
                                    process_concept,
                                    key, node, lesson, concept,
                                    difficulty_level, skills, learning_objectives,
                                    number_questions_per_concept, question_types, customized_difficulty, customized_prompt_instructions
                                )
                            )
                for future in concurrent.futures.as_completed(futures):
                    try:
                        qcs = future.result()
                        section['questions_choices'].extend(qcs)
                    except Exception as e:
                        print(f"Error processing a concept in section {key}: {type(e).__name__}: {e}")
                        traceback.print_exc()
                        print(qcs)

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
            temperature=settings.CHAT_COMPLETIONS_TEMPERATURE,
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
            for future in concurrent.futures.as_completed(futures, timeout=300):  # Optional timeout per future.
                try:
                    qc, eval_result = future.result(timeout=60)  # Timeout for each future's result.
                    if eval_result is not None:
                        qc['eval'] = eval_result  # Attach the evaluation result to the question object.
                except Exception as e:
                    print(f"Error in future result: {e}")
    
    return section_content_definitions



def json_to_dataframe(section_content_definitions):

    # Create mappings from skill name to skill URI and from difficulty name to difficulty URI.
    skill_to_id = {}
    difficulty_to_id = {}

    for section in section_content_definitions:
        # Process skills: section['skills'] is a dict keyed by content key.
        for content_key, skills in section.get('skills', {}).items():
            for skill in skills:
                name = skill.get('name')
                uri = skill.get('uri')
                if name and uri:
                    skill_to_id[name] = uri

        # Process difficulty levels: section['difficulty_level'] is a dict keyed by content key.
        for content_key, difficulty in section.get('difficulty_level', {}).items():
            if difficulty and difficulty.get('name') and difficulty.get('uri'):
                difficulty_to_id[difficulty['name']] = difficulty['uri']

    rows = []

    # Iterate over each section in your section_content_definitions.
    for section in section_content_definitions:
        # For each question–choice in the section.
        for qc in section.get('questions_choices', []):
            # Get the evaluation result already stored in the question object.
            eval_result = qc.get('eval', {})
            question = qc.get('question', {})
            for choice in qc.get('choices', []):
                # Get skill and difficulty URIs using the mappings
                skill_name = question.get('skillId')
                difficulty_name = question.get('difficultyLevelId')
                skill_uri = skill_to_id.get(skill_name, '')
                difficulty_uri = difficulty_to_id.get(difficulty_name, '')
                
                row = {
                    'sectionId': question.get('sectionId'),
                    'difficultyLevelId': question.get('difficultyLevelId'),
                    'difficultyLevelUri': difficulty_uri,
                    'skillId': question.get('skillId'),
                    'skillUri': skill_uri,
                    'category': question.get('category'),
                    'question_status': question.get('status'),
                    'question_content': question.get('content'),
                    'sourceCategory': question.get('sourceCategory'),
                    'source': question.get('source'),
                    'choice_status': choice.get('status'),
                    'choice_content': choice.get('content'),
                    'choice_isCorrect': choice.get('isCorrect'),
                    'choice_orderIndex': choice.get('orderIndex'),
                    # Include evaluation details if present.
                    'questionEvaluation': eval_result.get('questionEvaluation'),
                    'relevanceAndClarity': eval_result.get('relevanceAndClarity'),
                    'questionTypeDiversity': eval_result.get('questionTypeDiversity'),
                    'choiceQuality': eval_result.get('choiceQuality'),
                    'generalAdherence': eval_result.get('generalAdherence')
                }
                rows.append(row)

    questions_choices_df = pd.DataFrame(rows)

    return questions_choices_df


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
    
    # --- Phase 2: TF–IDF + cosine on the unique questions only ---
    tfidf = TfidfVectorizer().fit_transform(unique_questions_df[question_col].astype(str))
    sims  = cosine_similarity(tfidf)
    
    # --- Phase 3: pick which question-rows to drop (in the unique list) ---
    n       = sims.shape[0]
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
                temperature=settings.CHAT_COMPLETIONS_TEMPERATURE,
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
                temperature=settings.CHAT_COMPLETIONS_TEMPERATURE,
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
    batch_size: int = 50
) -> tuple[pd.DataFrame, dict]:
    """
    Convert a percentage of questions to case study format for increased complexity.
    
    Args:
        df: DataFrame containing questions and choices
        question_col: Column name containing question content
        conversion_percentage: Percentage of questions to convert (default 15%)
        batch_size: Number of questions to process in each batch
    
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
    
    print(f"Case study conversion: Selected {len(selected_questions)} questions for conversion out of {initial_unique_questions} total")
    
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
                messages = prompts.get_case_study_conversion_prompt(skill_id, question_objects)
                
                response = settings.call_openai_with_fallback(
                    model=settings.CHAT_COMPLETIONS_MODEL,
                    response_format=settings.CHAT_COMPLETIONS_RESPONSE_FORMAT,
                    temperature=settings.CHAT_COMPLETIONS_TEMPERATURE,
                    messages=messages
                )
                
                # Parse the response
                conversion_result = json.loads(response.choices[0].message.content)
                
                print(f"Case study conversion: Received {len(conversion_result.get('converted_questions', []))} converted questions for skill {skill_id}")
                
                # Process converted questions
                for converted_item in conversion_result.get('converted_questions', []):
                    original_index = converted_item.get('original_question_index')
                    case_study_question = converted_item.get('case_study_question', {})
                    choices = converted_item.get('choices', [])
                    
                    print(f"Case study conversion: Processing converted item with original_index={original_index}")
                    print(f"Case study conversion: Case study question keys: {list(case_study_question.keys())}")
                    
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
                                except Exception as e:
                                    print(f"Error creating new row for case study question: {e}")
                                    print(f"Case study question: {case_study_question}")
                                    print(f"Original row keys: {list(original_row.keys())}")
                                    continue
                    
            except Exception as e:
                print(f"Error in case study conversion for skill {skill_id}: {e}")
                # If there's an error, keep the original questions
                continue
    
    print(f"Case study conversion: Successfully converted {len(questions_that_were_converted)} unique questions")
    print(f"Case study conversion: Created {len(converted_questions)} choice rows")
    
    # Create new dataframe with converted questions
    if converted_questions:
        converted_df = pd.DataFrame(converted_questions)
        
        # Remove original questions that were converted and add the converted versions
        questions_to_remove = list(questions_that_were_converted)  # Use the set of actually converted questions
        df_without_converted = df[~df[question_col].isin(questions_to_remove)]
        
        print(f"Case study conversion: Removed {len(questions_to_remove)} original questions")
        print(f"Case study conversion: Remaining questions after removal: {len(df_without_converted)}")
        
        # Combine original questions with converted questions
        final_df = pd.concat([df_without_converted, converted_df], ignore_index=True)
        
        print(f"Case study conversion: Final dataframe has {len(final_df)} rows")
        print(f"Case study conversion: Final dataframe has {final_df[question_col].nunique()} unique questions")
    else:
        final_df = df.copy()
        print("Case study conversion: No questions were converted, returning original dataframe")
    
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
    
    # Filter questions that PASS evaluation
    passed_questions = unique_questions_df[unique_questions_df[evaluation_col] == 'PASS']['question_content'].tolist()
    
    # Filter the dataframe to only keep questions that passed evaluation
    filtered_df = df[df['question_content'].isin(passed_questions)].reset_index(drop=True)
    
    # Calculate statistics
    after_evaluation_unique_questions = len(passed_questions)
    failed_questions = initial_unique_questions - after_evaluation_unique_questions
    
    evaluation_stats = {
        'initial_unique_questions': initial_unique_questions,
        'after_evaluation_unique_questions': after_evaluation_unique_questions,
        'failed_questions': failed_questions,
        'evaluation_failure_percentage': (failed_questions / initial_unique_questions * 100) if initial_unique_questions > 0 else 0,
        'initial_choices': len(df),
        'final_choices': len(filtered_df)
    }
    
    return filtered_df, evaluation_stats

def generate_assessments(PROGRAM_KEYS, QUESTION_TYPES, NUMBER_QUESTIONS_PER_CONCEPT, CUSTOMIZED_DIFFICULTY, CUSTOMIZED_PROMPT_INSTRUCTIONS, TEMPERATURE, progress_bar=None, progress_text=None):
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

    step = 0

    update_progress(step)
    section_content_definitions = prep_program_keys(PROGRAM_KEYS)
    
    step += 1
    update_progress(step)
    section_content_definitions = add_program_data(section_content_definitions)

    step += 1
    update_progress(step)
    section_content_definitions = add_node_data(section_content_definitions)

    step += 1
    update_progress(step)
    process_nodes(section_content_definitions)

    step += 1
    update_progress(step)
    section_content_definitions = extract_content(section_content_definitions)

    step += 1
    update_progress(step)
    try:
        section_content_definitions = learning_objective_generator(section_content_definitions)
    except Exception as e:
        st.error(f"Error in learning objectives generation: {e}")
        st.error(format_exception_details(e))
        st.write(f"Section content definitions: {section_content_definitions}")
        raise e

    step += 1
    update_progress(step)
    section_content_definitions = process_concepts(section_content_definitions, NUMBER_QUESTIONS_PER_CONCEPT, QUESTION_TYPES, CUSTOMIZED_DIFFICULTY, CUSTOMIZED_PROMPT_INSTRUCTIONS)
    
    # Count questions after processing concepts
    total_questions = sum(len(section.get('questions_choices', [])) for section in section_content_definitions)
    # Estimate total choices (assuming average 4 choices per question)
    estimated_choices = total_questions * 4
    question_counts.append(("After Processing Concepts", total_questions, estimated_choices))

    step += 1
    update_progress(step)
    section_content_definitions = redistribute_order_indices(section_content_definitions)

    step += 1
    update_progress(step)
    section_content_definitions = evaluate_questions(section_content_definitions, QUESTION_TYPES)

    step += 1
    update_progress(step)
    questions_choices_df = json_to_dataframe(section_content_definitions)
    
    # Count questions after converting to dataframe
    total_choices = len(questions_choices_df)
    unique_questions = questions_choices_df['question_content'].nunique()
    question_counts.append(("After Converting to DataFrame", unique_questions, total_choices))

    step += 1
    update_progress(step)
    questions_choices_df, evaluation_stats = filter_questions_by_evaluation(questions_choices_df, evaluation_col="questionEvaluation")
    
    # Count questions after evaluation filtering
    total_choices_after_eval = len(questions_choices_df)
    unique_questions_after_eval = questions_choices_df['question_content'].nunique()
    question_counts.append(("After Evaluation Filtering", unique_questions_after_eval, total_choices_after_eval))

    step += 1
    update_progress(step)
    questions_choices_df, intermediate_counts = dedupe_questions_keep_choices(questions_choices_df, question_col="question_content", similarity_threshold=0.75)
    
    # Count questions after deduplication
    total_choices_final = len(questions_choices_df)
    unique_questions_final = questions_choices_df['question_content'].nunique()
    question_counts.append(("After Deduplication", unique_questions_final, total_choices_final))

    step += 1
    update_progress(step)
    questions_choices_df, filtering_stats = filter_content_specific_questions(questions_choices_df, question_col="question_content", batch_size=50)
    
    # Count questions after filtering
    total_choices_final_filtered = len(questions_choices_df)
    unique_questions_final_filtered = questions_choices_df['question_content'].nunique()
    question_counts.append(("After Filtering", unique_questions_final_filtered, total_choices_final_filtered))

    step += 1
    update_progress(step)
    questions_choices_df, conversion_stats = convert_questions_to_case_studies(questions_choices_df, question_col="question_content", conversion_percentage=0.15, batch_size=50)
    
    # Count questions after case study conversion
    total_choices_final_converted = len(questions_choices_df)
    unique_questions_final_converted = questions_choices_df['question_content'].nunique()
    question_counts.append(("After Case Study Conversion", unique_questions_final_converted, total_choices_final_converted))

    step += 1
    update_progress(step)

    # Return both the dataframe and the progress data
    progress_data = {
        'question_counts': question_counts,
        'evaluation_stats': evaluation_stats,
        'intermediate_counts': intermediate_counts,
        'filtering_stats': filtering_stats,
        'conversion_stats': conversion_stats
    }

    return questions_choices_df, progress_data


#========================================
#UI
#========================================

def main():
    st.title("Generating Assessments")
    st.markdown("Create AI-generated assessment questions for one or more Udacity programs.")
    
    # Initialize session state for storing results
    if 'generated_questions_df' not in st.session_state:
        st.session_state.generated_questions_df = None
    
    if 'progress_data' not in st.session_state:
        st.session_state.progress_data = {}
    
    # Processing time information
    st.info("**Processing Time**: Expect this to take 5~ minutes if you are generating for 2-3 programs. If it is taking longer than expected, try removing a program key.")

    with st.form('Generate Assessments'):
        st.markdown('#### Required Parameters')
        PROGRAM_KEYS = st.text_input(
            'Program Keys (comma separated)', 
            value='cd13303,cd13318,cd13267,cd1827', 
            placeholder='Enter program keys (comma separated)',
            help="Enter the program keys for the content you want to generate questions for (cd101, cd102, etc.)"
        )
        
        with st.expander("Advanced Settings"):
            QUESTION_TYPES = st.multiselect(
            'Question Types', 
            ['MULTIPLE_CHOICE', 'SINGLE_CHOICE'], 
            default=['MULTIPLE_CHOICE', 'SINGLE_CHOICE'],
            help="Select the types of questions you want to generate. Only SINGLE CHOICE and MULTIPLE CHOICE problems are supported."
        )

            NUMBER_QUESTIONS_PER_CONCEPT = st.select_slider('Number of Questions per Concept', options=[1, 2, 3, 4, 5], value=1, help="Select the number of questions you want to generate per concept.")

            CUSTOMIZED_DIFFICULTY = st.select_slider('Custom Difficulty', options=['Much Easier', 'Easier', 'A Little Easier', 'No Change', 'A Little Harder', 'Harder', 'Much Harder'], value='No Change', help="Adjust the base difficulty of the questions.")

            TEMPERATURE = st.slider(
            'Temperature', 
            value=0.2, 
            min_value=0.0, 
            max_value=1.0, 
            step=0.1,
            help="Controls creativity vs consistency. Lower values (0.1-0.3) produce more consistent questions. Higher values (0.7-0.9) produce more creative questions, with some risk of hallucinations."
        )

            CUSTOMIZED_PROMPT_INSTRUCTIONS = st.text_area('Custom Instructions', value='', help="Enter additional instructions for the prompt. This will be appended to the base prompt.")


        st.markdown('#### Staff Password')
        password = st.text_input("Staff Password", type="password", help="Enter the required staff password")
        
        submitted = st.form_submit_button("Generate Assessments", use_container_width=True)
        
        if submitted:
            #if password != st.secrets['password']:
            if password != 'Udacity2025!':
                st.error("❌ Incorrect password. Please try again.")
            elif not PROGRAM_KEYS.strip():
                st.error("❌ Please enter at least one program key.")
            elif not QUESTION_TYPES:
                st.error("❌ Please select at least one question type.")
            else:
                # Create progress bar and text elements
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                try:
                    with st.spinner("Generating assessments..."):
                        questions_choices_df, progress_data = generate_assessments(
                            PROGRAM_KEYS, 
                            QUESTION_TYPES, 
                            NUMBER_QUESTIONS_PER_CONCEPT, 
                            CUSTOMIZED_DIFFICULTY, 
                            CUSTOMIZED_PROMPT_INSTRUCTIONS, 
                            TEMPERATURE, 
                            progress_bar, 
                            progress_text
                        )
                    
                    # Store results in session state
                    st.session_state.generated_questions_df = questions_choices_df
                    st.session_state.progress_data = progress_data
                    st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ An error occurred during generation: {str(e)}")
                    st.error(format_exception_details(e))
                    st.error("Please check your program keys and try again.")

    # Display results outside the form
    if st.session_state.generated_questions_df is not None and not st.session_state.generated_questions_df.empty:
        questions_choices_df = st.session_state.generated_questions_df
        progress_data = st.session_state.get('progress_data', {})
        
        st.success(f"✅ Successfully generated {len(questions_choices_df)} assessment items!")
        
        # Show summary statistics
        st.markdown("### 📊 Generation Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            unique_questions = questions_choices_df['question_content'].nunique()
            st.metric("Unique Questions", unique_questions)
        with col2:
            total_choices = len(questions_choices_df)
            st.metric("Total Answer Choices", total_choices)
        with col3:
            correct_choices = questions_choices_df['choice_isCorrect'].sum()
            st.metric("Correct Answers", correct_choices)
        with col4:
            avg_choices_per_question = total_choices / unique_questions if unique_questions > 0 else 0
            st.metric("Avg Choices/Question", f"{avg_choices_per_question:.1f}")
        
        # Display question count progression
        if progress_data and 'question_counts' in progress_data:
            
            question_counts = progress_data['question_counts']
            if len(question_counts) >= 2:
                initial_questions = question_counts[0][1]
                final_questions = question_counts[-1][1]
                total_reduction = initial_questions - final_questions
                reduction_percentage = (total_reduction / initial_questions * 100) if initial_questions > 0 else 0
                generation_percentage = (final_questions / initial_questions * 100) if initial_questions > 0 else 0
                
                # Create a simple horizontal progress bar
                st.write(f"**Questions Pruned from Original Generation: {reduction_percentage:.0f}%**")
                
                # Create the progress bar using HTML/CSS for a native look
                progress_html = f"""
                <div style="margin: 15px 0;">
                    <div style="
                        background-color: #262730; 
                        height: 28px; 
                        border-radius: 14px; 
                        overflow: hidden; 
                        position: relative;
                        border: 1px solid #404040;
                        box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);
                    ">
                        <div style="
                            background: linear-gradient(90deg, #ff4b4b 0%, #ff6b6b 100%);
                            height: 100%; 
                            width: {generation_percentage}%; 
                            transition: width 0.8s ease-out;
                            border-radius: 14px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
                        "></div>
                        <div style="
                            position: absolute;
                            top: 50%;
                            left: 50%;
                            transform: translate(-50%, -50%);
                            color: white;
                            font-weight: 600;
                            font-size: 13px;
                            white-space: nowrap;
                        ">
                            {initial_questions:,} → {final_questions:,} questions
                        </div>
                    </div>
                </div>
                """
                st.markdown(progress_html, unsafe_allow_html=True)
                
                # Add a subtle breakdown
                with st.expander("Debug: Detailed breakdown of pruning", expanded=False):                    
                    # Step-by-step breakdown
                    # Get the original number of questions for consistent percentage calculations
                    original_questions = question_counts[0][1] if len(question_counts) > 0 else 0
                    
                    # Add evaluation filtering breakdown
                    if 'evaluation_stats' in progress_data:
                        evaluation_stats = progress_data['evaluation_stats']
                        evaluation_failure_percentage = (evaluation_stats['failed_questions'] / original_questions * 100) if original_questions > 0 else 0
                        
                        st.write(f"- **After evaluation filtering**: {evaluation_stats['after_evaluation_unique_questions']:,} questions ({evaluation_failure_percentage:.1f}% of original failed evaluation)")
                    
                    # Add detailed deduplication breakdown
                    if 'intermediate_counts' in progress_data:
                        intermediate_counts = progress_data['intermediate_counts']
                        tfidf_reduction_questions = intermediate_counts['initial_unique_questions'] - intermediate_counts['after_tfidf_unique_questions']
                        semantic_reduction_questions = intermediate_counts['after_tfidf_unique_questions'] - intermediate_counts['after_semantic_unique_questions']
                        
                        tfidf_percentage = (tfidf_reduction_questions / original_questions * 100) if original_questions > 0 else 0
                        semantic_percentage = (semantic_reduction_questions / original_questions * 100) if original_questions > 0 else 0
                        
                        st.write(f"- **After TF-IDF deduplication**: {intermediate_counts['after_tfidf_unique_questions']:,} questions ({tfidf_percentage:.1f}% of original removed)")
                        st.write(f"- **After semantic analysis**: {intermediate_counts['after_semantic_unique_questions']:,} questions ({semantic_percentage:.1f}% of original removed)")
                    
                    # Add content specificity filtering breakdown
                    if 'filtering_stats' in progress_data:
                        filtering_stats = progress_data['filtering_stats']
                        filtering_percentage = (filtering_stats['filtered_out_questions'] / original_questions * 100) if original_questions > 0 else 0
                        
                        st.write(f"- **After content specificity filtering**: {filtering_stats['after_filtering_unique_questions']:,} questions ({filtering_percentage:.1f}% of original removed)")

        # Display results
        st.markdown("### 📋 Generated Questions")
        st.dataframe(questions_choices_df, use_container_width=True)
        
        # Download section
        st.markdown("### 💾 Download Results")
        csv_data = questions_choices_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download {len(questions_choices_df)} Assessment Items",
            data=csv_data,
            file_name="generated_assessments.csv",
            mime="text/csv",
            use_container_width=True
        )

        
        # Next steps
        st.markdown("### Next Steps")
        st.markdown("""
        1. **Review the generated questions** in the table above
        2. **Download the CSV file** to save your results
        3. **Go to the 'Reviewing Assessments' tab** to review and edit questions one by one
        4. **Accept or reject questions** based on your quality standards
        5. **Download your final approved questions** from the review interface
        """)
        
    elif st.session_state.generated_questions_df is not None and st.session_state.generated_questions_df.empty:
        st.warning("No questions were generated. Please check your program keys and try again.")
        # Clear the empty result from session state
        st.session_state.generated_questions_df = None

if __name__ == "__main__":
    main()