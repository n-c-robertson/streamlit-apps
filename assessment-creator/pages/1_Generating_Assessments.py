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
import graphql_queries
import settings

#========================================
# FUNCTIONS
#========================================

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
        # Aggregate all difficulty levels, skills, and content from each content key.
        difficulty_levels = set()
        all_skills = set()
        all_content = str()
        
        for key in section['content_keys']:
            diff_obj = section['difficulty_level'].get(key)
            if diff_obj:
                difficulty_levels.add(diff_obj.get('name'))
            for skill in section['skills'].get(key, []):
                if skill.get('name'):
                    all_skills.add(skill.get('name'))
            content_str = section['content'].get(key, "")
            if content_str:
                all_content += "\n" + content_str
        
        # Convert sets to sorted lists for clarity.
        difficulties = sorted(list(difficulty_levels))
        skills = sorted(list(all_skills))
        aggregated_content = all_content.strip()
        
        start_completion_time = time.perf_counter()
        
        chat_completion_response = settings.openai_client.chat.completions.create(
            model=settings.CHAT_COMPLETIONS_MODEL,
            response_format=settings.CHAT_COMPLETIONS_RESPONSE_FORMAT,
            temperature=settings.CHAT_COMPLETIONS_TEMPERATURE,
            messages=[
                {
                    'role': 'system',
                    'content': f"""
                        You are an AI assistant for Udacity tasked with generating technical assessment learning objectives from provided Udacity content.
                        You will work with detailed course lessons, concepts, and atoms.
                        Return only valid JSON with no extra commentary.
                                        
                        JSON Schema:
                        {{
                        "title": string,
                        "objectives": [ string, string, string, string, string ]
                        }}
                        """
                                    },
                                    {
                                        'role': 'user',
                                        'content': f"""
                        Based on the following aggregated content, generate five high-level, technical learning objectives that clearly articulate what a student should understand after reviewing this content.

                        Aggregated Skills: {skills}
                        Aggregated Difficulty Levels: {difficulties}

                        Aggregated Content:
                        {aggregated_content}

                        For example, if the content were about "Machine Learning Fundamentals", a good set of learning objectives might be:

                        {{
                        "title": "Machine Learning Fundamentals",
                        "objectives": [
                            "Explain the core concepts of supervised and unsupervised learning.",
                            "Identify key algorithms used in regression and classification tasks.",
                            "Describe the process of training, evaluating, and deploying ML models.",
                            "Demonstrate how data quality and preprocessing impact model performance.",
                            "Analyze the ethical implications and limitations of ML applications."
                        ]
                        }}

                        Now, generate a similar set of objectives based on the aggregated content above.
                        """
                                    }
                                ]
                            )
                            
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

def process_concept(section_key, node, lesson, concept, difficulty_level, skills, learning_objectives, number_questions_per_concept, question_types):
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
    
    start_time = time.perf_counter()
    
    # Call the OpenAI API.
    chat_completion_response = settings.openai_client.chat.completions.create(
        model=settings.CHAT_COMPLETIONS_MODEL,
        response_format=settings.CHAT_COMPLETIONS_RESPONSE_FORMAT,
        temperature=settings.CHAT_COMPLETIONS_TEMPERATURE,
        messages=[
            {
                'role': 'system',
                'content': f"""
You are an AI assistant for Udacity tasked with generating technical assessment questions from provided Udacity content.
Your output must be a single, valid JSON object matching the schema provided below.
Return only valid JSON with no additional commentary or markdown formatting.

JSON Schema:
{{
  "questions_choices": [
    {{
      "question": {{
        "difficultyLevelId": string,
        "skillId": string,
        "category": string,
        "status": string,
        "content": string,
        "source": object
      }},
      "choices": [
        {{
          "status": string,
          "content": string,
          "isCorrect": boolean,
          "orderIndex": number
        }}
      ]
    }}
  ]
}}

Follow the instructions in the user prompt precisely. We want to generate the best technical assessments on the planet.
"""
            },
            {
                'role': 'user',
                'content': f"""
You are to generate {number_questions_per_concept} question(s) for the following Udacity content.
The questions must be tailored to the following difficulty level: {difficulty_level} and based on the following skills: {skills}.
The "difficultyLevelId" you output MUST BE this one: {difficulty_level}.
The "skillId" you output MUST BE one of the following: {skills}.

**Requirements:**  
- **Question Types**: Each question must be categorized as one of the following types: {question_types}.
- **Content Alignment**: Ensure each question is strictly based on conceptual knowledge, skills, or principles from the provided content.
- **Neutral Phrasing**: Use neutral language to ensure broad applicability and avoid context-specific references.
- **Learning Objectives**: Each question must align with at least one of the following learning objectives: {learning_objectives}.

- **Answer Choices**:
  - **Single Choice Questions**: One correct answer and three plausible distractors.
  - **Multiple Choice Questions**: Multiple correct answers (as appropriate) with distractors; total number of choices between 4 and 5.
  
- **Answer Choice Length and Detail**:  
  - All answer choices must be similar in length, detail, and complexity. The correct answer should never be obviously longer or more detailed than distractors.

- **Avoid Keyword Overlap**:  
  - The correct answer must not reuse distinctive terms or phrases directly from the question stem.

- **Difficulty Alignment**:
  - Generate questions explicitly aligned to the specified difficulty level as follows:
    - **Discovery/Fluency/Beginner**: Basic recall, definitions, straightforward comprehension.
    - **Intermediate**: Application, moderate analysis, conceptual understanding without excessive complexity.
    - **Advanced**: Complex reasoning, synthesis, nuanced analysis, problem-solving, or novel application scenarios.
  - If difficulty is unclear, prefer deeper reasoning or application-based questions rather than simple recall.

- **Distractor Guidelines**:
  - Distractors should be common misconceptions related to the content.
  - Distractors must be plausible, clear, concise, grammatically consistent, and free from clues to the correct answer.
  - Avoid negative phrasing, nonsensical content, superlatives, or combined answers (e.g., "Both A & C").
  - Maintain similarity in length and grammatical structure between correct answers and distractors to prevent unintended cues.
  - For questions involving ethical considerations, include multiple options related to ethics to avoid making the correct answer obvious.

- **Avoidance of Clues**: Ensure that the correct answer does not mimic the language of the question stem more than the distractors do.
  - Correct Answers should never be the longest answer.
  - Correct Answers should never include wording or clues from the question.
  - Incorrect answers should be plausible and realistic.
  - Questions should be meaningful on their own.

- **Programming Content**: For content that includes programming concepts, incorporate questions that assess code understanding or application.

- **Markdown Formatting for Code Snippets**:
  - **Inline Code**: Enclose short code snippets within single backticks. For example: `print("Hello, World!")`.
  - **Code Blocks**: For longer code examples or multiple lines of code, use triple backticks to create a fenced code block.
    Optionally, specify the language for syntax highlighting. For example:
    ```python
    def greet():
      print("Hello, World!")
    ```

This formatting ensures that code is clearly presented and easily readable within the assessment content.

- **Content for Question Generation**: {content}

Return only valid JSON as per the schema provided.

Example 1: Single Choice
{{
  "questions_choices": [
    {{
      "question": {{
        "difficultyLevelId": "Intermediate",
        "skillId": "Vector Operations",
        "category": "SINGLE_CHOICE",
        "status": "ACTIVE",
        "content": "What mathematical operation helps determine the relative orientation of two vectors in vector space?"
      }},
      "choices": [
        {{"status": "ACTIVE", "content": "Dot product calculation", "isCorrect": true, "orderIndex": 0}},
        {{"status": "ACTIVE", "content": "Random element extraction", "isCorrect": false, "orderIndex": 1}},
        {{"status": "ACTIVE", "content": "Scalar addition method", "isCorrect": false, "orderIndex": 2}},
        {{"status": "ACTIVE", "content": "Matrix row swapping", "isCorrect": false, "orderIndex": 3}}
      ]
    }}
  ]
}}

Example 2: Multiple Choice
{{
  "questions_choices": [
    {{
      "question": {{
        "difficultyLevelId": "Intermediate",
        "skillId": "Deep Learning Regularization Techniques",
        "category": "MULTIPLE_CHOICE",
        "status": "ACTIVE",
        "content": "Which of the following techniques help mitigate overfitting in deep learning models? Select all that apply."
      }},
      "choices": [
        {{"status": "ACTIVE", "content": "Dropout", "isCorrect": true, "orderIndex": 0}},
        {{"status": "ACTIVE", "content": "L2 Regularization", "isCorrect": true, "orderIndex": 1}},
        {{"status": "ACTIVE", "content": "Batch Normalization", "isCorrect": false, "orderIndex": 2}},
        {{"status": "ACTIVE", "content": "Data Augmentation", "isCorrect": true, "orderIndex": 3}},
        {{"status": "ACTIVE", "content": "Early Stopping", "isCorrect": true, "orderIndex": 4}}
      ]
    }}
  ]
}}
"""
            }
        ]
    )
    
    # Remove any JSON code fences and parse the result.
    chat_completion = json.loads(
        chat_completion_response.choices[0].message.content.replace('```json','').replace('```','')
    )
    elapsed = time.perf_counter() - start_time
    print(f"elapsed_completion_time_mins: {round(elapsed/60, 1)}")
    
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


def process_concepts(section_content_definitions, number_questions_per_concept, question_types):
    
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
                                    number_questions_per_concept, question_types
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
        response = settings.openai_client.chat.completions.create(
            model=settings.CHAT_COMPLETIONS_MODEL,
            response_format=settings.CHAT_COMPLETIONS_RESPONSE_FORMAT,
            temperature=settings.CHAT_COMPLETIONS_TEMPERATURE,
            messages=[
                {
                    'role': 'system',
                    'content': f"""
                    You are an AI evaluator tasked with reviewing technical assessment questions and choices generated for Udacity content.
                    Your evaluation must strictly adhere to these detailed guidelines without commentary or additional explanation.
                    Output only a valid JSON object matching exactly the schema provided below.

                    Evaluation Criteria:

                    - **Relevance and Clarity**:
                    - Must align with the specified difficulty level {qc['question']['difficultyLevelId']} and skill: {qc['question']['skillId']}.
                    - Language should be neutral, clear, concise, and strictly conceptual without context-specific references.

                    - **Question Type Suitability**:
                    - Question type ({qc['question']['category']}) must match one of these types: {question_types}.
                    - Ensure diversity and conceptual coverage.

                    - **Answer Choices**:
                    - For SINGLE_CHOICE: one correct, three plausible distractors.
                    - For MULTIPLE_CHOICE: multiple correct answers (where applicable) with 4-5 total options.
                    - Distractors must be plausible, distinct, concise, and grammatically consistent.
                    - Distractors must represent common misconceptions or errors, avoiding negative or ambiguous wording.
                    - Correct answers must not be overly lengthy or mimic language from the question stem.

                    - **General Adherence**:
                    - Strictly avoid context-specific examples or anecdotes.
                    - Question must align explicitly with provided learning objectives: {learning_objectives}.
                    - For programming questions, ensure clear markdown formatting:
                        - Inline Code: Use single backticks. Example: `print("Hello, World!")`
                        - Code Blocks: Use triple backticks with optional language specification for syntax highlighting.

                    JSON Schema:
                    {{
                    "questionEvaluation": string,     // "PASS" if criteria fully met; otherwise "FAIL"
                    "relevanceAndClarity": string,    // Concise evaluation of relevance, clarity, and alignment with skill level
                    "questionTypeDiversity": string,  // Evaluation of question type suitability and diversity
                    "choiceQuality": string,          // Quality and appropriateness of answer choices
                    "generalAdherence": string        // Adherence to learning objectives and question guidelines
                    }}
                    """
                                    },
                                    {
                                        'role': 'user',
                                        'content': f"""
                    Evaluate this JSON-formatted question and answer choices:

                    {qc}

                    Return your evaluation as per the specified JSON schema.
                    """
                                    }
                                ]
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
) -> pd.DataFrame:
    # --- Phase 1: collapse to one row per distinct question ---
    unique_qs = (
        df[[question_col]]
          .drop_duplicates()    # keep the first appearance of each exact question string
          .reset_index()         # preserve original df index in 'index' column
          .rename(columns={'index':'orig_row'})
    )
    
    # --- Phase 2: TF–IDF + cosine on the unique questions only ---
    tfidf = TfidfVectorizer().fit_transform(unique_qs[question_col].astype(str))
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
    
    # these are the *positions* in unique_qs that we want to KEEP
    keep_positions = [i for i in range(n) if i not in to_drop]
    keep_questions = unique_qs.loc[keep_positions, question_col]
    
    # --- Phase 4: filter the *original* df to only those questions ---
    return (
        df[df[question_col].isin(keep_questions)]
          .reset_index(drop=True)
    )

def generate_assessments(PROGRAM_KEYS, QUESTION_TYPES, NUMBER_QUESTIONS_PER_CONCEPT, TEMPERATURE, progress_bar=None, progress_text=None):
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
        "Deduplicating questions..."
    ]
    total_steps = len(steps)

    def update_progress(step_index):
        if progress_bar:
            progress_bar.progress((step_index + 1) / total_steps)
        if progress_text:
            progress_text.text(steps[step_index])

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
    section_content_definitions = learning_objective_generator(section_content_definitions)

    step += 1
    update_progress(step)
    section_content_definitions = process_concepts(section_content_definitions, NUMBER_QUESTIONS_PER_CONCEPT, QUESTION_TYPES)

    step += 1
    update_progress(step)
    section_content_definitions = redistribute_order_indices(section_content_definitions)

    step += 1
    update_progress(step)
    section_content_definitions = evaluate_questions(section_content_definitions, QUESTION_TYPES)

    step += 1
    update_progress(step)
    questions_choices_df = json_to_dataframe(section_content_definitions)

    step += 1
    update_progress(step)
    questions_choices_df = dedupe_questions_keep_choices(questions_choices_df, question_col="question_content", similarity_threshold=0.75)

    return questions_choices_df


#========================================
#UI
#========================================

def main():
    st.title("Generating Assessments")
    st.markdown("Create AI-generated assessment questions for one or more Udacity programs.")
    
    # Initialize session state for storing results
    if 'generated_questions_df' not in st.session_state:
        st.session_state.generated_questions_df = None
    
    # Processing time information
    st.info("**Processing Time**: Expect this to take 5~ minutes if you are generating for 2-3 programs. If it is taking longer than expected, try removing a program key or decreasing the number of questions per concept.")

    with st.form('Generate Assessments'):
        st.markdown('#### Required Parameters')
        PROGRAM_KEYS = st.text_input(
            'Program Keys (comma separated)', 
            value='cd13303,cd13318,cd13267,cd1827', 
            placeholder='Enter program keys (comma separated)',
            help="Enter the program keys for the content you want to generate questions for (cd101, cd102, etc.)"
        )
        QUESTION_TYPES = st.multiselect(
            'Question Types', 
            ['MULTIPLE_CHOICE', 'SINGLE_CHOICE'], 
            default=['MULTIPLE_CHOICE', 'SINGLE_CHOICE'],
            help="Select the types of questions you want to generate. Only SINGLE CHOICE and MULTIPLE CHOICE problems are supported."
        )
        
        st.markdown('#### Advanced Settings')
        NUMBER_QUESTIONS_PER_CONCEPT = st.number_input(
            'Number of Questions per Concept', 
            value=1, 
            min_value=1, 
            max_value=4, 
            step=1,
            help="How many questions to generate for each learning concept. Good if you want to generate a lot of questions, but will multiply the time it takes to generate."
        )
        TEMPERATURE = st.slider(
            'Temperature', 
            value=0.2, 
            min_value=0.0, 
            max_value=1.0, 
            step=0.1,
            help="Controls creativity vs consistency. Lower values (0.1-0.3) produce more consistent questions. Higher values (0.7-0.9) produce more creative questions, with some risk of hallucinations."
        )
        st.markdown('#### Staff Password')
        password = st.text_input("Staff Password", type="password", help="Enter the required staff password")
        
        submitted = st.form_submit_button("Generate Assessments", use_container_width=True)
        
        if submitted:
            if password != st.secrets['password']:
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
                        questions_choices_df= generate_assessments(
                            PROGRAM_KEYS, 
                            QUESTION_TYPES, 
                            NUMBER_QUESTIONS_PER_CONCEPT, 
                            TEMPERATURE, 
                            progress_bar, 
                            progress_text
                        )
                    
                    # Store results in session state
                    st.session_state.generated_questions_df = questions_choices_df
                    st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ An error occurred during generation: {str(e)}")
                    st.error("Please check your program keys and try again.")

    # Display results outside the form
    if st.session_state.generated_questions_df is not None and not st.session_state.generated_questions_df.empty:
        questions_choices_df = st.session_state.generated_questions_df
        
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