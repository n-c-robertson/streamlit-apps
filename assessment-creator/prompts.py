"""
Prompt Templates for Assessment Generation

This module contains all the prompt templates used in the assessment generation process.
Each prompt is defined as a list of message dictionaries and has a corresponding helper function
that formats the prompt with the required parameters.

Prompts included:
1. Learning Objectives Generation - Creates learning objectives from aggregated content
2. Question Generation - Generates assessment questions from content
3. Question Evaluation - Evaluates generated questions for quality and adherence

All JSON examples in the prompts use escaped curly braces ({{ and }}) to prevent
format string interpretation issues.
"""

# 1st prompt: generate learning objectives based on aggregated content.
generate_learning_objections_prompt = [
    {
        'role': 'system',
        'content': """You are an AI assistant for Udacity tasked with generating technical assessment learning objectives from provided Udacity content.
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
        'content': """Based on the following aggregated content, generate five high-level, technical learning objectives that clearly articulate what a student should understand after reviewing this content.

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

def get_learning_objectives_prompt(skills, difficulties, aggregated_content):
    user_prompt = generate_learning_objections_prompt[1]['content'].format(
        skills=skills,
        difficulties=difficulties,
        aggregated_content=aggregated_content
    )
    return [
        generate_learning_objections_prompt[0],
        {'role': 'user', 'content': user_prompt}
    ]


question_generation_prompt = [
    {
        'role': 'system',
        'content': """You are an AI assistant for Udacity tasked with generating technical assessment questions from provided Udacity content.
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
        'content': """You are to generate {number_questions_per_concept} question(s) for the following Udacity content.
The questions must be based on the following skills: {skills}.
The "difficultyLevelId" you output MUST BE this one: {difficulty_level}.
The "skillId" you output MUST BE one of the following: {skills}.

**Requirements:**  
- **Question Types**: Each question must be categorized as one of the following types: {question_types}.

- **Content Alignment**: Ensure each question is strictly based on conceptual knowledge, skills, or principles from the provided content.

- **Neutral Phrasing**: Use neutral language to ensure broad applicability and avoid context-specific references.

- **Avoid Content-Specific Questions**: 
  - DO NOT create questions that reference specific projects, code snippets, or examples from the course content
  - DO NOT ask "which code did you use" or "in which project" type questions
  - DO NOT test memory of specific page content, lesson titles, or concept names
  - DO create questions that test understanding of concepts, principles, and methodologies
  - DO create questions that could be answered by someone who learned the material from any source
  - Focus on general knowledge and application rather than specific course content

  - **Learning Objectives**: Each question must align with at least one of the following learning objectives: {learning_objectives}.

- **Answer Choices**:
  - **Single Choice Questions**: One correct answer and three plausible distractors.
  - **Multiple Choice Questions**: Multiple correct answers (as appropriate) with distractors; total number of choices between 4 and 5.

- **Answer Choice Length and Detail**:  
  - All answer choices must be similar in length, detail, and complexity. The correct answer should never be obviously longer or more detailed than distractors.

- **Avoid Keyword Overlap**:  
  - The correct answer must not reuse distinctive terms or phrases directly from the question stem.

- **Case Study Questions**:
  - If relevant, consider a short case study question that is aligned with the content.

- **Align with Blooms Taxonomy**:
  - Align the question with Bloom's taxonomy.
  - The question should be aligned with one of the 6 levels of Bloom's taxonomy: Remember, Understand, Apply, Analyze, Evaluate, Create.

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
  - If there is an opportunity to incorporate markdown formatted code snippets that feels relevant to the content, prioritize this.
  
- **Custom Instructions / Overrides**:
  - These instructions should OVERRIDE any previous instructions.
    - **customized difficulty**: the question difficulty should be {customized_difficulty} than the content.
    - **customized prompt instructions**: The customer has these unique requirements that MUST be factored in: {customized_prompt_instructions}.

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

Example 3: Code markdown question.
{{
      "question": {{
        "difficultyLevelId": "Intermediate",
        "skillId": "Python Programming",
        "category": "SINGLE_CHOICE",
        "status": "ACTIVE",
        "content": "What is the output of the following Python code?\n\n```python\nprint(len(set([1, 2, 2, 3, 4])))\n```"
      }},
      "choices": [
        {{"status": "ACTIVE", "content": "```2```", "isCorrect": false, "orderIndex": 0}},
        {{"status": "ACTIVE", "content": "```4```", "isCorrect": true, "orderIndex": 1}},
        {{"status": "ACTIVE", "content": "```5```", "isCorrect": false, "orderIndex": 2}},
        {{"status": "ACTIVE", "content": "```None```", "isCorrect": false, "orderIndex": 3}}
      ]
    }}

Example 4: Case study question.
{{
      "question": {{
        "difficultyLevelId": "Intermediate",
        "skillId": "Classification Metrics",
        "category": "SINGLE_CHOICE",
        "status": "ACTIVE",
        "content": "A healthcare startup uses an ML model to predict whether patients are at risk of a heart condition. They care most about not missing at-risk patients, even if that means more false alarms. Which evaluation metric should they prioritize?"
      }},
      "choices": [
        {{"status": "ACTIVE", "content": "Precision", "isCorrect": false, "orderIndex": 0}},
        {{"status": "ACTIVE", "content": "Recall", "isCorrect": true, "orderIndex": 1}},
        {{"status": "ACTIVE", "content": "Accuracy", "isCorrect": false, "orderIndex": 2}},
        {{"status": "ACTIVE", "content": "F1 Score", "isCorrect": false, "orderIndex": 3}}
      ]
    }}
  ]
}}
"""
    }
]

def get_assessment_questions_prompt(
    number_questions_per_concept,
    difficulty_level,
    skills,
    question_types,
    learning_objectives,
    content,
    customized_difficulty,
    customized_prompt_instructions
):
    user_prompt = question_generation_prompt[1]['content'].format(
        number_questions_per_concept=number_questions_per_concept,
        difficulty_level=difficulty_level,
        skills=skills,
        question_types=question_types,
        learning_objectives=learning_objectives,
        content=content,
        customized_difficulty=customized_difficulty,
        customized_prompt_instructions=customized_prompt_instructions
    )
    return [
        question_generation_prompt[0],
        {'role': 'user', 'content': user_prompt}
    ]

# Readiness question generation prompt for testing prerequisite skills
readiness_question_generation_prompt = [
    {
        'role': 'system',
        'content': """You are an AI assistant for Udacity tasked with generating technical assessment questions that test learners' readiness and prerequisite knowledge for engaging with provided Udacity content.
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

Follow the instructions in the user prompt precisely. We want to generate the best readiness assessments on the planet.
"""
    },
    {
        'role': 'user',
        'content': """You are to generate {number_questions_per_concept} question(s) that test learners' readiness and prerequisite knowledge for the following Udacity content.
The questions must be based on the following prerequisite skills: {skills}.
The "difficultyLevelId" you output MUST BE this one: {difficulty_level}.
The "skillId" you output MUST BE one of the following: {skills}.

**CRITICAL REQUIREMENT - SKILL RESTRICTION:**
- **ONLY use skills from this exact list**: {skills}
- **DO NOT use any skills mentioned in the content** (these are teaches_skills, not prerequisite_skills)
- **DO NOT create new skill names** - only use the skills provided above
- **Every question MUST have a skillId that exactly matches one from**: {skills}

**CRITICAL FOCUS - READINESS ASSESSMENT:**
- **Purpose**: These questions test whether learners have the foundational knowledge and skills needed to successfully engage with the provided content
- **Target**: Questions should assess prerequisite skills that are essential for understanding and applying the concepts in the content
- **Approach**: Focus on testing foundational knowledge, basic concepts, and fundamental skills that learners should have mastered before attempting this content
- **Difficulty Level**: Questions should be ONE STEP EASIER than the content being prepared for. If the content is Intermediate, questions should be Beginner-level. If the content is Advanced, questions should be Intermediate-level.
- **Content Independence**: Questions should NOT test knowledge of the specific content provided. Instead, test basic foundational knowledge that would be required to understand ANY content in this skill area.

**Requirements:**  
- **Question Types**: Each question must be categorized as one of the following types: {question_types}.

- **Prerequisite Skill Focus**: Ensure each question tests foundational knowledge and skills that are prerequisites for understanding the provided content.

- **Neutral Phrasing**: Use neutral language to ensure broad applicability and avoid context-specific references.

- **Avoid Content-Specific Questions**: 
  - DO NOT create questions that reference specific projects, code snippets, or examples from the course content
  - DO NOT ask "which code did you use" or "in which project" type questions
  - DO NOT test memory of specific page content, lesson titles, or concept names
  - DO create questions that test understanding of foundational concepts, principles, and methodologies
  - DO create questions that could be answered by someone who learned the prerequisite material from any source
  - Focus on general foundational knowledge and basic application rather than specific course content

- **Learning Objectives**: Each question must align with at least one of the following learning objectives: {learning_objectives}.

- **Answer Choices**:
  - **Single Choice Questions**: One correct answer and three plausible distractors.
  - **Multiple Choice Questions**: Multiple correct answers (as appropriate) with distractors; total number of choices between 4 and 5.

- **Answer Choice Length and Detail**:  
  - All answer choices must be similar in length, detail, and complexity. The correct answer should never be obviously longer or more detailed than distractors.

- **Avoid Keyword Overlap**:  
  - The correct answer must not reuse distinctive terms or phrases directly from the question stem.

- **Difficulty Alignment**:
  - Generate questions that are ONE STEP EASIER than the specified difficulty level:
    - **If content is Advanced**: Generate Intermediate-level questions (application, moderate analysis)
    - **If content is Intermediate**: Generate Beginner-level questions (basic recall, definitions, straightforward comprehension)
    - **If content is Beginner**: Generate Discovery/Fluency-level questions (basic recognition, simple recall)
  - Focus on foundational concepts and basic understanding rather than complex application

- **Readiness Assessment Focus**:
  - Questions should test whether learners have the necessary background knowledge
  - Focus on foundational concepts that are essential for success in the target content
  - Test basic understanding and application of prerequisite skills
  - Ensure questions assess readiness rather than mastery of the target content
  - Questions should be easier than the content they're preparing learners for

- **Content Independence**:
  - DO NOT test knowledge of the specific content provided
  - DO test basic foundational knowledge that would be required for ANY content in this skill area
  - Questions should be answerable by someone who has never seen the provided content
  - Focus on universal concepts and principles, not specific implementations or examples

- **Case Study Questions**:
  - If relevant, consider a short case study question that tests application of prerequisite skills at a basic level.

- **Align with Blooms Taxonomy**:
  - Align the question with Bloom's taxonomy.
  - The question should be aligned with one of the 6 levels of Bloom's taxonomy: Remember, Understand, Apply, Analyze, Evaluate, Create.
  - For readiness assessment, focus on Remember, Understand, and basic Apply levels.

- **Distractor Guidelines**:
  - Distractors should be common misconceptions related to the prerequisite skills.
  - Distractors must be plausible, clear, concise, grammatically consistent, and free from clues to the correct answer.
  - Avoid negative phrasing, nonsensical content, superlatives, or combined answers (e.g., "Both A & C").
  - Maintain similarity in length and grammatical structure between correct answers and distractors to prevent unintended cues.
  - For questions involving ethical considerations, include multiple options related to ethics to avoid making the correct answer obvious.

- **Avoidance of Clues**: Ensure that the correct answer does not mimic the language of the question stem more than the distractors do.
  - Correct Answers should never be the longest answer.
  - Correct Answers should never include wording or clues from the question.
  - Incorrect answers should be plausible and realistic.
  - Questions should be meaningful on their own.

- **Programming Content**: For content that includes programming concepts, incorporate questions that assess foundational code understanding or basic application.

- **Markdown Formatting for Code Snippets**:
  - **Inline Code**: Enclose short code snippets within single backticks. For example: `print("Hello, World!")`.
  - **Code Blocks**: For longer code examples or multiple lines of code, use triple backticks to create a fenced code block.
    Optionally, specify the language for syntax highlighting. For example:
    ```python
    def greet():
      print("Hello, World!")
    ```
  - If there is an opportunity to incorporate markdown formatted code snippets that feels relevant to the prerequisite skills, prioritize this.
  
- **Custom Instructions**:
  - Factor in these custom instructions. Where the custom instructions conflict with the rest of the prompt, defer to the custom instructions.
    - **customized difficulty**: the question difficulty should be {customized_difficulty} relative to the content. If "No Change", ignore this.
    - **customized prompt instructions**: The customer has these unique requirements:{customized_prompt_instructions}. If No instructions provided, ignore this.

This formatting ensures that code is clearly presented and easily readable within the assessment content.

- **Content for Readiness Assessment**: {content}

Return only valid JSON as per the schema provided.

Example 1: Single Choice (Readiness Focus - Easier than content)
{{
  "questions_choices": [
    {{
      "question": {{
        "difficultyLevelId": "Beginner",
        "skillId": "Basic Programming Concepts",
        "category": "SINGLE_CHOICE",
        "status": "ACTIVE",
        "content": "What is the primary purpose of a variable in programming?"
      }},
      "choices": [
        {{"status": "ACTIVE", "content": "To store and manipulate data", "isCorrect": true, "orderIndex": 0}},
        {{"status": "ACTIVE", "content": "To create visual effects", "isCorrect": false, "orderIndex": 1}},
        {{"status": "ACTIVE", "content": "To connect to databases", "isCorrect": false, "orderIndex": 2}},
        {{"status": "ACTIVE", "content": "To format text output", "isCorrect": false, "orderIndex": 3}}
      ]
    }}
  ]
}}

Example 2: Multiple Choice (Readiness Focus - Basic foundational knowledge)
{{
  "questions_choices": [
    {{
      "question": {{
        "difficultyLevelId": "Beginner",
        "skillId": "Mathematical Foundations",
        "category": "MULTIPLE_CHOICE",
        "status": "ACTIVE",
        "content": "Which of the following are basic mathematical operations? Select all that apply."
      }},
      "choices": [
        {{"status": "ACTIVE", "content": "Addition", "isCorrect": true, "orderIndex": 0}},
        {{"status": "ACTIVE", "content": "Subtraction", "isCorrect": true, "orderIndex": 1}},
        {{"status": "ACTIVE", "content": "Multiplication", "isCorrect": true, "orderIndex": 2}},
        {{"status": "ACTIVE", "content": "Division", "isCorrect": true, "orderIndex": 3}},
        {{"status": "ACTIVE", "content": "Matrix multiplication", "isCorrect": false, "orderIndex": 4}}
      ]
    }}
  ]
}}

Example 3: Code markdown question (Readiness Focus - Basic syntax).
{{
      "question": {{
        "difficultyLevelId": "Beginner",
        "skillId": "Basic Python Syntax",
        "category": "SINGLE_CHOICE",
        "status": "ACTIVE",
        "content": "What is the output of the following Python code?\n\n```python\nx = 5\ny = 3\nprint(x + y)\n```"
      }},
      "choices": [
        {{"status": "ACTIVE", "content": "```8```", "isCorrect": true, "orderIndex": 0}},
        {{"status": "ACTIVE", "content": "```53```", "isCorrect": false, "orderIndex": 1}},
        {{"status": "ACTIVE", "content": "```15```", "isCorrect": false, "orderIndex": 2}},
        {{"status": "ACTIVE", "content": "```Error```", "isCorrect": false, "orderIndex": 3}}
      ]
    }}

Example 4: Case study question (Readiness Focus - Basic problem-solving).
{{
      "question": {{
        "difficultyLevelId": "Beginner",
        "skillId": "Problem-Solving Fundamentals",
        "category": "SINGLE_CHOICE",
        "status": "ACTIVE",
        "content": "A student is trying to understand a simple algorithm but gets confused by the mathematical notation. What is the first step they should take to break down the problem?"
      }},
      "choices": [
        {{"status": "ACTIVE", "content": "Read the problem carefully and identify what is being asked", "isCorrect": true, "orderIndex": 0}},
        {{"status": "ACTIVE", "content": "Skip to the solution", "isCorrect": false, "orderIndex": 1}},
        {{"status": "ACTIVE", "content": "Memorize the formula", "isCorrect": false, "orderIndex": 2}},
        {{"status": "ACTIVE", "content": "Ask someone else to solve it", "isCorrect": false, "orderIndex": 3}}
      ]
    }}
  ]
}}
"""
    }
]

def get_readiness_assessment_questions_prompt(
    number_questions_per_concept,
    difficulty_level,
    skills,
    question_types,
    learning_objectives,
    content,
    customized_difficulty,
    customized_prompt_instructions
):
    user_prompt = readiness_question_generation_prompt[1]['content'].format(
        number_questions_per_concept=number_questions_per_concept,
        difficulty_level=difficulty_level,
        skills=skills,
        question_types=question_types,
        learning_objectives=learning_objectives,
        content=content,
        customized_difficulty=customized_difficulty,
        customized_prompt_instructions=customized_prompt_instructions
    )
    return [
        readiness_question_generation_prompt[0],
        {'role': 'user', 'content': user_prompt}
    ]


question_evaluation_prompt = [
    {
        'role': 'system',
        'content': """You are an AI evaluator tasked with reviewing technical assessment questions and choices generated for Udacity content.
Your evaluation must strictly adhere to these detailed guidelines without commentary or additional explanation.
Output only a valid JSON object matching exactly the schema provided below.

Evaluation Criteria:

- **Relevance and Clarity**:
- Must align with the specified difficulty level {difficulty_level} and skill: {skill}.
- Language should be neutral, clear, concise, and strictly conceptual without context-specific references.

- **Question Type Suitability**:
- Question type ({question_type}) must match one of these types: {question_types}.
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
        'content': """Evaluate this JSON-formatted question and answer choices:

{question_data}

Return your evaluation as per the specified JSON schema.
"""
    }
]


def get_question_evaluation_prompt(learning_objectives, qc, question_types):
    """
    Generate the evaluation prompt for a question choice.
    
    Args:
        learning_objectives: List of learning objectives
        qc: Question choice object containing the question and choices
        question_types: List of valid question types
    
    Returns:
        List of message dictionaries for the OpenAI API
    """
    system_prompt = question_evaluation_prompt[0]['content'].format(
        difficulty_level=qc['question']['difficultyLevelId'],
        skill=qc['question']['skillId'],
        question_type=qc['question']['category'],
        question_types=question_types,
        learning_objectives=learning_objectives
    )
    
    user_prompt = question_evaluation_prompt[1]['content'].format(
        question_data=qc
    )
    
    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]


# New prompt for semantic similarity analysis
semantic_similarity_prompt = [
    {
        'role': 'system',
        'content': """You are an AI assistant tasked with analyzing the semantic similarity of assessment questions within the same skill area.
Your goal is to identify questions that are too similar in meaning or content, even if they use different wording.

You will receive a list of questions that all test the same skill. Your task is to:
1. Analyze each question for semantic similarity
2. Identify groups of questions that are essentially testing the same concept or knowledge
3. Keep only the first question in each similarity group (the one that appears earliest in the list)

Return only valid JSON with no additional commentary.

JSON Schema:
{{
  "questions_to_keep": [number, number, number, ...],
  "reasoning": string
}}

Where:
- "questions_to_keep": Array of indices (0-based) of questions that should be kept
- "reasoning": Brief explanation of the similarity analysis and grouping decisions

Guidelines for similarity assessment:
- Questions that test the same core concept should be considered similar
- Questions that ask about the same topic but in different ways should be considered similar
- Questions that have the same learning objective should be considered similar
- Only keep questions that are genuinely different in their testing approach or content focus
"""
    },
    {
        'role': 'user',
        'content': """Analyze the following questions that all test the skill "{skill_id}" for semantic similarity.

Questions to analyze:
{questions_list}

For each question, consider:
- What concept or knowledge is being tested?
- How is the question structured?
- What type of thinking is required?

Group questions that are essentially testing the same thing, and keep only the first question from each group.

Return your analysis as per the specified JSON schema.
"""
    }
]


def get_semantic_similarity_prompt(skill_id, questions_list):
    """
    Generate the semantic similarity analysis prompt for a group of questions.
    
    Args:
        skill_id: The skill ID that all questions are testing
        questions_list: List of question strings to analyze
    
    Returns:
        List of message dictionaries for the OpenAI API
    """
    # Format the questions list for display
    formatted_questions = ""
    for i, question in enumerate(questions_list):
        formatted_questions += f"{i}. {question}\n"
    
    user_prompt = semantic_similarity_prompt[1]['content'].format(
        skill_id=skill_id,
        questions_list=formatted_questions
    )
    
    return [
        semantic_similarity_prompt[0],
        {'role': 'user', 'content': user_prompt}
    ]


# New prompt for content specificity analysis
content_specificity_prompt = [
    {
        'role': 'system',
        'content': """You are an AI assistant tasked with identifying assessment questions that are too specific to the course content.
Your goal is to filter out questions that test recall of specific course materials rather than understanding of concepts.

Return only valid JSON with no additional commentary.

JSON Schema:
{{
  "questions_to_keep": [number, number, number, ...],
  "reasoning": string
}}

Where:
- "questions_to_keep": Array of indices (0-based) of questions that should be kept (not too specific)
- "reasoning": Brief explanation of the filtering decisions

Guidelines for identifying overly specific questions:
- Questions that reference specific projects, code snippets, or examples from the course
- Questions that ask "which code did you use" or "in which project"
- Questions that test memory of specific page content rather than conceptual understanding
- Questions that require knowledge of specific course structure or organization
- Questions that reference specific lesson titles, concept names, or atom content

Questions to KEEP:
- Questions that test understanding of concepts, principles, or methodologies
- Questions that could be answered by someone who learned the material from any source
- Questions that focus on general knowledge and application rather than specific course content
- Questions that test problem-solving skills and conceptual thinking
"""
    },
    {
        'role': 'user',
        'content': """Analyze the following questions to identify those that are too specific to the course content.

Questions to analyze:
{questions_list}

For each question, determine if it:
1. Tests recall of specific course materials (REJECT)
2. Tests understanding of concepts and principles (KEEP)

Examples of questions to REJECT:
- "In the specific project, which code snippet did you use to do X?"
- "In the lesson about Y, which code did A/B/C?"
- "Which specific example was used in the course to demonstrate Z?"

Examples of questions to KEEP:
- "What is the purpose of X in programming?"
- "How would you implement Y using Z principles?"
- "Which approach is best for solving this type of problem?"

Return your analysis as per the specified JSON schema.
"""
    }
]


def get_content_specificity_prompt(questions_list):
    """
    Generate the content specificity analysis prompt for a group of questions.
    
    Args:
        questions_list: List of question strings to analyze
    
    Returns:
        List of message dictionaries for the OpenAI API
    """
    # Format the questions list for display
    formatted_questions = ""
    for i, question in enumerate(questions_list):
        formatted_questions += f"{i}. {question}\n"
    
    user_prompt = content_specificity_prompt[1]['content'].format(
        questions_list=formatted_questions
    )
    
    return [
        content_specificity_prompt[0],
        {'role': 'user', 'content': user_prompt}
    ]


# New prompt for converting questions to case study format
case_study_conversion_prompt = [
    {
        'role': 'system',
        'content': """You are an AI assistant tasked with converting assessment questions into case study format.
Your goal is to take existing questions and transform them into more challenging, real-world scenario-based questions
that test the same underlying concept but require deeper application and analysis.

Return only valid JSON with no additional commentary.

JSON Schema:
{{
  "converted_questions": [
    {{
      "original_question_index": number,
      "case_study_question": {{
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

Guidelines for case study conversion:
- **Maintain Concept Alignment**: The case study must test the same core concept as the original question
- **Add Context**: Provide relevant background information, constraints, or situational details
- **Require Application**: Students should need to apply the concept rather than just recall it
- **Realistic Scenarios**: Use believable, industry-relevant situations that professionals might encounter
- **Clear Problem Statement**: The case study should present a clear problem or decision to be made
- **Maintain Question Type**: Keep the same question type (SINGLE_CHOICE or MULTIPLE_CHOICE only - these are the only supported types)
- **Update Choices**: Modify answer choices to reflect the case study context while maintaining the same correct answer logic

**SUPPORTED QUESTION TYPES:**
- **SINGLE_CHOICE**: One correct answer and three plausible distractors (4 total choices)
- **MULTIPLE_CHOICE**: Multiple correct answers with distractors (4 total choices)

**CRITICAL LENGTH REQUIREMENTS:**
- **Target Length**: Aim for 30-60 words per case study question
- **Structure**: The case study should include:
  * A simple, clear scenario description (1-2 sentences)
  * Specific context and constraints (1 sentences)
  * A clear question the problem or decision to be made (1 sentence)

**Case Study Structure Template:**
"[Company/Organization] is [situation description with context]. [Specific problem or challenge they are facing]. [What decision or analysis is needed]?"

**Example of a good case study question**
A logistics company is gathering data from IoT sensors on fleet vehicles to enhance predictive maintenance and route optimization using real-time predictive analytics. To ensure data integrity and timeliness, which approach should be implemented for effective real-time data gathering?

"""
    },
    {
        'role': 'user',
        'content': """Convert the following questions into case study format. These questions test the skill "{skill_id}" 
and should be transformed into more challenging, real-world scenario-based questions.

Original questions to convert:
{questions_list}

For each question:
1. Identify the core concept being tested
2. Create a realistic, industry-relevant scenario that requires application of this concept
3. Formulate a question that presents a specific problem or decision to be made
4. Update the answer choices to reflect the case study context
5. Ensure the correct answer logic remains the same, but requires deeper analysis

The case study MUST:
- Test the same underlying concept but in a real-world application
- Maintain the same question type and structure
- Feel like a realistic scenario a practitioner would encounter

Return your converted questions as per the specified JSON schema.
"""
    }
]


def get_case_study_conversion_prompt(skill_id, questions_list, customized_prompt_instructions=""):
    """
    Generate the case study conversion prompt for a group of questions.
    
    Args:
        skill_id: The skill ID that all questions are testing
        questions_list: List of question objects to convert to case study format
        customized_prompt_instructions: Custom instructions to append to the prompt
    
    Returns:
        List of message dictionaries for the OpenAI API
    """
    # Format the questions list for display
    formatted_questions = ""
    for i, question_obj in enumerate(questions_list):
        formatted_questions += f"{i}. {question_obj['question']['content']}\n"
    
    # Add custom instructions if provided
    custom_instructions_text = ""
    if customized_prompt_instructions and customized_prompt_instructions.strip():
        custom_instructions_text = f"\n\n**Custom Instructions**: {customized_prompt_instructions.strip()}\n\nPlease incorporate these custom requirements into your case study conversions."
    
    user_prompt = case_study_conversion_prompt[1]['content'].format(
        skill_id=skill_id,
        questions_list=formatted_questions
    )
    
    # Append custom instructions to the user prompt
    user_prompt += custom_instructions_text
    
    return [
        case_study_conversion_prompt[0],
        {'role': 'user', 'content': user_prompt}
    ]

# Code conversion prompt for adding code markdown to questions
code_conversion_prompt = [
    {
        'role': 'system',
        'content': """You are an AI assistant for Udacity tasked with converting assessment questions to include code markdown formatting.
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

Instructions:
1. Convert the question content to include relevant code examples using markdown code blocks (```python, ```javascript, etc.)
2. Convert choice options to include code examples where appropriate
3. Ensure the code examples are relevant to the question and help illustrate the concept
4. Use appropriate language tags for code blocks (python, javascript, java, html, css, etc.)
5. Keep the original question structure and meaning intact
6. Only add code where it enhances understanding of the concept
7. Ensure all code examples are syntactically correct and follow best practices
8. **Content Alignment**: Ensure the code examples are strictly based on conceptual knowledge, skills, or principles from the provided content
9. **Learning Objectives**: Each converted question must align with at least one of the provided learning objectives
10. **Difficulty Alignment**: Maintain the specified difficulty level in your code examples
11. **Avoid Content-Specific References**: DO NOT reference specific projects, code snippets, or examples from the course content that aren't in the provided content
12. **Question Types**: Only SINGLE_CHOICE and MULTIPLE_CHOICE question types are supported
    - **SINGLE_CHOICE**: One correct answer and three plausible distractors (4 total choices)
    - **MULTIPLE_CHOICE**: Multiple correct answers with distractors (4-5 total choices)
13. **Markdown Formatting for Code Snippets**:
    - **Inline Code**: Enclose short code snippets within single backticks. For example: `print("Hello, World!")`
    - **Code Blocks**: For longer code examples or multiple lines of code, use triple backticks to create a fenced code block
    - Optionally, specify the language for syntax highlighting. For example:
      ```python
      def greet():
        print("Hello, World!")
      ```
14. **Custom Instructions**: Factor in any custom instructions provided. Where custom instructions conflict with the rest of the prompt, defer to the custom instructions

**Examples of code conversion:**

**SINGLE_CHOICE Example:**
{{
  "questions_choices": [
    {{
      "question": {{
        "difficultyLevelId": "Intermediate",
        "skillId": "Python Programming",
        "category": "SINGLE_CHOICE",
        "status": "ACTIVE",
        "content": "What is the output of the following Python code?\n\n```python\nprint(len(set([1, 2, 2, 3, 4])))\n```"
      }},
      "choices": [
        {{"status": "ACTIVE", "content": "```2```", "isCorrect": false, "orderIndex": 0}},
        {{"status": "ACTIVE", "content": "```4```", "isCorrect": true, "orderIndex": 1}},
        {{"status": "ACTIVE", "content": "```5```", "isCorrect": false, "orderIndex": 2}},
        {{"status": "ACTIVE", "content": "```None```", "isCorrect": false, "orderIndex": 3}}
      ]
    }}
  ]
}}

**MULTIPLE_CHOICE Example:**
{{
  "questions_choices": [
    {{
      "question": {{
        "difficultyLevelId": "Intermediate",
        "skillId": "Python Data Structures",
        "category": "MULTIPLE_CHOICE",
        "status": "ACTIVE",
        "content": "Which of the following Python code snippets will create a list containing the numbers 1, 2, and 3? Select all that apply.\n\n```python\n# Option analysis required\n```"
      }},
      "choices": [
        {{"status": "ACTIVE", "content": "```[1, 2, 3]```", "isCorrect": true, "orderIndex": 0}},
        {{"status": "ACTIVE", "content": "```list(range(1, 4))```", "isCorrect": true, "orderIndex": 1}},
        {{"status": "ACTIVE", "content": "```list(range(1, 3))```", "isCorrect": false, "orderIndex": 2}},
        {{"status": "ACTIVE", "content": "```[x for x in range(1, 4)]```", "isCorrect": true, "orderIndex": 3}}
      ]
    }}
  ]
}}
"""
    },
    {
        'role': 'user',
        'content': """Convert the following assessment questions to include code markdown formatting. 
The questions should include relevant code examples that help illustrate the concepts being tested.

**Context Information:**
- **Difficulty Level**: {difficulty_level}
- **Skills**: {skills}
- **Learning Objectives**: {learning_objectives}
- **Question Types**: {question_types}
- **Custom Difficulty**: {customized_difficulty}
- **Custom Instructions**: {customized_prompt_instructions}

**Original Content Context:**
{content}

**Questions to Convert:**
Skill ID: {skill_id}
Questions to convert:
{questions_list}

**Conversion Guidelines:**
- Convert the question content to include relevant code examples using markdown code blocks
- Convert choice options to include code examples where appropriate
- Ensure the code examples are relevant to the question and help illustrate the concept
- Use appropriate language tags for code blocks (python, javascript, java, html, css, etc.)
- Keep the original question structure and meaning intact
- Only add code where it enhances understanding of the concept
- Ensure all code examples are syntactically correct and follow best practices
- **Content Alignment**: Ensure the code examples are strictly based on conceptual knowledge, skills, or principles from the provided content
- **Learning Objectives**: Each converted question must align with at least one of the provided learning objectives
- **Difficulty Alignment**: Maintain the specified difficulty level in your code examples
- **Avoid Content-Specific References**: DO NOT reference specific projects, code snippets, or examples from the course content that aren't in the provided content

Convert these questions to include appropriate code markdown while maintaining their educational value and clarity.
"""
    }
]

def get_code_conversion_prompt(skill_id, questions_list, difficulty_level="", skills="", learning_objectives="", question_types="", content="", customized_difficulty="No Change", customized_prompt_instructions=""):
    """
    Format the code conversion prompt with the required parameters.
    
    Parameters:
        skill_id (str): The skill ID for the questions
        questions_list (str): JSON string of questions to convert
        difficulty_level (str): The difficulty level for the questions
        skills (str): The skills being tested
        learning_objectives (str): The learning objectives to align with
        question_types (str): The types of questions being generated
        content (str): The original content context
        customized_difficulty (str): Custom difficulty instructions
        customized_prompt_instructions (str): Custom instructions for the conversion
    
    Returns:
        list: Formatted prompt messages
    """
    user_prompt = code_conversion_prompt[1]['content'].format(
        skill_id=skill_id,
        questions_list=questions_list,
        difficulty_level=difficulty_level,
        skills=skills,
        learning_objectives=learning_objectives,
        question_types=question_types,
        content=content,
        customized_difficulty=customized_difficulty,
        customized_prompt_instructions=customized_prompt_instructions
    )
    return [
        code_conversion_prompt[0],
        {'role': 'user', 'content': user_prompt}
    ]

# Distractor tuning prompt for improving incorrect answer choices
distractor_tuning_prompt = [
    {
        'role': 'system',
        'content': """You are an AI assistant tasked with improving the quality of distractors (incorrect answer choices) in assessment questions.
Your goal is to make mild tweaks to existing distractors to make them more challenging while maintaining educational value.

A good distractor should be:
- Not obviously wrong
- True in some context
- Plausible answer
- Gives no clues to the correct answer
- Reflects a distinct misconception

Return only valid JSON with no additional commentary.

JSON Schema:
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

Guidelines for distractor improvement:
- **Maintain Original Meaning**: Keep the core concept being tested unchanged
- **Subtle Improvements**: Make only mild tweaks to make distractors more plausible
- **Common Misconceptions**: Ensure distractors represent realistic misconceptions students might have
- **Plausibility**: Each distractor should seem reasonable to someone who doesn't fully understand the concept
- **No Obvious Clues**: Avoid making any choice obviously correct or incorrect
- **Consistent Structure**: Maintain similar length and grammatical structure across all choices
- **Educational Value**: Distractors should help identify specific knowledge gaps
- **Context Appropriate**: Ensure distractors are appropriate for the difficulty level and skill being tested
- **Question Types**: Only SINGLE_CHOICE and MULTIPLE_CHOICE question types are supported
  - **SINGLE_CHOICE**: One correct answer and three plausible distractors (4 total choices)
  - **MULTIPLE_CHOICE**: Multiple correct answers with distractors (4-5 total choices)
"""
    },
    {
        'role': 'user',
        'content': """Tune the distractors in the following question to make them stronger. Only make mild tweaks to make the question a little more challenging.

Original Question:
{question_data}

**Instructions:**
- Keep the correct answer unchanged
- Improve the incorrect options to make them better distractors
- Ensure each distractor represents a plausible misconception
- Make the distractors challenging but not tricky or unfair
- Maintain the same question structure and format
- Only make subtle improvements to enhance the educational value

**Example of good distractor tuning:**
Original distractor: "Any one of these three classes"
Improved distractor: "Any one of these three classes, as inheritance allows method calls across all classes"

The improved version is more plausible because it includes a partially correct technical explanation that might confuse students who don't fully understand inheritance.

**Examples of good distractor tuning:**

**SINGLE_CHOICE Example:**
Original distractor: "Any one of these three classes"
Improved distractor: "Any one of these three classes, as inheritance allows method calls across all classes"

The improved version is more plausible because it includes a partially correct technical explanation that might confuse students who don't fully understand inheritance.

**MULTIPLE_CHOICE Example:**
Original distractor: "Normalization techniques"  
Improved distractor: "Batch Normalization (helps with training stability but doesn't directly prevent overfitting)"

The improved version provides context that makes it seem more plausible while still being incorrect for the specific question about overfitting prevention.

Return the tuned question with improved distractors as per the specified JSON schema.
"""
    }
]

def get_distractor_tuning_prompt(question_data):
    """
    Generate the distractor tuning prompt for a question.
    
    Args:
        question_data: The question object containing question and choices to tune
    
    Returns:
        List of message dictionaries for the OpenAI API
    """
    import json
    
    user_prompt = distractor_tuning_prompt[1]['content'].format(
        question_data=json.dumps(question_data, indent=2)
    )
    
    return [
        distractor_tuning_prompt[0],
        {'role': 'user', 'content': user_prompt}
    ]

# Question selection prompt for intelligent question limiting
question_selection_prompt = [
    {
        'role': 'system',
        'content': """You are an AI assistant for Udacity tasked with selecting the highest quality assessment question from a list of candidates for a specific skill.

Your goal is to identify the question that best tests the given skill with optimal:
- Clarity and precision
- Appropriate difficulty level
- Educational value
- Quality of distractors (for multiple choice)
- Practical applicability
- Alignment with learning objectives

Return only valid JSON with no extra commentary.

JSON Schema:
{{
  "selected_question_index": number,
  "reasoning": string
}}"""
    },
    {
        'role': 'user',
        'content': """Select the best question for the skill: {skill}

Evaluate each candidate question based on:
1. **Clarity**: Is the question clear, unambiguous, and well-written?
2. **Skill Alignment**: Does the question directly test the specified skill?
3. **Difficulty Appropriateness**: Is the difficulty level appropriate for the skill?
4. **Educational Value**: Does answering this question help demonstrate mastery of the skill?
5. **Question Quality**: Is this a well-constructed assessment item?
6. **Practical Relevance**: Is the question relevant to real-world application of the skill?

Candidate Questions:
{candidate_questions}

Select the question index (0-based) that represents the highest quality assessment for the skill "{skill}".

Return your selection as JSON with the question index and brief reasoning."""
    }
]

def get_question_selection_prompt(skill, candidate_questions):
    """
    Generate a prompt for selecting the best question from candidates for a skill.
    
    Args:
        skill: The skill name
        candidate_questions: List of question content strings
    
    Returns:
        List of message dictionaries for the prompt
    """
    # Format candidate questions with indices
    formatted_candidates = ""
    for i, question in enumerate(candidate_questions):
        formatted_candidates += f"\n{i}. {question}\n"
    
    user_prompt = question_selection_prompt[1]['content'].format(
        skill=skill,
        candidate_questions=formatted_candidates
    )
    
    return [
        question_selection_prompt[0],
        {'role': 'user', 'content': user_prompt}
    ]
