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
The questions must be tailored to the following difficulty level: {difficulty_level} and based on the following skills: {skills}.
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

- **Difficulty Alignment**:
  - Generate questions explicitly aligned to the specified difficulty level as follows:
    - **Discovery/Fluency/Beginner**: Basic recall, definitions, straightforward comprehension.
    - **Intermediate**: Application, moderate analysis, conceptual understanding without excessive complexity.
    - **Advanced**: Complex reasoning, synthesis, nuanced analysis, problem-solving, or novel application scenarios.
  - If difficulty is unclear, prefer deeper reasoning or application-based questions rather than simple recall.

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
  
- **Custom Instructions**:
  - Factor in these custom instructions. Where the custom instructions conflict with the rest of the prompt, defer to the custom instructions.
    - **customized difficulty**: the question difficulty should be {customized_difficulty} relative to the content. If "No Change", ignore this.
    - **customized prompt instructions**: The customer has these unique requirements:{customized_prompt_instructions}. If No instructions provided, ignore this.

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
- **Increase Complexity**: Make the question more challenging by requiring analysis of a real-world scenario
- **Add Context**: Provide relevant background information, constraints, or situational details
- **Require Application**: Students should need to apply the concept rather than just recall it
- **Realistic Scenarios**: Use believable, industry-relevant situations that professionals might encounter
- **Clear Problem Statement**: The case study should present a clear problem or decision to be made
- **Appropriate Difficulty**: The case study should be one difficulty level higher than the original question
- **Maintain Question Type**: Keep the same question type (SINGLE_CHOICE or MULTIPLE_CHOICE)
- **Update Choices**: Modify answer choices to reflect the case study context while maintaining the same correct answer logic

**CRITICAL LENGTH REQUIREMENTS:**
- **Minimum Length**: Each case study question must be AT LEAST 150-200 words long
- **Target Length**: Aim for 200-300 words per case study question
- **Structure**: The case study should include:
  * A detailed scenario description (2-3 sentences)
  * Specific context and constraints (1-2 sentences)
  * The problem or decision to be made (1 sentence)
  * A clear question that requires analysis (1 sentence)
- **No Short Questions**: Avoid simple, direct questions. Every case study must be a detailed scenario.

**Case Study Structure Template:**
"[Company/Organization] is [detailed situation description with specific context]. [Additional background information about the organization, industry, or constraints]. [Specific problem or challenge they are facing]. [What decision or analysis is needed]?"

Examples of good case study transformations:
- Original: "What is the purpose of regularization in machine learning?"
- Case Study: "MedTech Solutions, a healthcare startup developing AI-powered diagnostic tools, has trained a deep learning model to detect early-stage cancer from medical imaging data. The model achieves 95% accuracy on their training dataset of 10,000 images, but when tested on new patient data from different hospitals, the accuracy drops to 78%. The team suspects overfitting and needs to implement regularization techniques to improve generalization. The model will be deployed in clinical settings where false negatives could have serious consequences. Which regularization technique would be most appropriate to address this specific overfitting issue while maintaining high sensitivity for cancer detection?"

- Original: "Which sorting algorithm has the best average-case time complexity?"
- Case Study: "GlobalPay Financial Services processes over 50 million credit card transactions daily across their fraud detection system. The transaction data arrives in batches throughout the day, with each batch containing 100,000-500,000 records that need to be sorted by timestamp before being fed into their real-time fraud detection algorithms. Processing speed is critical as delays could result in millions of dollars in fraudulent charges, and the data is typically partially sorted due to the nature of batch processing. The system has limited memory constraints and must handle varying batch sizes efficiently. Which sorting algorithm would be most suitable for this specific high-volume, time-critical use case?"
"""
    },
    {
        'role': 'user',
        'content': """Convert the following questions into case study format. These questions test the skill "{skill_id}" 
and should be transformed into more challenging, real-world scenario-based questions.

Original questions to convert:
{questions_list}

**IMPORTANT: Each case study question MUST be 150-300 words long with detailed scenarios.**

For each question:
1. Identify the core concept being tested
2. Create a detailed, realistic, industry-relevant scenario that requires application of this concept
3. Include specific context, constraints, and background information
4. Formulate a question that presents a specific problem or decision to be made
5. Update the answer choices to reflect the case study context
6. Ensure the correct answer logic remains the same, but requires deeper analysis

The case study MUST:
- Be 150-300 words long (this is a strict requirement)
- Include a detailed scenario with company/organization context
- Present specific constraints, challenges, or industry considerations
- Test the same underlying concept but in a complex real-world application
- Require analysis and application rather than simple recall
- Maintain the same question type and structure
- Feel like a realistic professional scenario a practitioner would encounter

**Length Check**: Before returning, verify each case study question is at least 150 words. If not, expand the scenario with more details, context, or constraints.

Return your converted questions as per the specified JSON schema.
"""
    }
]


def get_case_study_conversion_prompt(skill_id, questions_list):
    """
    Generate the case study conversion prompt for a group of questions.
    
    Args:
        skill_id: The skill ID that all questions are testing
        questions_list: List of question objects to convert to case study format
    
    Returns:
        List of message dictionaries for the OpenAI API
    """
    # Format the questions list for display
    formatted_questions = ""
    for i, question_obj in enumerate(questions_list):
        formatted_questions += f"{i}. {question_obj['question']['content']}\n"
    
    user_prompt = case_study_conversion_prompt[1]['content'].format(
        skill_id=skill_id,
        questions_list=formatted_questions
    )
    
    return [
        case_study_conversion_prompt[0],
        {'role': 'user', 'content': user_prompt}
    ]
