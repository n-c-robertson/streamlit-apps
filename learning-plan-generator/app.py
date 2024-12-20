# For running the streamlit prototype.
import streamlit as st
import streamlit.components.v1 as components

# Set the background.
background = """
<style>
        [data-testid="stAppViewContainer"] > .main {
            background: linear-gradient(to bottom right, #FAFAFA, #E9E8FF, #4D44FF);
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            min-height: 100vh; /* Ensures it covers the full viewport height */
            position: relative; /* Set the position to relative to position the logo */
        }
        
        .logo {
            position: absolute; /* Allows for absolute positioning */
            top: 0px; /* Adjust top position as needed */
            right: -100px; /* Adjust right position as needed */
            width: 50px; /* Set logo width */
        }
        
        /* Set all Streamlit components to have white backgrounds */
        [data-testid="stTextArea"] {
            background-color: white;
            border-radius: 5px; /* Optional: Add border radius for rounded corners */
            padding: 10px; /* Optional: Add padding for better spacing */
        }
        
        [data-testid="stSidebar"] {
            background-color: #F6F6F6;
            border-radius: 5px; /* Optional: Add border radius for rounded corners */
            padding: 10px; /* Optional: Add padding for better spacing */
        }
        
        .stButton > button {
            background-color: #2015FF; /* Change to your desired button color */
            color: white; /* Text color */
            border: none; /* Remove border */
            border-radius: 5px; /* Rounded corners */
            padding: 10px 10px; /* Padding for the button */
            cursor: pointer; /* Pointer cursor on hover */
            transition: background-color 0.3s; /* Smooth transition */
        }
        
        .stButton > button:hover {
            background-color: #000D85; /* Change to your desired hover color */
            color: white;
            border: none
        }

        /* Click (active) effect */
        .stButton > button:active {
            background-color: #000D85; /* Even darker shade for click */
            color: white;
        }

        .stExpander {
            background-color: #fff; /* Change to your desired expander background color */
            border-radius: 5px; /* Rounded corners */
        }
        
        .stExpander:hover {
            background-color: #fff; /* Change to your desired hover color */
        }

        /* Style for expander title */
        .stExpander .stExpanderHeader {
            background-color: #fff; /* Header background color */
            color: #000; /* Header text color */
            padding: 10px; /* Padding for header */
            cursor: pointer; /* Pointer cursor on hover */
            transition: background-color 0.3s; /* Smooth transition for header */
        }

        .card-button {
            background-color: #2015FF; /* Custom background color */
            color: white !important; /* Custom text color */
            padding: 10px 20px; /* Padding for the button */
            text-align: center; /* Center text */
            text-decoration: none; /* Remove underline */
            display: inline-block; /* Make it behave like a button */
            border: none; /* No border */
            border-radius: 5px; /* Rounded corners */
            transition: background-color 0.3s, color 0.3s; /* Smooth transition */
        }
        
        .card-button:hover {
            background-color: #000D85; /* Background color on hover */
            color: white !important; /* Text color on hover (corrected) */
        }

</style>

<div class="main">
    <img src="https://media.licdn.com/dms/image/v2/C560BAQHiNYfm0YHKrg/company-logo_200_200/company-logo_200_200/0/1656621848677/udacity_logo?e=2147483647&v=beta&t=dA-PCLHt6eLXR8UzSr1r2JOASy7TBAd1HSPulqQiJtw" alt="Logo" class="logo">
    <!-- Other content goes here -->
</div>

"""

st.markdown(background, unsafe_allow_html=True)

# Streamlit hack for getting bootstrap styling.
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.min.js" integrity="sha384-0pUGZvbkm6XF6gxjEnlmuGrJXVbNuzT9qBBavbLwCsOGabYfZo0T0to5eqruptLy" crossorigin="anonymous"></script>
<style>
    .card-img-left {
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-top-left-radius: 4px;
        border-bottom-left-radius: 0px;
    }
    .card-body {
        padding: 1.25rem; /* Increased padding for better spacing */
    }
    .card-title {
        font-weight: bold; /* Bold title */
    }
    .chip {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 0.75em;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        color: #004085;
        background-color: #cce5ff;
    }
    .card-container {
        border: 1px solid #ccc;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        background: rgb(255,255,255,0);
        position: relative;
        transition: transform 0.3s ease, box-shadow 0.3s ease; /* Smooth transition */
    }
    .card-container:hover {
        transform: scale(1.01); /* Slightly increases the size of the card */
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4); /* Increases shadow on hover */
    }
</style>
""", unsafe_allow_html=True)

# File processing.
import PyPDF2
from docx import Document
import io

# Data processing.
import pandas as pd
import numpy as np

# Fetching / formatting data from Udacity APIs.
import requests
import datetime
import json
import time
import humanize

# For handling data Learning Plan data models.
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional, Literal

# For narrowing down the catalog based on the query.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# For chatting with OpenAI.
import openai
client = openai.OpenAI(api_key=st.secrets["OpenAI_key"])

# One time function to load catalog data. Caches after first run.
@st.cache_data
def fetch_catalog():

    # Fetch and filter catalog for public offerings.
    catalog_url = 'https://api.udacity.com/api/unified-catalog/search'
    catalog_data = requests.post(catalog_url, json={'pageSize': 1000, 'SortBy': 'avgRating'}).json()
    catalog_results = [r for r in catalog_data['searchResult']['hits'] if r['is_offered_to_public']]

    # Helper functions
    convert_duration = lambda mins: humanize.naturaldelta(datetime.timedelta(minutes=mins))
    convert_program_type = lambda t: 'FreeCourse' if t == 'Course' else 'Nanodegree' if t == 'Degree' else 'PaidCourse'
    convert_slug_to_url = lambda slug: f"https://www.udacity.com/course/{slug}"

    # Process catalog results with helper functions.
    programs = [
        {
            'program_type': convert_program_type(result['semantic_type']),
            'catalog_url': convert_slug_to_url(result['slug']),
            'duration': convert_duration(result['duration']),
            'difficulty': result['difficulty'],
            'title': result['title'],
            'summary': result['summary'],
            'skill_names': result['skill_names'],
            'image_url': result['image_url']
        } for result in catalog_results
    ]

    # Return programs.
    return programs

# Limit the catalog size to 50 most potentially relevant courses.

def retrieve_matching_courses(query, programs, top_n=50):
    # Create a list of all course skills
    summaries = [course['title'] + ' ' + course['summary'] + ' ' + ' '.join(course['skill_names']) for course in programs]
    course_titles = [course['title'] for course in programs]

    # Use TF-IDF vectorizer to vectorize query and course summaries
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([query] + summaries)
    
    # Compute cosine similarity between the query and all course summaries
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    
    # Get top N matching courses
    matching_indices = np.argsort(cosine_similarities)[::-1][:top_n]
    
    return [course_titles[i] for i in matching_indices]


# File processing.
# Helper function to process uploaded file
def process_file(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1]
    
    if file_type == 'pdf':
        return extract_text_from_pdf(uploaded_file), "Extracted Text from PDF:"
    elif file_type == 'docx':
        return extract_text_from_docx(uploaded_file), "Extracted Text from DOCX:"
    elif file_type == 'txt':
        return extract_text_from_txt(uploaded_file), "Extracted Text from TXT:"
    elif file_type == 'csv':
        return extract_data_from_csv(uploaded_file), "Extracted Data from CSV:"
    elif file_type == 'xlsx':
        return extract_data_from_excel(uploaded_file), "Extracted Data from Excel:"
    else:
        return None

# Helper function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# Helper function to extract text from DOCX
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Helper function to extract text from TXT
def extract_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")

# Helper function to extract data from CSV
def extract_data_from_csv(csv_file):
    return pd.read_csv(csv_file)

# Helper function to extract data from Excel
def extract_data_from_excel(excel_file):
    return pd.read_excel(excel_file)

# Specify data model for Learning Plans. This will be used to guide
# OpenAI's response.

class StepType(Enum):
    PROGRAM = "PROGRAM"
    ASSESSMENT = "ASSESSMENT"
    SURVEY = "SURVEY"
    PROJECT = "PROJECT"
    GRADUATION = "GRADUATION"

class StepStatus(Enum):
    LOCKED = "LOCKED"
    STARTABLE = "STARTABLE"
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"

# Requirement Model
class Requirement(BaseModel):
    description: str

class curationStep(BaseModel):
    explanation: str
    output: str

# Curation Reason Model
class curationReasoning(BaseModel):
    steps: list[curationStep]
    final_answer: str

# LearningPlanStep Base Model
class LearningPlanStep(BaseModel):
    step_type: StepType
    label: str
    image_url: str = Field(
        ...,
        description=f"""the URL to the image for the offering based on the catalog's data. You MUST find this - this data exists in the catalog. Do NOT skip."""
    )
    duration: str = Field(
        ...,
        description=f"""the string value for the amount of time based on the program catalog data."""
    )
    catalog_url: str = Field(
        ...,
        description=f"""the url to the public offering based on the program's catalog data."""
    )
    short_description: str = Field(
        ...,
        description=f"""the offering's short description based on the program's catalog data. 150 char MAX."""
    )
    long_description: str = Field(
        ...,
        description=f"""the offering's long description based on the program's catalog data."""
    )
    skills: str = Field(
        ...,
        description=f"""the skills associated with the offering based on the program's catalog data."""
    )
    difficulty: str = Field(
        ...,
        description=f"""the difficulty associated with the offering based on the program's catalog data."""
    )
    recommendation_reason: str = Field(
        ...,
        description=f"""The rationale for why this step helps solve the learning plan goal, and how these skills map 
        to the goal. This is a client facing summary that will be reviewed by a financial decision maker. Write in clear 
        straight forward language about how this offering will help the customer succeed in their learning goal."""
    )
    assessments: Optional[List['LearningPlanAssessment']] = None
    starting_requirements: List[Requirement]
    completion_requirements: List[Requirement]
    status: StepStatus
    children: Optional[List['LearningPlanStep']] = None  # Nested steps

# Step Type: Program
class LearningPlanProgram(LearningPlanStep):
    key: str
    #locale: str = 'en-us'
    use_major_version: Optional[int] = None
    use_latest_version: Optional[bool] = None

# Step Type: Assessment
class LearningPlanAssessment(LearningPlanStep):
    program: Optional[LearningPlanProgram] = None

# Learning Plan Model
class learningPlan(BaseModel):
    key: str
    #locale: str = 'en-us'
    version: str  # semver
    title: str
    short_title: str = Field(
        ...,
        description="A short, compelling, marketable title for the learning plan"
    )
    slug: str
    video_url: Optional[str] = None
    image_url: Optional[str] = None
    short_description: Optional[str] = Field(
        ...,
        description=f"""A short description of what is compelling about the learning plan."""
    )
    long_description: Optional[str] = Field(
        ...,
        description=f"""A compelling description of what is covered in the learning plan, and how it meets 
        the requirements of the original prompt."""
    )
    solutionCoverage: str = Field(
        ...,
        description=f"""A thorough analysis of how well this learning plan covers the learning goals that 
        were given, MUST be formatted in markdown."""
    )
    solutionGap: str = Field(
        ...,
        description=f"""A thorough analysis of the potential gaps / things not covered by the learning plan that are 
        need for the learning goals that were given, MUST be formatted in markdown. Suggest potential additional recommendations 
        from the Udacity catalog that could help fill these gaps. You MUST get this from the catalog, and you MUST provide 
        a URL from the catalog data."""
    )
    prerequisites: str = Field(
        ...,
        description=f"""A thorough analysis of the potential pre-requisites that a learner might need to succeed in this 
        plan, MUST be formatted in markdown (don't to include the title of the section). Suggest potential additional 
        recommendations from the Udacity catalog that could help satisfy these pre-requisites. You MUST get this from 
        the catalog, and you MUST provide a URL from the catalog data."""
    )
    steps: List[LearningPlanStep]
    completion_requirements: List[Requirement]
    curationReasoning: curationReasoning

# Formatting prompt into a structure that OpenAI accepts.
def prompt(message):
    return [
        {'role': 'system', 
         'content': f"""prompt: You are a solutions architect at Udacity. You create learning plans based on enterprise 
         customer's learning and development needs. You need to be persuasive and informative on why this learning plan 
         is a great match for the learner or customer's needs. Back up your justification for different parts of the plan 
         persuasively. You MUST respect constraints given to you, such as how long they want the learning plan to be or what 
         type of skills matter to them. Make sure your learning plan has a reasonable sequence, where one course prepares you 
         for the next."""},
        {'role': 'user', 
         'content': f"""{message}"""}
            ]

# Fetching OpenAI's response.
def chatgpt(message,format_,model='gpt-4o-2024-08-06'):

        result = client.beta.chat.completions.parse(
                      model=model,
                      messages=message,
                      response_format=format_)

        content = result.choices[0].message.parsed

        return content

# Generate learning plan.
def generateLearningPlan(message, jobProfile, uploadedFile):

    # Fetch programs.
    programs = fetch_catalog()

    # pre-filtering of programs.
    with st.empty():
            with st.status("Prefiltering Catalog..."):
                filtered_titles = retrieve_matching_courses(query=message, programs=programs)
                filtered_programs = [p for p in programs if p['title'] in filtered_titles]
            
            with st.status("Building Learning Plan (takes 1 minute)..."):
        
                message = f"""Build a Udacity learning plan that meets the following requirements: {message}. Build something for the 
                following job profile: {jobProfile}. ONLY use offerings in the catalog dataset, where you'll find relevant metadata 
                to the model you need to grab. Catalog: {filtered_programs}. Here is some additional information that might help: {uploadedFile}.
                REASON through this step by step. You MUST fetch all relevant data from the catalog data (example: image_url - don't skip this!)"""
                myPrompt = prompt(message)
                response = chatgpt(myPrompt, format_=learningPlan)

   # Manually convert the response object to a dictionary
    def requirement_to_dict(requirement):
        return {"description": requirement.description}
    
    def step_to_dict(step):
        # Base step dictionary with common fields
        step_dict = {
            "step_type": step.step_type.value,
            "label": step.label,
            "duration": step.duration if hasattr(step, 'duration') else None,
            "difficulty": step.difficulty if hasattr(step, 'difficulty') else None,
            "catalog_url": step.catalog_url if hasattr(step, 'catalog_url') else None,
            "image_url": step.image_url if hasattr(step, 'image_url') else ([program['image_url'] for program in programs if program['title'] == step.label][0] if any(program['title'] == step.label for program in programs) else None),            "short_description": step.short_description if hasattr(step, 'short_description') else None,
            "long_description": step.long_description if hasattr(step, 'long_description') else None,
            "skills": step.skills if hasattr(step, 'skills') else None,
            "recommendation_reason": step.recommendation_reason if hasattr(step, 'recommendation_reason') else '',
            "status": step.status.value,
            "starting_requirements": [requirement_to_dict(req) for req in step.starting_requirements],
            "completion_requirements": [requirement_to_dict(req) for req in step.completion_requirements],
            "children": [step_to_dict(child) for child in step.children] if step.children else []
        }

        # Handle specific fields for LearningPlanProgram
        if isinstance(step, LearningPlanProgram):
            step_dict.update({
                "key": step.key,
                #"locale": step.locale,
                "use_major_version": step.use_major_version,
                "use_latest_version": step.use_latest_version
            })

        # Handle specific fields for LearningPlanAssessment
        if isinstance(step, LearningPlanAssessment):
            step_dict["program"] = step_to_dict(step.program) if step.program else None

        return step_dict

    # Convert the entire learning plan to a dictionary
    response_dict = {
        "key": response.key,
        "version": response.version,
        "title": response.title,
        "short_title": response.short_title,
        "slug": response.slug,
        "video_url": response.video_url,
        "image_url": response.image_url,
        "short_description": response.short_description,
        "long_description": response.long_description,
        "solution_coverage": response.solutionCoverage,
        "solution_gap": response.solutionGap,
        "prerequisites": response.prerequisites,
        "steps": [step_to_dict(step) for step in response.steps],
        "completion_requirements": [requirement_to_dict(req) for req in response.completion_requirements],
        "curation_reasoning": response.curationReasoning
    }

    return response_dict, filtered_titles


def horizontalCard(step):
    """Generates an HTML card for a learning step."""
    skills_chips = ''
    if step['skills']:
        skills_list = step['skills'].split(', ')  # Assuming skills are a comma-separated string
        skills_chips = ' '.join(f'<span class="chip">{skill}</span>' for skill in skills_list)

    return f"""
        <div class="container mt-5">
            <div class="card">
                <div class="row g-0">
                    <div class="col-3">
                        <img src=\"{step['image_url']}\" class="card-img-left" alt="Learning Step Image">
                    </div>
                    <div class="col-9">
                        <div class="card-body">
                            <h5 class="card-title">{step['label']}</h5>
                            <p class="card-text"><b>Description:</b> {step['short_description']}</p>
                            <p><b>Skills:</b> {skills_chips if skills_chips else 'N/A'}</p>
                            <a href=\"{step['catalog_url']}\" target="_blank" class="card-button">View Program</a>
                        </div>
                    </div>
                </div>
                <div class="card-footer text-muted">
                    <small>Duration: {step['duration']}</small>
                    <span class="float-end"><small>Difficulty: {step['difficulty']}</small></span>
                </div>
            </div>
        </div>
    """



def learning_plan_generator():

    with st.sidebar:
        generatePlan = st.button("Generate Plan")
        
        # Input for learning requirements
        learningRequirements = st.text_area(
            "Learning Requirements", 
             placeholder="Enter your learning requirements...",
             value=f"""Generate a learning plan for quantitative analysts. I want to take entry level data analysts and prepare them to build AI/ML workflows in quant contexts. It should take 6 months EXACTLY.""",
             height=100
        )
        
        # Input for job description
        jobProfile = st.text_area(
            "Job Description", 
            placeholder="Enter the job description...", 
            value=f"""I'm training data analysts into quantitative analysts. Quants do financial modeling, predicting, and forecasting to guide investments at our company.""",
            height=100
        )
    
        fileUpload = st.file_uploader("Feel free to upload supporting assets", type=["pdf", "docx", "txt", "csv", "xlsx"], accept_multiple_files=True)
    
    # If the plan was clicked.
    if generatePlan:

        # If both requirements and job profile data was provided.
        if learningRequirements and jobProfile:

            # If there is a file / list of files, process them.
            if fileUpload is not None:

                # Empty var for file concatenation.
                unifiedFile = ''

                # Process each file according to its type.
                for f in fileUpload:
                    file = process_file(f)

                    # Add it to the unified file.
                    unifiedFile += file

            # Otherwise, specifiy no file given.
            else:
                unifiedFile = 'No Support Assets Provided.'

            # Create plan.
            plan, considered_titles = generateLearningPlan(learningRequirements, jobProfile, unifiedFile)
            
            # Basic formatted Learning Plans.
            st.markdown(f"# **<{plan['title']}**")

            for step in plan['steps']:
                st.markdown(horizontalCard(step), unsafe_allow_html=True)
                                
            # Divider before Summary
            st.divider()

            st.subheader('Summary')

            # Description section
            with st.expander(f"Short Description", expanded=False):
                st.write(plan['short_description'])

            with st.expander(f"Long Description", expanded=False):
                st.write(plan['long_description'])

            # Optional Solution Coverage and Gaps
            if plan['solution_coverage']:
                with st.expander(f"Solution Coverage", expanded=False):
                    st.write(plan['solution_coverage'])

            if plan['solution_gap']:
                with st.expander(f"Solution Gaps", expanded=False):
                    st.write(plan['solution_gap'])

            # Optional Prerequisites
            if plan['prerequisites']:
                with st.expander(f"Prerequisites", expanded=False):
                    st.write(plan['prerequisites'])

            # Divider before diagnostic
            st.divider()
                
            # Some diagnostics for monitoring.
            st.subheader('Diagnostics')

            with st.expander('Expand to see underlying data structure...'):
                st.write(plan)

            with st.expander('Searching these courses for a Learning Plan...'):
                st.write(considered_titles)

            with st.expander('AI step-by-step logic for how it reached this curation.'):
                st.write(plan['curation_reasoning'])

        # Throw eror if both not called.
        else:
            st.error("Please fill in both fields before submitting.")
            
    elif not generatePlan:
        st.header("**Generate Your First Learning Plan!**")
        st.markdown(f"""
            * Write a short and clear description of the goals of the learning plan.
            * Write a short and clear description of the job profile this learning plan serves.
            * Optional: Upload supporting context files from the client.
            * Click "Generate Plan".""")
        
# Call the function to render the form
learning_plan_generator()
