# For running the streamlit prototype.
import streamlit as st
import streamlit.components.v1 as components

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
            'title': result['title']
        } for result in catalog_results
    ]

    # Problem to solve: too much content for OpenAI context. Need to
    # filter it down.
    return programs[:500]

programs = fetch_catalog()

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

# LearningPlanStep Base Model
class LearningPlanStep(BaseModel):
    step_type: StepType
    label: str
    duration: str = Field(
        ...,
        description="the string value for the amount of time based on the children's program catalog data."
    )
    catalog_url: str = Field(
        ...,
        description="the url to the public offering based on the program's catalog data."
    )
    short_description: str = Field(
        ...,
        description="the offering's short description based on the program's catalog data."
    )
    long_description: str = Field(
        ...,
        description="the offering's long description based on the program's catalog data."
    )
    skills: str = Field(
        ...,
        description="the skills associated with the offering based on the program's catalog data."
    )
    recommendation_reason: str = Field(
        ...,
        description="The rationale for why this step helps solve the learning plan goal, and how these skills map to the goal. This is a client facing summary that will be reviewed by a financial decision maker. Write in clear straight forward language about how this offering will help the customer succeed in their learning goal."
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
        description="A short description of what is compelling about the learning plan."
    )
    long_description: Optional[str] = Field(
        ...,
        description="A compelling description of what is covered in the learning plan, and how it meets the requirements of the original prompt."
    )
    solutionCoverage: str = Field(
        ...,
        description="A short analysis of how well this learning plan covers the learning goals that were given."
    )
    solutionGap: str = Field(
        ...,
        descriptoin="A short analysis of the potential gaps / things not covered by the learning plan that are need for the learning goals that were given."
    )
    steps: List[LearningPlanStep]
    completion_requirements: List[Requirement]


# Formatting prompt into a structure that OpenAI accepts.
def prompt(message):
    return [
        {'role': 'system', 
         'content': f"""prompt: You are a solutions architect at Udacity. You create learning plans based on enterprise customer's learning
         and development needs."""},
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
def generateLearningPlan(message, jobProfile):
    
    message = f"""Build a Udacity learning plan that meets the following requirements: {message}. Build someone for the following job profile: {jobProfile}.
    ONLY use offerings in the catalog dataset, where you'll find relevant metadata to the model you need to grab. Catalog: {programs[:500]}/"""
    myPrompt = prompt(message)
    response = chatgpt(myPrompt, format_=learningPlan)

    print(response)

   # Manually convert the response object to a dictionary
    def requirement_to_dict(requirement):
        return {"description": requirement.description}
    
    def step_to_dict(step):
        # Base step dictionary with common fields
        step_dict = {
            "step_type": step.step_type.value,
            "label": step.label,
            "duration": step.duration if hasattr(step, 'duration') else None,
            "catalog_url": step.catalog_url if hasattr(step, 'catalog_url') else None,
            "short_description": step.short_description if hasattr(step, 'short_description') else None,
            "long_description": step.long_description if hasattr(step, 'long_description') else None,
            "skills": step.skills if hasattr(step, 'skills') else None,
            "recommendation_reason": ' '.join(step.recommendation_reason) if hasattr(step, 'recommendation_reason') else '',
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
        "steps": [step_to_dict(step) for step in response.steps],
        "completion_requirements": [requirement_to_dict(req) for req in response.completion_requirements]
    }

    return response_dict



def learning_plan_generator():
    st.title("Learning Plan Generator")
    st.write("Enter your learning requirements and job description to generate a personalized learning plan.")
    
    # Input for learning requirements
    learningRequirements = st.text_area(
        "Learning Requirements", 
         placeholder="Enter your learning requirements...",
         value=f"""Generate a learning plan for ai. I want to take people who know nothing about ai and give them some basic fluency. By the end of the plan, they should have a decent conceptual understanding of ai, as well as some basic scripting skills with ai libraries.""",
         height=100
    )
    
    # Input for job description
    jobProfile = st.text_area(
        "Job Description", 
        placeholder="Enter the job description...", 
        value=f"""I'm training to train data analysts. They will be responsible for BI / data analysis functions in the company. But, we are also trying to make them more AI / ML focused, and push for more predictive and gen AI capabilities in the company. """,
        height=100
    )
    
    # Button to submit form
    if st.button("Generate Plan"):
        if learningRequirements and jobProfile:
            st.success("Plan generating! This will take about 30 seconds to load.")

            # Create plan.
            plan = generateLearningPlan(learningRequirements, jobProfile)

            # Data dictionary for reference.
            with st.expander('Expand to see underlying data structure...'):
                st.write(plan)

            # Basic formatted Learning Plans.
            st.title(plan['title'])

            st.subheader("Short Description")
            st.write(plan['short_description'])

            st.subheader("Long Description")
            st.write(plan['long_description'])

            st.subheader("Solution Coverage")
            st.write(plan['solution_coverage'])
            
            st.subheader("Solution Gaps")
            st.write(plan['solution_gap'])
            
            st.subheader("Learning Plan Steps")

            for step in plan['steps']:
                with st.expander(step['label']):
                    st.write(f"**Duration:** {step['duration']}")
                    st.write(f"**Description:** {step['short_description']}")
                    st.write(f"**Skills:** {step['skills']}")
                    st.write(f"**Status:** {step['status']}")
                    st.write(f"**Recommendation Reason"** {step['recommendation_reason']})
                    
                    st.markdown(f"[View Program]({step['catalog_url']})")

                    st.write("**Starting Requirements:**")
                    for req in step['starting_requirements']:
                        st.write(f"- {req['description']}")

                    st.write("**Completion Requirements:**")
                    for req in step['completion_requirements']:
                        st.write(f"- {req['description']}")

            st.subheader("Completion Requirements")
            for req in plan['completion_requirements']:
                st.write(f"- {req['description']}")

        else:
            st.error("Please fill in both fields before submitting.")

# Call the function to render the form
learning_plan_generator()
