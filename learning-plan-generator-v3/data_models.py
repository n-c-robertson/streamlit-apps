# Import packages.
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, conlist
from enum import Enum
import pandas as pd

"""
This utility stores all of the different data models that are used to structured the outputs from
OpenAI.
"""

# Learning Goal: a learning goal that is tied to an enterprise customer's requirements. Learning Goal
# assumes a list of potential goals, with each one having a goal, a importance ranking (high, medium, low),
# and an explanation of the goal.

class LearningGoal(BaseModel):
	learning_goal: List[str]
	importance: conlist(Literal['high', 'medium', 'low']) 
	explanation: List[str]

	def to_dataframe(self):
		"""Convert the instance to a Pandas DataFrame."""
		data = {
			'Learning Goal': [goal for goal in self.learning_goal],
			'Importance': [importance for importance in self.importance],
			'Explanation': [explanation for explanation in self.explanation]
		}
		return pd.DataFrame(data)


# Classes for handling the Domain>Subject>Skill taxonomy. These are effectively the same, but they are named
# differently in case there is any future use cases where we need these classes to interact with each other.

class SkillsClass(BaseModel):
	name: List[str]

class SubjectsClass(BaseModel):

	name: List[str]

class DomainsClass(BaseModel):
	name: List[str]

# Program classes for represent a piece of content in the catalog.

# A program in the Learning Plan.
class Program(BaseModel):
	program_type: str
	catalog_url: str
	duration: str
	difficulty: str
	title: str
	summary: str
	skill_names: List[str]

# A list of programs, inherting the class Programs.
class ProgramList(BaseModel):
	programs: List[Program]

# The rest of the classes are used to define the Learning Plan data model. This model has a lot nested in it --
# and the prototype isn't using all of it yet (for example, we don't do anything with assessments yet). but this is
# a replication of the production data model for Learning Plans. Some modifications have been made for the sake of 
# prototype use cases.

# Step type. Discovery and fluency course types are modifications from Learning Plans.
class StepType(Enum):
	NANODEGREE = "Nanodegree"
	COURSE = "Course"
	DISCOVERY_COURSE = "Discovery Course"
	FLUENCY_COURSE = "Fluency Course"

# Step status. This not used in this prototype, but theoretically an addition could be made to
# add conditional logic to lock / unlock content based on different conditions.

#class StepStatus(Enum):
#	LOCKED = "LOCKED"
#	STARTABLE = "STARTABLE"
#	STARTED = "STARTED"
#	COMPLETED = "COMPLETED"

# This is a heavily pared down representation of a learning plan step compared to what is in production. The rationale:
# when we asked the learning plan step to render all of the catalog data, we had a high rate of hallucination errors.
# So instead, we only have determine (1) the program key and (2) the data we need to generate. From there, we later use
# the program key to fetch all of the information from the catalog data.
class LearningPlanStep(BaseModel):
	step_type: StepType
	program_key: str = Field(
		...,
		description="The udacity course key for this course. Will have something like `cd` or `nd` in it")
	recommendation_reason: str = Field(
		...,
		description=f"""The rationale for why this step helps solve the learning plan goal, and how these skills map to the goal. 
		This is a client facing summary that will be reviewed by a financial decision maker. Write in clear straight forward language 
		about how this offering will help the customer succeed in their learning goal."""
	)
	#children: None
	short_description: str = Field(
		...,
		description="the offering's short description based on the program's catalog data."
	)

# This is the Learning Plan model. This is where we lean the most on OpenAI to create metadata about the plan itself.
class LearningPlan(BaseModel):
	key: str
	#locale: str = 'en-us'
	version: str 
	title: str
	short_title: str = Field(
		...,
		description="A short, compelling, marketable title for the learning plan"
	)
	slug: str
	video_url: Optional[str] = None # Could play around with this later...
	image_url: Optional[str] = None # Could play around wit this later.
	short_description: Optional[str] = Field(
		...,
		description="A short description of what is compelling about the learning plan."
	)
	long_description: Optional[str] = Field(
		...,
		description=f"""A compelling description of what is covered in the learning plan, and how it meets the requirements 
		of the original prompt."""
	)
	solutionCoverage: str = Field(
		...,
		description=f"""A thorough analysis of how well this learning plan covers the learning goals that were given. Use plain 
		text, NO MARKDOWN, do not include a section header."""
	)
	solutionGap: str = Field(
		...,
		description=f"""A thorough analysis of the potential gaps / things not covered by the learning plan that are need for 
		the learning goals that were given. Suggest potential additional recommendations from the Udacity catalog that could help 
		fill these gaps.  Use plain text, NO MARKDOWN, do not include a section header."""
	)
	steps: List[LearningPlanStep]
	prerequisites: List[str] = Field(
		...,
		description=f"""The udadicty course keys for 3 programs that might be good prerequisites to consider for the learning plan. 
		Will have something like `cd` or `nd` in it. There MUST be 3.""")
	extracurricular: List[str] = Field(
		...,
		description=f"""The udacity program keys for 3 programs that might be good to go to after this learning plan. Will have 
		something like `cd` or `nd` in it. There MUST be 3.""")

# This mode wraps around LearningPlan to create a list of multiple plans.
class LearningPlans(BaseModel):
	plans: List[LearningPlan]

# This is something that is appended on to Learning Plans near the end of dev for the prototype -- the idea of listing coverage in the 
# skills that were identified as important to the customer.

# Defining the different coverage levels.
CoverageLevel = Literal["Strong", "Medium", "Weak"]

# Defining the model to specify the coverage for a single subject.
class SubjectCoverage(BaseModel):
	"""Model for individual subject coverage."""
	subject: str = Field(..., description="The name of the subject.")
	coverage: CoverageLevel = Field(..., description="The coverage level of the subject.")

# Defining the model to specify a list of coverages for multiple subjects.
class SkillsReport(BaseModel):
	"""Model that accepts a list of subjects and their coverage."""
	subjects: List[SubjectCoverage] = Field(..., description="A list of subjects with their coverage levels.")
