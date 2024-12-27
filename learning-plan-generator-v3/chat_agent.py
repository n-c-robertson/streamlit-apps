# Import packages
import streamlit as st
import openai

# Read in supporting functions from the same directory.
from keys import openai_key

# Set up Open AI client.
client = openai.OpenAI(api_key=openai_key())

# System prompt for translating customer requirements into goals.
def requirements_prompt():
	return f"""prompt: You are a solutions architect at Udacity. Your job is to listen to learning and development buyers at corporations, 
	and figure out what should be the objectives of their learning objectives. You should write clear, friendly, professional summaries that 
	cover the primary goals of the organization. You MUST break it down into 3 or 4 goals minimum to start, feel free to make some inferences 
	based on what the client has told you, they will correct you if you are wrong. Rank goals from highest to lowest perceived priority."""

# System prompt for translating customer skill needs into a skills map.
def generate_skills_prompt():
	return f"""prompt: You are a solutions architect at Udacity. Your job is to translate learning objectives / goals into skills that a 
	corporation needs to train to reach those goals. When you respond, factor in the organizations goals: {st.session_state['requirements']}"""

# System prompt for translating goals and skills into a list of preferred courses.
def select_courses_prompt():
	return f"""prompt: You are a solutions architect at Udacity Your job is to look as skill and learning requirements, and filter for courses that 
	you think would be a good fit for a customer's learning program. When you respond, factor in the organizations goals and skills: 
	{st.session_state['requirements']}, {st.session_state['skills']}"""

# System prompt for generating a learning plan.
def generate_learning_plans_prompt():
	return f"""prompt: You are a solutions architect at Udacity. Your job is taking a wide range of context from customers, and generating Learning 
	Plans that could achieve their goal. When you respond, factor in the the organizations goals, skills, and preferred courses. YOU MUST only 
	recommend courses that are in the provided catalog of "Preferred Courses" and "Additional Courses" - don't make your own courses up! - 
	use the exact program_key provided. Requirements: {st.session_state['requirements']}, Domains/Subjects/Skills: {st.session_state['skills']}, 
	Preferred Courses: {st.session_state['preferred_catalog']}, Secondary Courses (can use to complement preferred!): {st.session_state['matches']}"""

# System prompt for generating a skills report.
def generate_skills_report_prompt(learningPlan):
	return f"""prompt: you are now going to use the information available to you to determine how well the learning plan covers different areas. 
	Create one record for each SUBJECT - not domain, not skill. ALL SUBJECTS must be accounted for. LearningPlan: {learningPlan}"""

def update_conversation(message, role='user'):

	"""
	A utility function for updating the context window for the OpenAI chat agent.

		Args:
			message: the message to add to the context window.
			role: who spoke in the conversation (role, system, agent).

		Returns:
			Updates the session state for `conversation` to incldue the latest message.
	"""

	st.session_state['conversation'].append({'role': role, 'content': message})

def reset_conversation():

	"""
	A utility function for reseting the context window. This allows us to prune the window
	once we have all the necessary context to complete the task of creating a learning plan.
	"""

	st.session_state['conversation'] = []

def validate_and_convert_conversation_format():
	"""
	A utilty function for making sure the string is in a format that works for OpenAI.
	"""
	for index, message in enumerate(st.session_state['conversation']):
		if not isinstance(message['content'], str):
			# Convert the content to string
			message['content'] = str(message['content'])  # Convert to string


def chatgpt(format_,model='gpt-4o-2024-08-06'):

	"""
	Primary function for having a conversation with OpenAI.

		Args:
			format_: The expect format for the message OpenAI should return.
			model: The OpenAI model.
	"""

	# Quick validation to make sure the context window didn't get corrupted.
	validate_and_convert_conversation_format()

	# Get result from OpenAI based on the conversation. Eventually, I'll need to 
	# update this to remove the beta reference that was required when first using
	# structured outputs from OpenAI.
	result = client.beta.chat.completions.parse(
					  model=model,
					  messages=st.session_state['conversation'],
					  response_format=format_
					  )

	# Get the response.
	content = result.choices[0].message.parsed

	# Add the response to the context window.
	update_conversation(content, role='assistant')

	# Return the response.
	return content