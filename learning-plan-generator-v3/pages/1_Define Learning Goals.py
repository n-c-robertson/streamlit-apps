# Import packages.
import streamlit as st
import pandas as pd 
import numpy as np
import requests
import json
import openai

# Read in supporting functions from the same directory.
import page_layout
import chat_agent
import file_management
import data_models

# Load page layout.
page_layout.streamlit_page_layout()

# add `requirements_prompt()` to the chat agent's context if this is the
# first time a user has visited this page.
if 'initial_requirements_prompt_already_ran' not in st.session_state:

	# Update the chat agent with context required for this page.
	chat_agent.update_conversation(chat_agent.requirements_prompt(), role='system')

	# Then, mark that the prompt has already been added to the chat conversation.
	st.session_state['initial_requirements_prompt_already_ran'] = True

def process_submission():

	"""
	Translate a customer's requirements into a table of goals.

		Args:
			None

		Returns:
			Prints table of goals, which is also saved to the session state
			in the variable `requirements`.
	"""
	
	# User-facing status message.
	with st.status("Creating Learning Goals..."):

		# Upload all files.
		files = file_management.upload_and_merge_files(uploads)

		# Update conversation with user input and files.
		chat_agent.update_conversation(user_input + 'Supporting Files: ' + files, role='user')
	
		# Generate Learning Goals based on input, formatted to match the output 
		# defined in data_models.LearningGoal.
		response = chat_agent.chatgpt(format_=data_models.LearningGoal).to_dataframe()

	# Save the goals.
	st.session_state['requirements'] = file_management.df_to_string_csv(response)   # string result to be used by OpenAI later.
	st.session_state['response'] = response   # dataframe result for rendering the page.

	# Send status messages.
	st.toast(page_layout.random_toast_message())
	st.success('When ready, proceed to the next step.', icon="✅")

	# Show table of goals.
	st.table(response)

# Page title.
st.title('Define Learning Goals')

# Chat interface.
user_input = st.text_area(
	'Chat interface',
	value='I\'m trying to train 50 data analysts in Python and SQL. I need to prepare them to have stronger skill sets in Python, A/B testing, and experimentation. This will allow our analysts to contribute more to engineering teams that are creating new products for our business.',
	key='input_val'
)

# File uploads. Allows pdf, docx, txt, csv, and xlsx. Supports multiple file uploads.
uploads = st.file_uploader(label='Upload Supporting Assets', type=["pdf", "docx", "txt", "csv", "xlsx"], accept_multiple_files=True)

# Trigger the processing when the 'submit' button is pressed.
if st.button('submit'):
	process_submission()

# If the user has not yet clicked submit, check to see if the user already has a saved state. If yes, show the table.
else:
	if 'response' in st.session_state:
		st.table(st.session_state['response'])
