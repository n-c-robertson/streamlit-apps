# Import Packages.
import streamlit as st
import pandas as pd

# Read in supporting functions from the same directory.
import data_models 
import chat_agent 
import page_layout 
import catalog
import file_management

# add `select_courses_prompt()` to the chat agent's context if this is the
# first time a user has visited this page.
if 'initial_courses_prompt_already_ran' not in st.session_state:
	chat_agent.update_conversation(chat_agent.select_courses_prompt(), role='system')
	st.session_state['initial_courses_prompt_already_ran'] = True

# Get 50~ courses that could match the user's requirements based on the context from the prior pages. Save this in the session states 
# `matches` (which cannot be overidden) and `prefiltered_catalog` (which a user can override and filter).
matches = catalog.retrieve_matching_courses(query=' '.join([st.session_state['requirements'], st.session_state['skills']]))
st.session_state['matches'] = matches

# But don't accidentally override the current preferred results.
if 'preferred_catalog' not in st.session_state:
	st.session_state['preferred_catalog'] = matches

def process_submission():

	"""
	Filter the 50~ potential matches into a smaller subset that matches the customer's preferences.

		Args:
			None

		Returns:
			A series of course cards for the preferred programs that can be explored by the user,
			and then saved in the session state variable `preferred_catalog`.
	"""

	# Each time, we always start with the immutable variable `matches` so we can refer back to these 50 courses
	# as many times as we like. If we used the mutable version, we'd only be able to filter to smaller and smaller
	# subsets of the 50 courses -- we couldn't start over, or ask for a broader set of results.
	matches = st.session_state['matches']

	# Filter courses.
	with st.status("Filtering Courses..."):
		# Update conversation with the user's input.
		chat_agent.update_conversation(user_input, role='user')
		# Intercept with a system message to make sure we only use courses from the matches list.
		chat_agent.update_conversation(f'''When processing the user's request, ONLY use courses from this catalog {matches}''', role='system')
		# Receive the results from OpenAI.
		preferred_catalog = chat_agent.chatgpt(format_=data_models.ProgramList).programs

	# Overwrite the preferred catalog session state variable with OpenAI results.
	st.session_state['preferred_catalog'] = pd.DataFrame([program.dict() for program in preferred_catalog]) # dataframe result for rendering the page.
	st.session_state['courses'] = file_management.df_to_string_csv(st.session_state['preferred_catalog']) # string result for OpenAI in the final prompt.

	# Display success message.
	st.success('When ready, proceed to the next step.', icon="✅")
	st.toast(page_layout.random_toast_message())

	# Display the courses in rows with three columns
	catalog.showCourses(st.session_state['preferred_catalog'], num_columns=3)

# Page title.
st.title('Curate Courses')

# Load page layout.
page_layout.streamlit_page_layout()

# If the user hasn't completed the previous pages, we don't want them to do anything here yet.
if st.session_state['requirements'] == 'Not Defined' or st.session_state['skills'] == 'Not Defined':
	st.error("Oops! Looks like you might not have gone through all the steps. To unlock this page, make sure you go through each of the previous steps on the other tabs.",icon='🤖')

else:
	# Chat interface.
	user_input = st.text_area(
		'Which courses do you want to include in your Learning Plan?', 
		value='I\'d like for these courses to be filtered for 10-15 courses that are the best fits for solving the Python skills I care about, as well as a handful of SQL skills-focused courses.', 
		key='input_val'
	)

	# Trigger the processing when the 'submit' button is pressed.
	if st.button('submit'):
		process_submission()

	else:
		# If someone revisits the page, they see the preferred catalog from their last interaction. If there is not last interaction,
		# it shows the full set of potential matches.
		print(type(st.session_state['preferred_catalog']))
		catalog.showCourses(st.session_state['preferred_catalog'], num_columns=3)