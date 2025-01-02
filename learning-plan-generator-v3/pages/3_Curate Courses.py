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

def generate_catalog(type_selections,duration_selections,difficulty_selections):

	"""
	Generate a catalog based on the filters passed through.
	"""

	# Get 50~ courses that could match the user's requirements based on the context from the prior pages and their filters. Save this in 
	# the session states `matches` (which cannot be overidden) and `prefiltered_catalog` (which a user can override and filter).
	query = ' '.join([st.session_state['requirements'], st.session_state['skills']])

	matches = catalog.retrieve_matching_courses(
		query=query,
		type_selections=type_selections,
		duration_selections=duration_selections,
		difficulty_selections=difficulty_selections
		)

	st.session_state['matches'] = matches
	st.session_state['preferred_catalog'] = matches


if 'catalog_seeded' not in st.session_state:

	# Run generate_catalog once so we have something to seed the page.
	types = ['Nanodegree','Course','Free Course']
	durations = ['months','weeks','days','hours','minutes']
	difficulties = ['Advanced','Intermediate','Beginner','Fluency','Discovery']

	generate_catalog(types, durations, difficulties)

	# Prevent it from re-running.
	st.session_state['catalog_seeded'] = True

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
		preferred_catalog_keys = [program.program_key for program in preferred_catalog]
		preferred_catalog = catalog.fetch_catalog(keys=preferred_catalog_keys)

	# Overwrite the preferred catalog session state variable with OpenAI results.
	st.session_state['preferred_catalog'] = pd.DataFrame([program for program in preferred_catalog]) # dataframe result for rendering the page.
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

	st.write('First, select the types of programs you want in your plan.')

	types = ['Nanodegree','Course','Free Course']
	durations = ['months','weeks','days','hours','minutes']
	difficulties = ['Advanced','Intermediate','Beginner','Fluency','Discovery']

	# Multiselect filters.
	type_selections = st.multiselect(
			"Select Types",
			types,
			types
		)

	duration_selections = st.multiselect(
			"Select Durations",
			durations,durations
		)

	difficulty_selections = st.multiselect(
			"Select Difficulties",
			difficulties,
			difficulties
		)

	# Chat interface.
	user_input = st.text_area(
			'Which courses do you want to include in your Learning Plan?', 
			value='I\'d like for these courses to be filtered for 10-15 courses that are the best fits for solving the Python skills I care about, as well as a handful of SQL skills-focused courses.', 
			key='input_val'
		)

	if st.button('Generate Catalog'):

		if 'catalog_generated' not in st.session_state:
				st.session_state['catalog_generated'] = True

		# Update the catalog first.
		generate_catalog(type_selections,duration_selections,difficulty_selections)

		# Process any OpenAI querying on top of the catalog.
		process_submission()

		# Before a user clicks submit, allow them to see the catalog.
		catalog.showCourses(st.session_state['preferred_catalog'], num_columns=3)

			# Allow the user to download a CSV of the preferred courses.
		st.download_button(
				"Download Preferred Courses",
				st.session_state['preferred_catalog'].to_csv(index=False).encode("utf-8"),
				"preferred_courses.csv",
				"text/csv",
				key='preferred-courses-csv'
				)

	else:
		# If someone revisits the page, they see the preferred catalog from their last interaction. If there is not last interaction,
		# it show a catalog only if they generated one from the filters. Othrewise, nothing is shown.

		if 'catalog_generated' in st.session_state:

			catalog.showCourses(st.session_state['preferred_catalog'], num_columns=3)

			# Allow the user to download a CSV of the preferred courses.
			st.download_button(
				"Download Preferred Courses",
				st.session_state['preferred_catalog'].to_csv(index=False).encode("utf-8"),
				"preferred_courses.csv",
				"text/csv",
				key='preferred-courses-csv'
				)