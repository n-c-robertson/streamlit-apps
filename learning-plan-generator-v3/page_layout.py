# Import packages.
import streamlit as st
import random
import pandas as pd

def initialize_session_state():

	"""
	This prototype relies heavily on streamlit's concept of session state variables. As different
	actions are taken throughout the prototype, certain variables are saved in the session. The presence
	of these variables and their values determine the user's experience.

	This function handles setting up the session state.

		Args:
			None.

		Returns:
			All of the specified session state variables that are in the function are set up when
			the user starts their session.
	"""

	# Some of the core states we will use. Start them off as "Not Defined""
	# so we can keep track of which ones haven't been defined yet (this will)
	# happen when a user action overwrites this state).
	for key in ['requirements', 'skills', 'courses']:
		if key not in st.session_state:
			st.session_state[key] = 'Not Defined'

	# Set up some other session states that are used at various points of the
	# user journey.
	if 'conversation' not in st.session_state:
		st.session_state['conversation'] = []

	if 'skills' not in st.session_state:
		st.session_state.setdefault('skills', "")
	
	if 'taxonomy_df' not in st.session_state:
		st.session_state.setdefault('taxonomy_df', pd.DataFrame())


def display_sidebar(full=False):

	"""
	Displayers a sidebar with information about the context that has been gathered so far by
	the application.

		Args:
			full: If False, then don't show the sidbar.

		Returns:
			A sidebar that shows the context that has been built so far, with a ✅ if the context
			is there, and a ⚠️ if the context is not there yet.
	"""

	with st.sidebar:
		# Create a single `st.empty()` container for the entire sidebar as a baseline.
		sidebar_container = st.empty()
		# Grab the sidebar container.
		with sidebar_container.container():
			# If it is displayed:
			if full==True:
				st.write('Checking context...')
			# Loop through each key and display a message based on session state
			for key, label in zip(['requirements', 'skills', 'courses'],['Learning Goals','Skills Map','Courses']):
				if st.session_state.get(key) != 'Not Defined':
					st.success(f'Context loaded for {label}.', icon="✅")
				else:
					# if you full is set to False, you don't have to do anything.
					if full==False:
						pass
					# If full is set to True, then throw a warning.
					elif full==True:
						st.warning(f'No context for {label}', icon="⚠️")

def streamlit_page_layout():

	"""
	Main function for defining page layout.

		Args:
			None
		
		Returns: 
			Sets some basic HTML on the webpage, and then initializes the 
			session state.
	"""

	# Set some markdown rules for how to handle badges.
	st.markdown("""
	<style>
			.badge {
			display: inline-block;
			padding: .25em .4em;
			font-size: 75%;
			font-weight: 700;
			line-height: 1;
			text-align: center;
			white-space: nowrap;
			vertical-align: baseline;
			border-radius: .25rem;
			transition: color .15s ease-in-out,background-color .15s ease-in-out,border-color .15s ease-in-out,box-shadow .15s ease-in-out;
		}
			.badge-primary { background-color: #142580; color: #fff; }
			.badge-secondary { background-color: #808080; color: #fff; }
			</style>
	""", unsafe_allow_html=True)


	# Initialize session state
	initialize_session_state()


def update_session_state_on_input(target_key, input_key='input_val'):
	"""
	Update the the variable for a session based on a text input.

		Args:
			target_key: the key to change.
			input_key: the value to change (it is going to come from a text field in streamlit).

		Returns:
			Updates the session state variable.
	"""
	st.session_state[target_key] = st.session_state[input_key]

def random_toast_message():

	"""
	Create a bank of randomized toast messages that can use for different
	successful states.

		Args:
			None

		Returns:
			a randomly selected value from the list of messages in the function.
	"""

	messages = [
	"Integrating this information into your Learning Plan...",
	"Incorporating this insight into your Learning Plan...",
	"Adding this context to your Learning Plan...",
	"Storing this detail in your Learning Plan...",
	"Saving this knowledge for your Learning Plan...",
	"Recording this information for your Learning Plan...",
	"Including this insight in your Learning Plan...",
	"Capturing this context for your Learning Plan...",]

	return random.choice(messages)

def generateInstructions():

	"""
	Set the HTML used for the introduction.py page of the application.

		Args:
			None

		Returns:
			HTML renders for the instructions on introduction.py.
	"""


	st.markdown("""
	<style>
		.table-container {
			width: 100%;
			border-collapse: collapse;
		}
		.table-row {
			display: flex;
			padding: 10px 0;
			border-bottom: 1px solid #ddd;
		}
		.table-left {
			width: 40%;
			font-weight: bold;
			font-size: 1.1em;
			padding-right: 15px;
		}
		.table-right {
			width: 60%;
		}
	</style>
	
	<div class="table-container">
		<div class="table-row">
			<div class="table-left">🎯 Define Learning Goals</div>
			<div class="table-right">Provide text or upload documents that describe the learning goals of the organization. You will receive in return a table of goals synthesized from the input.</div>
		</div>
		<div class="table-row">
			<div class="table-left">🗺️ Create Skill Map</div>
			<div class="table-right">Provide some description about the types of skills or competencies that matter to you. You will receive a graphic representation of the skills, as well as the opportunity to remove skills you think are not relevant.</div>
		</div>
		<div class="table-row">
			<div class="table-left">🗂️ Curate Courses</div>
			<div class="table-right">You will receive a list of 50 courses that could be good candidates. You can provide instructions to filter this list down to a preferred list of courses you want.</div>
		</div>
		<div class="table-row">
			<div class="table-left">🎉 Generate Plans</div>
			<div class="table-right">You provide what kind of Learning Plan you want. You will receive Learning Plans that reflect all of the context you gave us throughout the process.</div>
		</div>
	</div>
""", unsafe_allow_html=True)
