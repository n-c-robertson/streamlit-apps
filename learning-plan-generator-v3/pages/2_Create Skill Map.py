# Import Packages.
import streamlit as st
import pandas as pd

# Read in supporting functions from the same directory.
import taxonomy 
import data_models
import chat_agent
import page_layout
import file_management

# Load page layout
page_layout.streamlit_page_layout()

# add `generate_skills_prompt()` to the chat agent's context if this is the
# first time a user has visited this page.
if 'generate_skills_prompt_already_ran' not in st.session_state:
	chat_agent.update_conversation(chat_agent.generate_skills_prompt(), role='system')
	st.session_state['generate_skills_prompt_already_ran'] = True

def process_submission():
	"""
	Identify the skills required to supporting the Learning Goals.

		Args:
			None

		Returns:
			Returns a sunburst visualization of relevant domain > subject > skills 
			to the Learning Goals, and saves the data in the session state variable 
			`skills`.
	"""

	# Which domains seem relevant?
	with st.status("Identifying Domains..."):
		domain_response = taxonomy.identify_domains(user_input)

	# For the selected domains, which subjects seem relevant?
	with st.status("Selecting Subjects..."):
		subject_response = taxonomy.select_subjects(user_input, domain_response)

	# For the selected subjects, which skills seem relevant?
	with st.status("Curating Skills..."):
		skill_response = taxonomy.curate_skills(user_input, domain_response, subject_response)

	# Prune the tree so that we only include domains and subjects where relevant skills were found.
	# This is necessary because the top down approach of this curation process might identify a domain
	# that sounds relevant, but then none of the subjects or skills underneath it match. This final check
	# Goes bottom up and says "if there are any domains or subjects where I didn't find skills, I'll remove
	# those from the final dataset.."
	with st.status("Pruning Results..."):
		curated_taxonomy = taxonomy.prune_taxonomy(domain_response, subject_response, skill_response, taxonomy.taxonomy)

	# Rebuild the taxonomy and update session state
	st.session_state['curated_taxonomy'] = curated_taxonomy

	# Display success message to the user.
	#page_layout.display_sidebar()
	st.success('When ready, proceed to the next step.', icon="✅")
	st.toast(page_layout.random_toast_message())

	# Flatten the taxonomy results into a table view.
	taxonomy_df = taxonomy.generate_sunburst_tableview(curated_taxonomy)

	# Save the results.
	st.session_state['skills'] = file_management.df_to_string_csv(taxonomy_df)  # string result for OpenAI in the final prompt.
	st.session_state['taxonomy_df'] = taxonomy_df  # dataframe result for rendering the page.

	# Generate and display the sunburst chart
	taxonomy.generate_sunburst(st.session_state['curated_taxonomy'])

# Page title.
st.title('Create Skills Map')

# If the user hasn't completed the previous pages, we don't want them to do anything here yet.
if st.session_state['requirements'] == 'Not Defined':
	st.error("Oops! Looks like you might not have gone through all the steps. To unlock this page, make sure you go through each of the previous steps on the other tabs.",icon='🤖')

else:

	# Chat interface.
	user_input = st.text_area(
		'Chat Interface', 
		value="My Analysts need to learn Python and SQL. More specifically, they need experience in standard Python libraries for data science like pandas, numpy, and matplotlib. SQL, they really just need to know how to query databases and do basic ETLs. Aspirationally, we\'d love to also see some of them get exposure to machine learning, but that is a stretch goal.", 
		key='input_val', 
	)

	# Trigger the processing when the 'submit' button is pressed.
	if st.button('submit'):
		process_submission()

		# Allow the user to download a CSV of the latest version of the taxonomy.
		st.download_button(
			   "Download Skills CSV",
			   st.session_state['taxonomy_df'].to_csv(index=False).encode("utf-8"),
			   "taxonomy.csv",
			   "text/csv",
			   key='taxonomy-csv'
			)

	else:
		# If the user has not yet clicked submit, check to see if the user already has a saved state. If yes, show the sunburst chart.
		if not st.session_state['taxonomy_df'].empty:
			taxonomy.generate_sunburst(st.session_state['curated_taxonomy'])

			# Allow the user to download a CSV of the latest version of the taxonomy.
			st.download_button(
			   "Download Skills CSV",
			   st.session_state['taxonomy_df'].to_csv(index=False).encode("utf-8"),
			   "taxonomy.csv",
			   "text/csv",
			   key='taxonomy-csv'
			)
