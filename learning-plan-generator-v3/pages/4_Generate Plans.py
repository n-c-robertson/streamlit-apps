# Import Packages.
import streamlit as st

# Read in supporting functions from the same directory.
import ugraphql
import data_models 
import chat_agent 
import plan_generator
import page_layout
import file_management

# add `generate_learning_plans_prompt()` to the chat agent's context if this is the
# first time a user has visited this page. We are also going to reset the chat conversation because
# we are going to ready in only the curated context that was built in the other pages -- and leave out
# any of the back and forth conversation that user had. This also helps to keep the context window smaller.
if 'initial_courses_prompt_already_ran' not in st.session_state:
	chat_agent.reset_conversation()
	chat_agent.update_conversation(chat_agent.generate_learning_plans_prompt(), role='system')
	st.session_state['initial_learning_plans_prompt_already_ran'] = True

def process_submission():

	"""
	Generates and renders Learning Plans for the user to review.
	"""

	# Get a list of raw learning plans from OpenAI.
	with st.status("Generating Learning Plans (This may take a minute!)..."):
		chat_agent.update_conversation(user_input, role='user')
		raw_learningPlans = chat_agent.chatgpt(format_=data_models.LearningPlans)

	# Display success message, set expectations that it takes a minute to load.
	st.success('We are rendering your Learning Plans now! This may take a minute. Want to start over? Refresh your browser and go back to the first tab.', icon="✅")

	# List of successfully generated learning plans.
	learningPlans = []

	# Loop through each plan.
	for idx, plan in enumerate(raw_learningPlans.plans):
		
		# Try to render the leanrning plan.
		try:
			#learningPlans.append(plan_generator.formatLearningPlan(plan))
			plan = plan_generator.formatLearningPlan(plan)		# Clean up formatting.
			plan_generator.generateLearningPlan(plan)			# Render learning plan.

			# Render the skills report as an add on to the plan - to do would be to move this into `generateLearningPlan`.
			skills_report_prompt = chat_agent.generate_skills_report_prompt(plan)
			chat_agent.update_conversation(skills_report_prompt)
			skills_report_raw = chat_agent.chatgpt(format_=data_models.SkillsReport)
			with st.expander('skills report'):
				plan_generator.generateSkillsReport(skills_report_raw)

			# append results.
			st.write(type(skills_report_raw))
			st.write(skills_report_raw)
			learningPlans.append((plan, skills_report_raw.subjects))

		# If any error is hit that isn't handled at a lower level, throw an error and say it is a hallucination.
		# There were some lower level errors that used to occur that looked like hallucinations. But those should
		# Start to go away as the error handling gets better at the lower levels.
		except Exception as e:
			# Expander saying "an error occured"
			with st.expander(f' 🤖 Hallucination was detected in Learning Plan #{idx+1}, so it is hidden.'):
				st.write(e)		# What error occurred?
				st.code(plan)	# What did the raw plan output look like?

	# Save as a session state so someone could come back to this page in the same session and see the learning plans.
	# This is not functional yet. More changes need to be made for this to work, I think.
	st.session_state['learningplans'] = learningPlans

# Page title.
st.title('Generate Plans')

# Load page layout.
page_layout.streamlit_page_layout()
page_layout.display_sidebar(full=True)

# If the user hasn't completed the previous pages, we don't want them to do anything here yet.
if st.session_state['requirements'] == 'Not Defined' or st.session_state['courses'] == 'Not Defined' or st.session_state['skills'] == 'Not Defined':
	st.error("Oops! Looks like you might not have gone through all the steps. To unlock this page, make sure you go through each of the previous steps on the other tabs.",icon='🤖')

else:
	# Chat interface.
	user_input = st.text_area(
		'Provide final instructions for generating learning plans, and we\'ll provide you with a few options.' , 
		value='Provide 3-5 learning plans. They should be roughly 6 months long each. They should be sequenced in an intuitive order where one step builds upon the next. Each plan should have at least 3 steps, with one Discovery or Fluency course at the beginning that is shorter.', 
		key='input_val'
	)

	# Trigger the processing when the 'submit' button is pressed
	if st.button('Submit'):
		process_submission()