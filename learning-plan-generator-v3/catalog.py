# Import packages.
import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from jinja2 import Template
import humanize

@st.cache_data
def fetch_catalog(keys=None):

	"""
	Fetch the unified catalog.

		Args:
			keys: a list of program keys to pass to the catalog.

		Returns:
			List of dictionaries where each dictionary is a piece of content in the public catalog.
	"""

	# Fetch public offerings in unified catalog (public offerings as defined by is_offered_to_public==True).
	catalog_url = 'https://api.udacity.com/api/unified-catalog/search'
	catalog_data = requests.post(catalog_url, json={'pageSize': 1000, 'SortBy': 'avgRating'}).json()
	catalog_results = [r for r in catalog_data['searchResult']['hits'] if r['is_offered_to_public']]

	def convert_duration(mins):

		"""
		Convert a minute-based integer time value into a more human readable representation.

			Args:
				mins: the number of minutes long a content offering is.

			Returns:
				A human-readable time value rounded to the nearest whole number. For example,
				if the input is 44,000, that will first be translated into months, which will be 
				1.1(...) months. this will then be rounded to 1, and return as a string "1 months".
		"""

		if mins >= 43200:  # 28 days or more
			months = mins / 43200  # Approximate minutes in a month (30 days)
			rounded_value = round(months)
			unit = 'month' if rounded_value == 1 else 'months'
			return f"{rounded_value} {unit}"

		elif mins >= 10080: # 7 days or more
			weeks = mins / 10080
			rounded_value = round(weeks)
			unit = 'week' if rounded_value == 1 else 'weeks'
			return f"{rounded_value} {unit}"

		elif mins >= 1440:  # 1 day or more
			days = mins / 1440  # Minutes in a day
			rounded_value = round(days)
			unit = 'day' if rounded_value == 1 else 'days'
			return f"{rounded_value} {unit}"

		elif mins >= 60:  # 1 hour or more
			hours = mins / 60  # Minutes in an hour
			rounded_value = round(hours)
			unit = 'hour' if rounded_value == 1 else 'hours'
			return f"{rounded_value} {unit}"

		else:  # Less than 1 hour, use minutes
			rounded_value = round(mins)
			unit = 'minute' if rounded_value == 1 else 'minutes'
			return f"{rounded_value} {unit}"

	# Lambda function for translating the program type into something more intuitive.
	convert_program_type = lambda t: 'Free Course' if t == 'Course' else 'Nanodegree' if t == 'Degree' else 'Course' if t == 'Part' else 'Unknown'
	
	# Lamba function for creating the URL to the public catalog offering.
	convert_slug_to_url = lambda slug: f"https://www.udacity.com/course/{slug}"

	# Process catalog results with helper functions.
	programs = [
		{
			'program_key': result['key'],
			'program_type': convert_program_type(result['semantic_type']),
			'catalog_url': convert_slug_to_url(result['slug']),
			'duration': convert_duration(result['duration']),
			'difficulty': result['difficulty'],
			'title': result['title'],
			'summary': result['summary'],
			'skill_names': result['skill_names']
		} for result in catalog_results
	]

	# If we are fetching metadata for a specific set of course keys, filter those here.
	if keys != None:
		programs = [program for program in programs if program['program_key'] in keys]

	# Return programs.
	return programs

# Read in programs as part of memory.
programs = fetch_catalog()

# Also store a pandas DataFrame representation of the value.
programs_df = pd.DataFrame(programs)

def retrieve_matching_courses(query, type_selections, duration_selections, difficulty_selections, top_n=50, programs=programs):

	"""
	Using term-frequency-inverse-document-frequency to return the top `n` programs based on the
	course title, summary, and skills.

		Args:
			query: The search query that the user runs. In the context of this application, this will 
			be the user's context from their OpenAI conversation.
			type_selections: the program types to include.
			duration_selections: the program durations to include.
			difficulty_selections: the difficulty seleections to include.
			top_n: the number of results to return (default 50).
			programs: The programs data returned by `fetch_catalog`.

		Returns:
			A pandas DataFrame of the top programs.
	"""

	# Create a copy of the catalog that can be filtered.
	filtered_programs_ = programs.copy()

	# Filter that catalog based on the requirements passed through.
	filtered_programs_ = [program for program in filtered_programs_ if program['program_type'] in type_selections]
	filtered_programs_ = [program for program in filtered_programs_ if program['difficulty'] in difficulty_selections]

	# Slightly more wordy handling of durations.
	filtered_programs = []

	for program in filtered_programs_:
		for duration in duration_selections:
			if duration[:-1] in program['duration']:
				filtered_programs.append(program)
				continue
			continue

	print(type_selections)
	print(duration_selections)
	print(difficulty_selections)
	print(len(programs))
	print(len(filtered_programs))



	# Create a list of all course skills
	summaries = [course['title'] + ' ' + course['summary'] + ' ' + ' '.join(course['skill_names']) for course in filtered_programs]
	course_titles = [course['title'] for course in filtered_programs]

	# Use TF-IDF vectorizer to vectorize query and course summaries
	vectorizer = TfidfVectorizer(stop_words='english')
	vectors = vectorizer.fit_transform([query] + summaries)
	
	# Compute cosine similarity between the query and all course summaries
	cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
	
	# Get top N matching courses
	matching_indices = np.argsort(cosine_similarities)[::-1][:top_n]
	
	titles = [course_titles[i] for i in matching_indices]

	return pd.DataFrame([program for program in filtered_programs if program['title'] in titles])


def generateCourseCard(program):

	"""
	Create a course card for a program.

		Args:
			program: an enry in the programs DataFrame.

		Returns:
			HTML rendering of a program card that can be used in a variet of contexts.
	"""

	template = Template("""
	<style>
		.card {
			border: 1px solid #ddd;
			border-radius: 10px;
			padding: 15px;
			margin-bottom: 20px;
			font-family: Arial, sans-serif;
		}
		.card h3 {
			margin-bottom: 10px;
		}
		.card p {
			margin: 5px 0;
		}
		.card a {
			color: #007bff;
			text-decoration: none;
		}
		.card a:hover {
			text-decoration: underline;
		}
		.chips {
			display: flex;
			flex-wrap: wrap;
			gap: 8px;
			margin-top: 10px;
		}
		.chip-primary {
			background-color: #142580;
			color: #FFF;
			border-radius: 16px;
			padding: 4px 12px;
			font-size: 14px;
		}
		.chip-secondary {
			background-color: #DFDFDF;
			border-radius: 16px;
			padding: 4px 12px;
			font-size: 14px;
		}

	</style>
	<div class="card">
		<h3>{{ title }}</h3>
		<div class="chips">
			<span class="chip-primary">{{ duration }}</span>
			<span class="chip-primary">{{ difficulty }}</span>
			<span class="chip-primary">{{ program_type }}</span>
		</div>
		<p>{{ summary }}</p>
		<a href="{{ catalog_url }}" target="_blank">Learn More</a>
	</div>
	""")

	return template.render(
		title=program['title'],
		duration=program['duration'],
		difficulty=program['difficulty'],
		summary=' '.join(program['summary'].split()[:15])+'...',
		skill_names=' '.join(program['skill_names']),
		catalog_url=program['catalog_url'],
		program_type=program['program_type']
	)

def showCourses(matches, num_columns=1):

	"""
	A wrapper functinon that can creates rows of content from the 
	output of the `generateCourseCard` function.

		Args:
			matches: A filtered DataFrame of programs.
			num_columns: the number of columns for each row.

		Returns:
			An HTML rendering of n program cards per row, where n = num_columns.
	"""

	# Display the courses in rows with three columns
	columns = st.columns(num_columns)

	for index, row in matches.iterrows():
		# Calculate which column to use based on the index
		col = columns[index % num_columns]
		
		with col:
			st.markdown(generateCourseCard(row), unsafe_allow_html=True)