# Import necessary packages.
import pandas as pd
import streamlit as st
import numpy as np

# Streamlit decorator. Caches the function so it doesn't rerun unless there is a change.
@st.cache_data
def fetch_data():
	#sheet_id='1qRbSyWjB86cFDsc2FacuPGK_F1BlEfVhD5rDu5w4noI'
	#sheet_name='Sheet1'
	#url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
	#df = pd.read_csv(url, dtype=str, sep=',',header=0)
	#df = pd.read_csv('lesson_catalog.csv')
	df = pd.read_csv('https://raw.githubusercontent.com/n-c-robertson/streamlit-apps/refs/heads/main/lesson_recommender/lesson_catalog.csv')
	return df

def display_results(df_search):
	"""Displays search results as cards in Streamlit."""
	
	N_cards_per_row = 1
	for n_row, row in df_search.reset_index().iterrows():
		i = n_row % N_cards_per_row
		if i == 0:
			st.write("---")
			cols = st.columns(N_cards_per_row, gap="large")
		with cols[i]:
			st.markdown(f"**{row['lesson_title']}**")

			st.markdown(row['lesson_summary'])
			st.markdown(f"_Program: {row['program_title']} > Course: {row['part_title']} > Lesson: {row['lesson_title']}_")

def apply_difficulty_filter(df_search, workera_assessment_score):
	"""Filters the DataFrame based on difficulty level."""
	difficulty_mapping = {
		"Beginner (0-100)": ['Fluency', 'Beginner', 'Discovery'],
		"Developing (101-200)": ['Intermediate'],
		"Accomplished (201-300)": ['Advanced']
	}
	
	if workera_assessment_score in difficulty_mapping:
		df_search = df_search[df_search['difficulty_level'].isin(difficulty_mapping[workera_assessment_score])]
	
	return df_search


def parse_embeddings(s):
    try:
        s = str(s).replace('\n', ' ').replace('...', '').strip()
        return np.array(ast.literal_eval(s))
    except Exception as e:
        print(f"Failed to parse: {s[:60]}...\nError: {e}")
        return np.nan
