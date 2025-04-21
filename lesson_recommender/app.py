# Import necessary packages.
import pandas as pd
import streamlit as st
import utils
import keyword_search
import embedding_search
import embedding_plus_skills

st.set_page_config(layout='wide')

# fetch data.
df = utils.fetch_data()

# Title text.
st.header('Prototype: Lesson Search')
st.markdown(f'_Lessons indexed: {len(df)}_')
st.write('A prototype of lesson discovery search engine that uses lesson title, lesson summary, course title, course summary, program title, and program summary to index results. It is biased toward matching key words against lesson metadata first, then course, then program.')

# Split two columns.
col1, col2 = st.columns(2)

# Text input for search.
with col1:
	text_search = st.text_input(f"Search for lessons by keyword search", value="")

# Filter to simulate filtering content from Workera Assessment.
with col2:
	workera_assessment_score = st.radio(f"Workera Assessment results", ["Didn't take an assessment","Beginner (0-100)", "Developing (101-200)","Accomplished (201-300)"])

# Different search options.
st.divider()

col1, col2, col3 = st.columns(3)

search = st.button('search')

if search:

    with col1:
        st.markdown('### Method 1')
        df_search = df.copy()
        df_search['combined_text'] = df_search.apply(embedding_plus_skills.combine_lesson_text, axis=1)
        df_search = embedding_plus_skills.search_lessons(text_search, df_search)
        df_search = utils.apply_difficulty_filter(df_search, workera_assessment_score)[:100]
        utils.display_results(df_search)
        
    with col2:
        st.markdown('### Method 2')
        df_search = df.copy()
        df_search['combined_text'] = df_search.apply(embedding_search.combine_lesson_text, axis=1)
        df_search = embedding_search.search_lessons(text_search, df_search)
        df_search = utils.apply_difficulty_filter(df_search, workera_assessment_score)[:100]
        utils.display_results(df_search)

    with col3:
        st.markdown('### Method 3')
        df_search = df.copy()
        df_search = keyword_search.filter_dataframe(df_search, text_search)
        df_search = utils.apply_difficulty_filter(df_search, workera_assessment_score)[:100]
        utils.display_results(df_search)

