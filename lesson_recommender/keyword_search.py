# Import necessary packages.
import pandas as pd
import streamlit as st

"""
Series of functions to lazily search metadata based on input terms.
"""

def count_matches(text, words):
    """Counts the number of search words appearing in a given text."""
    if pd.isna(text):
        return 0
    return sum(word in text.lower() for word in words)

def filter_dataframe(df, text_search):
    """Filters the DataFrame based on search input and assigns ranking scores."""
    if not text_search:
        return df.copy()
    
    search_words = text_search.lower().split()
    match_columns = {
        'program_match_count': 'program_title',
        'course_title_match_count': 'part_title',
        'course_summary_match_count': 'part_summary',
        'lesson_title_match_count': 'lesson_title',
        'lesson_summary_match_count': 'lesson_summary',
    }
    
    for col, source in match_columns.items():
        df[col] = df[source].apply(lambda x: count_matches(x, search_words))
    
    df_search = df[df[list(match_columns.keys())].sum(axis=1) > 0]
    
    # Higher weighting for lesson title and lesson summary.
    df_search['rank_score'] = (
        df_search['lesson_title_match_count'] * 5 +
        df_search['lesson_summary_match_count'] * 4 +
        df_search['course_title_match_count'] * 3 +
        df_search['course_summary_match_count'] * 2 +
        df_search['program_match_count'] * 1
    )
    
    df_search['match_count'] = df_search[list(match_columns.keys())].sum(axis=1)
    
    return df_search.sort_values(by=['rank_score', 'match_count'], ascending=[False, False])