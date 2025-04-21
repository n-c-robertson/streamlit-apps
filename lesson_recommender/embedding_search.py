import streamlit as st
import requests
import pandas as pd
from openai import OpenAI
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai_key 
import ast
import utils


# Initiate OpenAI client.
client = OpenAI(
    api_key = openai_key.key
)

def combine_lesson_text(row):
    fields = [
        f"Lesson Title: {row['lesson_title']}",
        f"Lesson Summary: {row['lesson_summary']}",
        f"Part Title: {row['part_title']}",
        f"Part Summary: {row['part_summary']}",
        f"Program Title: {row['program_title']}",
        f"School: {row['primary_school']}",
        f"Level: {row['difficulty_level']}",
        f"Duration: {row['duration']}"
    ]
    return "\n".join([str(f) for f in fields if pd.notna(f)])

#df['combined_text'] = df.apply(combine_lesson_text, axis=1)


def search_lessons(query, df, client=client, top_k=10000):

    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    query_embedding = response.data[0].embedding

    #parquet_df = pd.read_parquet('lesson_embeddings_openai.parquet')
    parquet_df = pd.read_parquet('https://raw.githubusercontent.com/n-c-robertson/streamlit-apps/refs/heads/main/lesson_recommender/lesson_embeddings_openai.parquet')
    lesson_embeddings = np.array(parquet_df['openai_embedding'].tolist())

    similarities = cosine_similarity([query_embedding], lesson_embeddings)[0]

    df_temp = df.copy()
    df_temp['similarity'] = similarities

    df_temp = df_temp.sort_values(by='similarity', ascending=False)

    # Drop duplicates by lesson title (keep the most relevant version)
    df_temp = df_temp.drop_duplicates(subset='lesson_title', keep='first')

    # Return top results
    return df_temp.head(top_k)
