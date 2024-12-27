# Import packages.
import streamlit as st

# API Keys.
def jwt_token():
	jwt_token = st.secrets['jwt_token']
	return jwt_token

def openai_key():
	openai_key = st.secrets['openai_key']
	return openai_key