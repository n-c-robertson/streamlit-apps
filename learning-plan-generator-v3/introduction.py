# Import packages.
import streamlit as st

# Read in supporting functions from the same directory.
import page_layout

# Page title.
st.title('How to Get Started')

# Generate markdown of instructions (defined in utils).
st.markdown("Use the navigation on the left to create your Learning Plan.")
page_layout.generateInstructions()