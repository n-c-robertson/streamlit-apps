import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Assessment Creator! 👋")

st.markdown(
    """
    This is a short-term prototype created by the Skills product delivery team to facilitate the generation, review, and creation of Assessments and their Questions in in the assessments api (Udacity Assessments).

    - Tab 1: Generate Assessments based on program keys from the Udacity catalog.
    - Tab 2: A human in the loop workflow for reviewing questions.
    - Tab 3: Uploading your reviewed question set to Udacity and creating your assessments.

    A staff password is required to run Tab 1 and 3 since execute read and create tasks against Udacity's services. If you do not have the staff password, reach out to Nathan Robertson.

    **Udacity staff JWT**: Tabs 1, 3, 4, and 5 also require your own Udacity staff JWT, which you paste into the sidebar on first use. The JWT is stored only in your browser session (Streamlit session state) — it is never written to the app's secrets or disk — so you'll need to re-enter it when it expires or after starting a new session.
    """
   )

st.markdown(" ### Video Demo")

st.video('https://www.youtube.com/watch?v=lHf9TINm4E4')
