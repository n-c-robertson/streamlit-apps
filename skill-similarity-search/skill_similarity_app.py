import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import re
from typing import List, Dict, Tuple

# Set page config
st.set_page_config(
    page_title="Skill Similarity Search",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load and preprocess the taxonomy and subskills data from Google Sheets"""
    try:
        # Get URLs from Streamlit secrets
        taxonomy_url = st.secrets["TAXONOMY_URL"]
        subskills_url = st.secrets["SUBSKILLS_URL"]
        
        # Try using requests first (better SSL handling)
        try:
            import requests
            import io
            
            # Load taxonomy data from Google Sheets using requests
            response = requests.get(taxonomy_url, verify=False)  # Disable SSL verification for now
            response.raise_for_status()
            taxonomy_df = pd.read_csv(io.StringIO(response.text))
            
            # Load subskills data from Google Sheets using requests
            response = requests.get(subskills_url, verify=False)  # Disable SSL verification for now
            response.raise_for_status()
            subskills_df = pd.read_csv(io.StringIO(response.text))
            
        except ImportError:
            # Fallback to pandas with SSL context if requests not available
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Load taxonomy data from Google Sheets
            taxonomy_df = pd.read_csv(taxonomy_url)
            
            # Load subskills data from Google Sheets
            subskills_df = pd.read_csv(subskills_url)
        
        # Clean and preprocess taxonomy data
        taxonomy_df = taxonomy_df.fillna('')
        
        # Handle hierarchical structure - forward fill Domain and Subject values
        # This handles the case where Domain and Subject are only specified once
        # and subsequent rows inherit these values until a new one is specified
        current_domain = ''
        current_subject = ''
        
        for idx, row in taxonomy_df.iterrows():
            # Update current domain if this row has a domain
            if row['Domain'].strip():
                current_domain = row['Domain'].strip()
            
            # Update current subject if this row has a subject
            if row['Subject'].strip():
                current_subject = row['Subject'].strip()
            
            # If this row has a skill but no domain/subject, inherit current ones
            if row['Udaciskill'].strip():
                taxonomy_df.at[idx, 'Domain'] = current_domain
                taxonomy_df.at[idx, 'Subject'] = current_subject
        
        # Create a combined text field for each skill for better similarity matching
        # Focus on skill name and definition for more targeted results
        taxonomy_df['combined_text'] = (
            taxonomy_df['Udaciskill'].astype(str) + ' ' +
            taxonomy_df['Definition [NOT FOR EXTERNAL USE]'].astype(str)
        ).str.strip()
        
        # Filter out rows where Udaciskill is empty (these are category headers)
        skills_df = taxonomy_df[taxonomy_df['Udaciskill'] != ''].copy()
        
        # Clean subskills data
        subskills_df = subskills_df.fillna('')
        subskills_df = subskills_df[subskills_df['Skill'] != '']
        
        return skills_df, subskills_df
        
    except Exception as e:
        st.error(f"Error loading data from Google Sheets: {str(e)}")
        st.error("Please check your internet connection and that the Google Sheets are publicly accessible.")
        st.info("Troubleshooting Tips:\n"
                "- Ensure the Google Sheets are set to 'Anyone with the link can view'\n"
                "- Check if you can access the sheets directly in your browser\n"
                "- Try refreshing the page if this is a temporary network issue")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data
def create_similarity_matrix(skills_df):
    """Create TF-IDF vectors and similarity matrix"""
    if skills_df.empty:
        return None, None
    
    # Use combined text for vectorization
    texts = skills_df['combined_text'].tolist()
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.95
    )
    
    # Fit and transform the texts
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    return vectorizer, tfidf_matrix

def find_similar_skills(query: str, skills_df: pd.DataFrame, vectorizer, tfidf_matrix, top_k: int = 10):
    """Find skills similar to the query using cosine similarity"""
    if vectorizer is None or tfidf_matrix is None:
        return []
    
    # Transform the query
    query_vector = vectorizer.transform([query])
    
    # Calculate cosine similarities
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get top k similar skills
    top_indices = similarities.argsort()[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0:  # Only include skills with some similarity
            skill_info = skills_df.iloc[idx]
            results.append({
                'skill': skill_info['Udaciskill'],
                'domain': skill_info['Domain'],
                'subject': skill_info['Subject'],
                'synonyms': skill_info['Synonyms'],
                'definition': skill_info['Definition [NOT FOR EXTERNAL USE]'],
                'similarity': similarities[idx],
                'combined_text': skill_info['combined_text']
            })
    
    return results

def get_subskills(skill_name: str, subskills_df: pd.DataFrame) -> List[str]:
    """Get subskills for a given skill"""
    if subskills_df.empty:
        return []
    
    # Find exact matches first
    exact_matches = subskills_df[subskills_df['Skill'].str.strip() == skill_name.strip()]
    if not exact_matches.empty:
        subskills = exact_matches['Subskill'].dropna().tolist()
        return [sub for sub in subskills if sub.strip()]
    
    # If no exact match, try partial matching
    partial_matches = subskills_df[subskills_df['Skill'].str.contains(skill_name, case=False, na=False)]
    if not partial_matches.empty:
        subskills = partial_matches['Subskill'].dropna().tolist()
        return [sub for sub in subskills if sub.strip()]
    
    return []

def create_comprehensive_taxonomy_graph(skills_df: pd.DataFrame, results: List[Dict], query: str):
    """Create a treemap visualization showing Domain → Subject → Skills hierarchy"""
    if not results:
        return None
    
    # Prepare data for treemap
    treemap_data = []
    
    # Group results by domain and subject
    hierarchy = {}
    domain_colors = {}
    # Use a darker color palette that works well with white text
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                    '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173']
    
    for result in results:
        domain = result['domain']
        subject = result['subject']
        skill = result['skill']
        similarity = result['similarity']
        
        # Assign color to domain if not already assigned
        if domain not in domain_colors:
            color_idx = len(domain_colors) % len(color_palette)
            domain_colors[domain] = color_palette[color_idx]
        
        if domain not in hierarchy:
            hierarchy[domain] = {}
        if subject not in hierarchy[domain]:
            hierarchy[domain][subject] = []
        
        hierarchy[domain][subject].append({
            'skill': skill,
            'similarity': similarity,
            'definition': result.get('definition', '')
        })
    
    # Build treemap data structure with colors
    colors = []
    
    for domain, subjects in hierarchy.items():
        # Add domain entry - always black
        domain_total_similarity = sum(
            skill_info['similarity'] 
            for subject_skills in subjects.values() 
            for skill_info in subject_skills
        )
        
        treemap_data.append({
            'ids': f"domain_{domain}",
            'labels': domain,
            'parents': "",
            'values': domain_total_similarity,
            'level': 'domain'
        })
        colors.append('black')
        
        for subject, skills in subjects.items():
            # Add subject entry - always black
            subject_total_similarity = sum(skill_info['similarity'] for skill_info in skills)
            subject_id = f"subject_{domain}_{subject}"
            
            treemap_data.append({
                'ids': subject_id,
                'labels': subject,
                'parents': f"domain_{domain}",
                'values': subject_total_similarity,
                'level': 'subject'
            })
            colors.append('black')
            
            # Add skill entries - use domain color
            for skill_info in skills:
                skill_id = f"skill_{skill_info['skill']}"
                skill_color = domain_colors[domain]
                
                treemap_data.append({
                    'ids': skill_id,
                    'labels': skill_info['skill'],
                    'parents': subject_id,
                    'values': skill_info['similarity'],
                    'level': 'skill',
                    'similarity': skill_info['similarity'],
                    'definition': skill_info['definition']
                })
                colors.append(skill_color)
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(treemap_data)
    
    # Create the treemap
    fig = go.Figure(go.Treemap(
        ids=df['ids'],
        labels=df['labels'],
        parents=df['parents'],
        values=df['values'],
        branchvalues="total",
        maxdepth=3,
        
        # Use custom colors: black for domains/subjects, domain colors for skills
        marker=dict(
            colors=colors,
            line=dict(width=2, color='white')
        ),
        
        # Custom text formatting - white text for all elements
        textinfo="label",
        texttemplate="<b>%{label}</b>",
        textfont=dict(size=14, color='white'),
        textposition="middle center",
        
        # Custom hover information
        hovertemplate='<b>%{label}</b><br>' +
                     'Value: %{value:.3f}<br>' +
                     '<extra></extra>',
        
        # Remove pathbar to simplify
        pathbar=dict(visible=False)
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Skill Taxonomy by Domain - Search Results for: '{query}'" if query else "Skill Taxonomy by Domain",
        font=dict(size=14),
        height=700,
        margin=dict(t=80, l=10, r=10, b=10)
    )
    
    return fig

def main():
    st.title("Skill Similarity Search")
    st.markdown("Find similar skills from the taxonomy using AI-powered semantic search")
    
    # Load data
    with st.spinner("Loading skill taxonomy data from Google Sheets..."):
        skills_df, subskills_df = load_data()
    
    if skills_df.empty:
        st.error("Failed to load data from Google Sheets. Please check your internet connection and verify that the Google Sheets are publicly accessible.")
        return
    
    st.success(f"Successfully loaded {len(skills_df)} skills and {len(subskills_df)} subskill relationships from Google Sheets")
    
    # Create similarity matrix
    with st.spinner("Building similarity search index..."):
        vectorizer, tfidf_matrix = create_similarity_matrix(skills_df)
    
    if vectorizer is None:
        st.error("Failed to create similarity index.")
        return
    
    # Sidebar controls
    st.sidebar.header("Search Parameters")
    top_k = st.sidebar.slider("Number of results", min_value=5, max_value=30, value=10)
    
    # Main search interface
    st.header("Search for Similar Skills")
    
    # Create a form for password-protected search
    with st.form("skill_search_form"):
        query = st.text_input(
            "Enter a skill, job competency, or knowledge component:",
            placeholder="e.g., machine learning, data analysis, project management"
        )
        password = st.text_input("Password", type="password", placeholder="Enter password to access search")
        search_submitted = st.form_submit_button("Search", type="primary")
    
    # Check password and process search
    if search_submitted:
        if password != st.secrets["PASSWORD"]:
            st.error("Incorrect password. Please enter the correct password to access the skill search.")
        elif not query.strip():
            st.warning("Please enter a search term.")
        else:
            with st.spinner("Searching for similar skills..."):
                results = find_similar_skills(query, skills_df, vectorizer, tfidf_matrix, top_k)
            
            if results:
                st.success(f"Found {len(results)} similar skills")
                
                # Display results
                st.header("Search Results")
                
                for i, result in enumerate(results, 1):
                    with st.expander(f"{i}. {result['skill']} (Similarity: {result['similarity']:.3f})", expanded=i<=3):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Taxonomical Trace:**")
                            st.markdown(f"**Domain:** {result['domain']}")
                            st.markdown(f"**Subject:** {result['subject']}")
                            st.markdown(f"**Skill:** {result['skill']}")
                            
                            if result['synonyms']:
                                st.markdown(f"**Synonyms:** {result['synonyms']}")
                            
                            if result['definition']:
                                st.markdown(f"**Definition:** {result['definition']}")
                        
                        with col2:
                            st.metric("Similarity Score", f"{result['similarity']:.3f}")
                            
                            # Always show subskills
                            subskills = get_subskills(result['skill'], subskills_df)
                            if subskills:
                                st.markdown("**Related Subskills:**")
                                for subskill in subskills[:5]:  # Show top 5 subskills
                                    st.markdown(f"• {subskill}")
                                if len(subskills) > 5:
                                    st.markdown(f"*... and {len(subskills) - 5} more*")
                
                # Always show network visualization
                st.header("Network Visualization")
                network_fig = create_comprehensive_taxonomy_graph(skills_df, results, query)
                if network_fig:
                    st.plotly_chart(network_fig, use_container_width=True)
                
            else:
                st.warning("No similar skills found. Try a different search term.")
    
if __name__ == "__main__":
    main() 