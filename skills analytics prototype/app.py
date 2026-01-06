import data_pipeline
import streamlit as st
import settings
import pandas as pd
import altair as alt
from chat_ui import render_chat_sidebar
from agent import generate_overall_insights, generate_recommendations_insights

# Set page config for wide mode
st.set_page_config(
    page_title="Skills Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# AUTHENTICATION SIDEBAR
# =============================================================================

def render_auth_sidebar():
    """Render the authentication inputs in the sidebar."""
    with st.sidebar:
        st.markdown("### 🔐 Connect to APIs")
        
        jwt_token = st.text_input(
            "Udacity JWT Token",
            type="password",
            help="Enter your Udacity staff JWT token",
            key="jwt_input"
        )
        
        company_id = st.text_input(
            "Company ID",
            value=str(settings.default_company_id),
            help=f"Approved company IDs: {', '.join(str(id) for id in settings.approved_company_ids)}",
            key="company_id_input"
        )
        
        connect_button = st.button("🔌 Connect & Load Data", use_container_width=True, type="primary")
        
        if connect_button:
            if not jwt_token:
                st.error("Please enter a JWT token")
                return False
            if not company_id:
                st.error("Please enter a Company ID")
                return False
            
            try:
                company_id_int = int(company_id.strip())
            except ValueError:
                st.error("Company ID must be a number")
                return False
            
            # Validate company ID is in approved list
            if company_id_int not in settings.approved_company_ids:
                st.error("❌ Error: This company ID is not approved to use this application.")
                return False
            
            # Clean the JWT token
            clean_jwt = jwt_token.strip()
            clean_jwt = ''.join(c for c in clean_jwt if ord(c) < 128)
            
            with st.spinner("Fetching Workera API key..."):
                try:
                    # Fetch the Workera API key
                    workera_api_key = data_pipeline.fetch_workera_api_key(company_id_int, clean_jwt)
                    
                    # Store credentials in session state
                    st.session_state['authenticated'] = True
                    st.session_state['jwt_token'] = clean_jwt
                    st.session_state['workera_api_key'] = workera_api_key
                    st.session_state['company_id'] = company_id_int
                    
                    # Set credentials in data_pipeline
                    data_pipeline.set_credentials(clean_jwt, workera_api_key, company_id_int)
                    
                    st.success("✅ Connected successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
                    return False
        
        # Show connection status
        if st.session_state.get('authenticated'):
            st.markdown("---")
            st.success(f"✅ Connected to Company ID: {st.session_state.get('company_id')}")
            
            if st.button("🔄 Disconnect", use_container_width=True):
                # Clear session state
                for key in ['authenticated', 'jwt_token', 'workera_api_key', 'company_id']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.cache_data.clear()
                st.rerun()
        
        return st.session_state.get('authenticated', False)

# Check authentication status
if not st.session_state.get('authenticated'):
    # Show authentication UI
    render_auth_sidebar()
    
    st.title("Skills Analytics Dashboard")
    st.info("👈 Please enter your JWT token and Company ID in the sidebar to connect and load data.")
    st.stop()

# If we get here, user is authenticated - set credentials in data_pipeline
data_pipeline.set_credentials(
    st.session_state['jwt_token'],
    st.session_state['workera_api_key'],
    st.session_state['company_id']
)

# Render auth sidebar (to show connected status and disconnect button)
render_auth_sidebar()

st.title("Skills Analytics Dashboard")

# Cache data fetching functions
@st.cache_data
def load_emc_content_data():
    return data_pipeline.get_emc_content_data()

@st.cache_data
def load_assessment_data(_emc_content_data):
    return data_pipeline.get_all_assessment_attempts(_emc_content_data)

@st.cache_data
def load_skill_acquisitions(_emc_content_data, _udacity_assessment_data):
    return data_pipeline.get_all_skill_acquisitions(
        _emc_content_data, 
        _udacity_assessment_data
    )

@st.cache_data(show_spinner=False)
def load_user_progress(_emc_content_data):
    return data_pipeline.get_all_user_progress(_emc_content_data)

@st.cache_data
def load_learner_frequency(_user_progress_data):
    return data_pipeline.get_learner_frequency_by_month(_user_progress_data)

@st.cache_data
def load_combined_assessments(_udacity_assessment_data, _emc_content_data):
    return data_pipeline.get_all_assessment_attempts_combined(_udacity_assessment_data, _emc_content_data)

@st.cache_data
def load_passed_projects(_emc_content_data):
    return data_pipeline.get_passed_projects_timeseries_detailed(_emc_content_data)

@st.cache_data
def load_skills_by_enrollments(_emc_content_data):
    return data_pipeline.get_skills_by_enrollments(_emc_content_data)

@st.cache_data
def load_skills_by_graduations(_emc_content_data):
    return data_pipeline.get_skills_by_graduations(_emc_content_data)

@st.cache_data
def load_workera_data():
    return data_pipeline.fetch_workera_results()

@st.cache_data
def load_recommendations(_workera_data):
    return data_pipeline.create_recommended_lessons_dataset(_workera_data)

@st.cache_data
def process_skills_timeseries(_skill_acquisitions_df):
    """Process skills acquisition data for time series charts."""
    df = _skill_acquisitions_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['domain_name'].notna()]
    
    # Group by month and domain
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    grouped = df.groupby(['month', 'domain_name']).size().reset_index(name='count')
    
    # Create complete date range
    all_months = pd.date_range(
        start=grouped['month'].min(),
        end=grouped['month'].max(),
        freq='MS'
    )
    all_domains = grouped['domain_name'].unique()
    complete_index = pd.MultiIndex.from_product(
        [all_months, all_domains],
        names=['month', 'domain_name']
    )
    
    # Reindex to fill missing combinations
    grouped = grouped.set_index(['month', 'domain_name']).reindex(
        complete_index, 
        fill_value=0
    ).reset_index()
    
    # Calculate cumulative count
    grouped = grouped.sort_values(['domain_name', 'month'])
    grouped['cumulative_count'] = grouped.groupby('domain_name')['count'].cumsum()
    
    # Rename for consistency
    grouped.rename(columns={'month': 'date'}, inplace=True)
    grouped['date_str'] = grouped['date'].dt.strftime('%Y-%m')
    
    return grouped

@st.cache_data
def process_source_counts(_skill_acquisitions_df):
    """Process skill acquisitions by source."""
    return _skill_acquisitions_df.groupby('source').size().reset_index(name='count')

@st.cache_data
def process_assessment_performance(_combined_assessments, assessment_type):
    """Process assessment performance data for a specific assessment type."""
    filtered = _combined_assessments[
        _combined_assessments['assessmentSource'] == assessment_type
    ].copy()
    
    if len(filtered) == 0:
        return pd.DataFrame()
    
    # Calculate average score by assessment (only for assessments with scores)
    avg_scores = filtered[filtered['score'] > 0].groupby('assessment_name').agg({
        'score': 'mean',
        'email': 'count'
    }).reset_index()
    avg_scores.columns = ['assessment_name', 'avg_score', 'num_attempts']
    avg_scores = avg_scores.sort_values('avg_score', ascending=False)
    avg_scores['avg_score'] = avg_scores['avg_score'].round(1)
    
    return avg_scores

@st.cache_data
def process_projects_data(_passed_projects):
    """Process passed projects data."""
    project_counts = _passed_projects.groupby('projectName')['passed_projects'].sum().reset_index()
    return project_counts.sort_values('passed_projects', ascending=False).head(15)

@st.cache_data
def process_enrollment_skills(_enrollment_skills):
    """Process enrollment skills by skill name."""
    df = _enrollment_skills.copy()
    # Group by skill name and count occurrences
    skill_counts = df.groupby('skill').size().reset_index(name='skill_count')
    # Sort by count descending and take top 20
    skill_counts = skill_counts.sort_values('skill_count', ascending=False).head(20)
    return skill_counts

@st.cache_data
def process_graduation_skills(_graduation_skills):
    """Process graduation skills by skill name."""
    df = _graduation_skills.copy()
    # Group by skill name and count occurrences
    skill_counts = df.groupby('skill').size().reset_index(name='skill_count')
    # Sort by count descending and take top 20
    skill_counts = skill_counts.sort_values('skill_count', ascending=False).head(20)
    return skill_counts

@st.cache_data
def process_frequency_data(_frequency_df):
    """Process learning frequency data."""
    df = _frequency_df.copy()
    df['month_str'] = df['month'].dt.strftime('%Y-%m')
    return df

def format_duration(minutes):
    """Convert minutes to human-readable format."""
    if pd.isna(minutes):
        return 'N/A'
    
    minutes = int(minutes)
    
    if minutes <= 60:
        return f"{minutes} minutes"
    
    hours = minutes // 60
    remaining_minutes = minutes % 60
    
    if remaining_minutes == 0:
        return f"{hours} hours"
    else:
        return f"{hours} hours and {remaining_minutes} minutes"

@st.cache_data
def process_recommendations(_recommendations_df):
    """Process recommendations to make durations human readable."""
    df = _recommendations_df.copy()
    
    # Convert duration columns to human readable format
    if 'duration' in df.columns:
        df['duration'] = df['duration'].apply(format_duration)
    if 'parent_duration' in df.columns:
        df['parent_duration'] = df['parent_duration'].apply(format_duration)
    
    return df

# Fetch data with progress bar
import time
progress_container = st.empty()
status_container = st.empty()

progress_bar = progress_container.progress(0, text="Initializing...")

# Stage 1: EMC Content Data
status_container.caption("📊 Loading EMC content data...")
progress_bar.progress(0.05, text="Loading EMC content data...")
emc_content_data = load_emc_content_data()

# Stage 2: Assessment Data
status_container.caption("📊 Loading assessment data...")
progress_bar.progress(0.15, text="Loading assessment data...")
udacity_assessment_data = load_assessment_data(emc_content_data)

# Stage 3: Skill Acquisitions
status_container.caption("📊 Loading skill acquisitions...")
progress_bar.progress(0.30, text="Loading skill acquisitions...")
skill_acquisitions_df = load_skill_acquisitions(emc_content_data, udacity_assessment_data)

# Prepare cached data for chatbot (load additional data as needed)
try:
    # Stage 4: Combined Assessments
    status_container.caption("📊 Loading combined assessments...")
    progress_bar.progress(0.40, text="Loading combined assessments...")
    combined_assessments = load_combined_assessments(udacity_assessment_data, emc_content_data)
    
    # Stage 5: Passed Projects
    status_container.caption("📊 Loading passed projects...")
    progress_bar.progress(0.50, text="Loading passed projects...")
    passed_projects = load_passed_projects(emc_content_data)
    
    # Stage 6: User Progress Data
    status_container.caption("📊 Loading user progress data...")
    progress_bar.progress(0.60, text="Loading user progress data...")
    user_progress_data = load_user_progress(emc_content_data)
    
    # Stage 7: Learning Frequency
    status_container.caption("📊 Loading learning frequency...")
    progress_bar.progress(0.70, text="Loading learning frequency...")
    frequency_df = load_learner_frequency(user_progress_data)
    
    # Stage 8: Enrollment Skills
    status_container.caption("📊 Loading enrollment skills...")
    progress_bar.progress(0.75, text="Loading enrollment skills...")
    enrollment_skills = load_skills_by_enrollments(emc_content_data)
    
    # Stage 9: Graduation Skills
    status_container.caption("📊 Loading graduation skills...")
    progress_bar.progress(0.80, text="Loading graduation skills...")
    graduation_skills = load_skills_by_graduations(emc_content_data)
    
    # Stage 10: Workera Data
    status_container.caption("📊 Loading Workera data...")
    progress_bar.progress(0.90, text="Loading Workera data...")
    workera_data = load_workera_data()
    
    # Stage 11: Recommendations
    status_container.caption("📊 Loading recommendations...")
    progress_bar.progress(0.95, text="Loading recommendations...")
    recommendations = load_recommendations(workera_data)
    
    # Finalize
    progress_bar.progress(1.0, text="✅ Data loaded successfully!")
    status_container.empty()
    time.sleep(0.5)
    progress_container.empty()
    
    cached_data = {
        'emc_content_data': emc_content_data,
        'udacity_assessment_data': udacity_assessment_data,
        'skill_acquisitions_df': skill_acquisitions_df,
        'combined_assessments': combined_assessments,
        'passed_projects': passed_projects,
        'user_progress_data': user_progress_data,
        'frequency_df': frequency_df,
        'enrollment_skills': enrollment_skills,
        'graduation_skills': graduation_skills,
        'workera_data': workera_data,
        'recommendations': recommendations
    }
    
    # Render chatbot in sidebar
    render_chat_sidebar(cached_data)
except Exception as e:
    # Clear progress indicators on error
    progress_container.empty()
    status_container.empty()
    # If chatbot fails, show error in sidebar but don't break the main app
    with st.sidebar:
        st.error(f"Chatbot unavailable: {str(e)}")
        st.caption("The main dashboard is still functional.")

# Calculate metrics
if len(skill_acquisitions_df) > 0:
    # Count passed projects
    passed_projects_count = 0
    learners = emc_content_data.get('data', {}).get('company', {}).get('learners', {}).get('edges', [])
    for learner_edge in learners:
        node = learner_edge.get('node', {})
        roster = node.get('roster', [])
        for enrollment in roster:
            learner_activity = enrollment.get('learnerActivity', {})
            submissions = learner_activity.get('submissions', [])
            if submissions:  # Check if submissions is not None
                for submission in submissions:
                    if submission.get('status') == 'passed':
                        passed_projects_count += 1
    
    # Count passed assessments (Udacity + Workera)
    passed_assessments_count = 0
    # Udacity assessments
    for user_data in udacity_assessment_data:
        attempts = user_data.get('attempts', [])
        if attempts:  # Check if attempts is not None
            for attempt in attempts:
                if attempt.get('status') == 'COMPLETED' and attempt.get('result') == 'PASSED':
                    passed_assessments_count += 1
    
    # Workera assessments (above target score)
    try:
        workera_data = data_pipeline.fetch_workera_results()
        workera_data['created_at'] = pd.to_datetime(workera_data['created_at'])
        workera_latest = workera_data.sort_values('created_at', ascending=False).groupby(['user', 'domain_identifier']).first().reset_index()
        workera_passed = workera_latest[workera_latest['score'] > workera_latest['target_score']]
        passed_assessments_count += len(workera_passed)
    except:
        pass  # If workera data not available
    
    # Total skill acquisitions (unique skill + email + date combinations)
    total_skill_acquisitions = len(skill_acquisitions_df)
    
    # Display metrics in cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.metric(
                label="Passed Projects",
                value=f"{passed_projects_count:,}"
            )
    
    with col2:
        with st.container(border=True):
            st.metric(
                label="Passed Assessments",
                value=f"{passed_assessments_count:,}"
            )
    
    with col3:
        with st.container(border=True):
            st.metric(
                label="Total Skill Acquisitions",
                value=f"{total_skill_acquisitions:,}"
            )
    
    # AI-Powered Overall Insights
    with st.expander("🪄 Insights", expanded=False):
        with st.spinner("Generating AI-powered insights..."):
            try:
                cached_data_for_insights = {
                    'skill_acquisitions_df': skill_acquisitions_df,
                    'combined_assessments': combined_assessments,
                    'passed_projects': passed_projects,
                    'frequency_df': frequency_df
                }
                
                insights = generate_overall_insights(cached_data_for_insights)
                
                if insights:
                    st.markdown(insights)
                else:
                    st.info("💡 AI insights require OpenAI API key configuration. Add your key to `settings.py` to enable this feature.")
            except Exception as e:
                st.warning(f"Could not generate AI insights: {str(e)}")

# Process data for chart (cached for faster toggling)
if len(skill_acquisitions_df) > 0:
    grouped = process_skills_timeseries(skill_acquisitions_df)
    
    # Define distinct color palette (brand colors + extrapolated)
    distinct_colors = [
        '#2015FF',  # Bright Blue (primary)
        '#00C5A1',  # Teal
        '#BDEA09',  # Bright Lime
        '#B17CEF',  # Purple
        '#6491FC',  # Light Blue
        '#171A53',  # Dark Navy
        '#142580',  # Darker Blue
        '#8B5CF6',  # Medium Purple (extrapolated)
        '#10B981',  # Emerald (extrapolated)
        '#F59E0B',  # Amber (extrapolated)
        '#EF4444',  # Red (extrapolated)
        '#EC4899',  # Pink (extrapolated)
        '#14B8A6',  # Teal variant (extrapolated)
        '#8B5A00',  # Brown (extrapolated)
        '#6366F1',  # Indigo (extrapolated)
        '#A78BFA',  # Light Purple (extrapolated)
        '#34D399',  # Light Green (extrapolated)
        '#FBBF24',  # Yellow (extrapolated)
        '#F97316',  # Orange (extrapolated)
        '#0B0B0B'   # Black
    ]
    
    # Create card container with border
    with st.container(border=True):
        st.subheader("Skills Acquisition Over Time")
        
        # Add toggle for view type
        view_type = st.radio(
            "View Type:",
            ["Monthly", "Cumulative"],
            horizontal=True
        )
        
        if view_type == "Monthly":
            # Create the stacked bar chart
            chart = alt.Chart(grouped).mark_bar().encode(
                x=alt.X('date_str:N', title='Date', axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('count:Q', title='Number of Skills Acquired', stack='zero'),
                color=alt.Color('domain_name:N', 
                               title='Domain',
                               scale=alt.Scale(range=distinct_colors),
                               legend=alt.Legend(orient='bottom', columns=5)),
                tooltip=[
                    alt.Tooltip('date_str:N', title='Date'),
                    alt.Tooltip('domain_name:N', title='Domain'),
                    alt.Tooltip('count:Q', title='Skills Acquired')
                ]
            ).properties(
                height=500,
                title='Monthly Skill Acquisitions by Domain (Stacked Bar)'
            )
        else:  # Cumulative
            # For cumulative stacked area, use mark_area with cumulative_count
            # This shows how each domain's cumulative total grows over time, stacked
            chart = alt.Chart(grouped).mark_area(opacity=0.7).encode(
                x=alt.X('date_str:N', title='Date', axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('cumulative_count:Q', title='Cumulative Skills Acquired', stack=True),
                color=alt.Color('domain_name:N', 
                               title='Domain',
                               scale=alt.Scale(range=distinct_colors),
                               legend=alt.Legend(orient='bottom', columns=5)),
                order=alt.Order('domain_name:N'),
                tooltip=[
                    alt.Tooltip('date_str:N', title='Date'),
                    alt.Tooltip('domain_name:N', title='Domain'),
                    alt.Tooltip('count:Q', title='Skills Acquired (Period)'),
                    alt.Tooltip('cumulative_count:Q', title='Cumulative Skills')
                ]
            ).properties(
                height=500,
                title='Cumulative Skill Acquisitions by Domain (Stacked Area)'
            )
        
        # Display in Streamlit
        st.altair_chart(chart, width="stretch")
        
        # Show raw data
        with st.expander("View Raw Data"):
            # Prepare data for download: convert date to month format and remove date_str
            display_df = grouped.copy()
            display_df['month'] = display_df['date'].dt.strftime('%Y-%m')
            display_df = display_df.drop(columns=['date', 'date_str'])
            # Reorder columns for better readability
            display_df = display_df[['month', 'domain_name', 'count', 'cumulative_count']]
            st.dataframe(display_df)
    
    # Two column layout for additional charts
    col1, col2 = st.columns(2)
    
    # Left column: Skill Acquisition by Source (Donut Chart)
    with col1:
        with st.container(border=True):
            st.subheader("Skill Acquisition by Source")
            
            # Group by source and count (cached)
            source_counts = process_source_counts(skill_acquisitions_df)
            
            # Create donut chart
            donut_chart = alt.Chart(source_counts).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field='count', type='quantitative'),
                color=alt.Color(
                    field='source', 
                    type='nominal',
                    title='Source',
                    scale=alt.Scale(range=['#2015FF', '#00C5A1', '#B17CEF'])
                ),
                tooltip=[
                    alt.Tooltip('source:N', title='Source'),
                    alt.Tooltip('count:Q', title='Count')
                ]
            ).properties(
                height=300
            )
            
            st.altair_chart(donut_chart, width="stretch")
            
            # Show raw data
            with st.expander("View Raw Data"):
                st.dataframe(source_counts)
    
    # Right column: Learning Frequency (Line Chart)
    with col2:
        with st.container(border=True):
            st.subheader("Learning Frequency Over Time")
            
            try:
                # Load learning frequency data
                user_progress_data = load_user_progress(emc_content_data)
                frequency_df = load_learner_frequency(user_progress_data)
                
                if len(frequency_df) > 0:
                    # Process frequency data (cached)
                    frequency_df = process_frequency_data(frequency_df)
                    
                    # Create line chart showing average active days
                    line_chart = alt.Chart(frequency_df).mark_line(point=True, color='#6491FC').encode(
                        x=alt.X('month_str:N', title='Month', axis=alt.Axis(labelAngle=-45)),
                        y=alt.Y('average_active_days:Q', title='Avg Active Days per Learner'),
                        tooltip=[
                            alt.Tooltip('month_str:N', title='Month'),
                            alt.Tooltip('average_active_days:Q', title='Avg Active Days', format='.1f'),
                            alt.Tooltip('active_learners:Q', title='Active Learners')
                        ]
                    ).properties(
                        height=300
                    )
                    
                    st.altair_chart(line_chart, width="stretch")
                    
                    # Show raw data
                    with st.expander("View Raw Data"):
                        display_freq_df = frequency_df[['month_str', 'average_active_days', 'active_learners']].copy()
                        display_freq_df.columns = ['Month', 'Avg Active Days', 'Active Learners']
                        st.dataframe(display_freq_df)
                else:
                    st.info("No learning frequency data available")
            except Exception as e:
                st.error(f"Error loading learning frequency: {str(e)}")
    
    # Two column layout for assessment performance and most passed projects
    col1, col2 = st.columns(2)
    
    # Left column: Assessment Performance
    with col1:
        with st.container(border=True):
            st.subheader("Assessment Performance")
            
            # Toggle between Workera and Udacity
            assessment_type = st.radio(
                "Assessment Type:",
                ["Udacity", "Workera"],
                horizontal=True,
                key="assessment_type"
            )
            
            try:
                # Load combined assessment data
                combined_assessments = load_combined_assessments(udacity_assessment_data, emc_content_data)
                
                if len(combined_assessments) > 0:
                    # Process assessment performance (cached for faster toggling)
                    avg_scores = process_assessment_performance(combined_assessments, assessment_type)
                    
                    if len(avg_scores) > 0:
                        # Set scale and formatting based on assessment type
                        if assessment_type == "Udacity":
                            y_scale = alt.Scale(domain=[0, 1])
                            y_title = 'Average Score'
                            y_axis = alt.Axis(format='%', title='Average Score')
                            tooltip_format = '.1%'
                        else:  # Workera
                            y_scale = alt.Scale(domain=[0, 300])
                            y_title = 'Average Score'
                            y_axis = alt.Axis(values=[0, 100, 200, 300], title='Average Score')
                            tooltip_format = '.1f'
                        
                        # Create bar chart
                        bar_chart = alt.Chart(avg_scores).mark_bar(color='#2015FF').encode(
                            y=alt.Y('avg_score:Q', 
                                   title=y_title, 
                                   scale=y_scale,
                                   axis=y_axis),
                            x=alt.X('assessment_name:N', 
                                   title='Assessment', 
                                   sort='-y',
                                   axis=alt.Axis(labelAngle=-45, labelLimit=200)),
                            tooltip=[
                                alt.Tooltip('assessment_name:N', title='Assessment'),
                                alt.Tooltip('avg_score:Q', title='Avg Score', format=tooltip_format),
                                alt.Tooltip('num_attempts:Q', title='Attempts')
                            ]
                        ).properties(
                            height=400
                        )
                        
                        st.altair_chart(bar_chart, width="stretch")
                        
                        # Show raw data
                        with st.expander("View Raw Data"):
                            st.dataframe(avg_scores)
                    else:
                        st.info(f"No {assessment_type} assessments with scores available")
                else:
                    st.info("No assessment data available")
            except Exception as e:
                st.error(f"Error loading assessment data: {str(e)}")
    
    # Right column: Most Passed Projects
    with col2:
        with st.container(border=True):
            st.subheader("Most Passed Projects")
            
            try:
                # Load passed projects data
                passed_projects = load_passed_projects(emc_content_data)
                
                if len(passed_projects) > 0:
                    # Process project data (cached)
                    project_counts = process_projects_data(passed_projects)
                    
                    # Create horizontal bar chart
                    bar_chart = alt.Chart(project_counts).mark_bar(color='#BDEA09').encode(
                        x=alt.X('passed_projects:Q', title='Number of Passes'),
                        y=alt.Y('projectName:N', title='Project', sort='-x'),
                        tooltip=[
                            alt.Tooltip('projectName:N', title='Project'),
                            alt.Tooltip('passed_projects:Q', title='Passes')
                        ]
                    ).properties(
                        height=400
                    )
                    
                    st.altair_chart(bar_chart, width="stretch")
                    
                    # Show raw data
                    with st.expander("View Raw Data"):
                        st.dataframe(project_counts)
                else:
                    st.info("No project data available")
            except Exception as e:
                st.error(f"Error loading project data: {str(e)}")
    
    # Two column layout for skills by enrollment and graduation
    col1, col2 = st.columns(2)
    
    # Left column: Skills by Enrollments
    with col1:
        with st.container(border=True):
            st.subheader("Top Skills by Enrollments")
            
            try:
                with st.spinner("Loading enrollment data..."):
                    enrollment_skills = load_skills_by_enrollments(emc_content_data)
                
                if len(enrollment_skills) > 0:
                    # Process enrollment skills (cached)
                    skill_counts = process_enrollment_skills(enrollment_skills)
                    
                    # Create horizontal bar chart
                    bar_chart = alt.Chart(skill_counts).mark_bar(color='#00C5A1').encode(
                        x=alt.X('skill_count:Q', title='Number of Enrollments'),
                        y=alt.Y('skill:N', title='Skill', sort='-x'),
                        tooltip=[
                            alt.Tooltip('skill:N', title='Skill'),
                            alt.Tooltip('skill_count:Q', title='Count')
                        ]
                    ).properties(
                        height=400
                    )
                    
                    st.altair_chart(bar_chart, width="stretch")
                    
                    # Show raw data
                    with st.expander("View Raw Data"):
                        display_df = skill_counts.copy()
                        display_df.columns = ['Skill', 'Count']
                        st.dataframe(display_df)
                else:
                    st.info("No enrollment skills data available")
            except Exception as e:
                st.error(f"Error loading enrollment data: {str(e)}")
    
    # Right column: Skills by Graduations
    with col2:
        with st.container(border=True):
            st.subheader("Top Skills by Graduations")
            
            try:
                with st.spinner("Loading graduation data..."):
                    graduation_skills = load_skills_by_graduations(emc_content_data)
                
                if len(graduation_skills) > 0:
                    # Process graduation skills (cached)
                    skill_counts = process_graduation_skills(graduation_skills)
                    
                    # Create horizontal bar chart
                    bar_chart = alt.Chart(skill_counts).mark_bar(color='#B17CEF').encode(
                        x=alt.X('skill_count:Q', title='Number of Graduations'),
                        y=alt.Y('skill:N', title='Skill', sort='-x'),
                        tooltip=[
                            alt.Tooltip('skill:N', title='Skill'),
                            alt.Tooltip('skill_count:Q', title='Count')
                        ]
                    ).properties(
                        height=400
                    )
                    
                    st.altair_chart(bar_chart, width="stretch")
                    
                    # Show raw data
                    with st.expander("View Raw Data"):
                        display_df = skill_counts.copy()
                        display_df.columns = ['Skill', 'Count']
                        st.dataframe(display_df)
                else:
                    st.info("No graduation skills data available")
            except Exception as e:
                st.error(f"Error loading graduation data: {str(e)}")
    
    # Full-width recommendations table
    with st.container(border=True):
        st.subheader("Recommended Lessons for Skills Improvement")
        
        try:
            with st.spinner("Loading recommendations..."):
                workera_data = load_workera_data()
                recommendations_raw = load_recommendations(workera_data)
            
            if len(recommendations_raw) > 0:
                # AI-Powered Recommendations Insights
                with st.expander("🪄 Insights", expanded=False):
                    with st.spinner("Analyzing recommendations..."):
                        insights = generate_recommendations_insights(recommendations_raw)
                        
                        if insights:
                            st.markdown(insights)
                        else:
                            st.info("💡 AI insights require OpenAI API key configuration.")
                
                # Process recommendations to format durations
                recommendations = process_recommendations(recommendations_raw)
                st.dataframe(recommendations, width="stretch", height=500)
            else:
                st.info("No recommendations available")
        except Exception as e:
            st.error(f"Error loading recommendations: {str(e)}")
else:
    st.warning("No skill acquisition data available")