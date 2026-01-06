"""
Chat UI for Skills Analytics Chatbot
Renders in Streamlit sidebar with message history and interactive elements
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
from agent import SkillsAgent
from chart_helpers import generate_chart
import io


def initialize_chat():
    """Initialize chat session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'agent' not in st.session_state:
        st.session_state.agent = None


def render_chat_sidebar(cached_data: Dict[str, Any]):
    """
    Render the chat interface in the sidebar.
    
    Args:
        cached_data: Dictionary of cached data from main app
    """
    initialize_chat()
    
    # Initialize agent with cached data if not already done
    if st.session_state.agent is None:
        st.session_state.agent = SkillsAgent(cached_data)
    
    with st.sidebar:
        st.title("Chat with my Data")
        st.caption("Ask me anything about the skills data! I can see all the same data you can see on this dashboard.")
        
        # Check if OpenAI API key is configured
        import settings
        if not hasattr(settings, 'openai_api_key') or not settings.openai_api_key:
            st.warning("⚠️ OpenAI API key not configured. Please add OPENAI_API_KEY to your Streamlit secrets.")
            st.caption("In Streamlit Cloud: Settings → Secrets → Add `OPENAI_API_KEY = \"your-key\"`")
            return
        
        # Example questions in an expander
        with st.expander("Example Questions", expanded=False):
            st.markdown("""
            **General Questions:**
            - What are the overall metrics?
            - Show me skill acquisitions by domain
            - Which assessments have the highest scores?
            - What are the top 5 most passed projects?
            - Show me learning frequency over time
            
            **Explicit Chart Requests:**
            - Make a chart with the top 5 projects in descending order
            - Create a bar chart of skills by enrollment
            - Show me a visualization of assessment performance
            
            **Recommendations:**
            - Summarize the Workera recommendations
            - Which programs are most recommended?
            """)
        
        # Clear chat button
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent = SkillsAgent(cached_data)
            st.rerun()
        
        st.divider()
        
        # Display chat messages
        chat_container = st.container(height=500)
        with chat_container:
            # Show welcome message if no messages yet
            if len(st.session_state.messages) == 0:
                with st.chat_message("assistant"):
                    st.markdown("""
                    👋 **Welcome to the Skills Analytics Assistant!**
                    
                    I can help you explore and analyze the skills data. Here's what I can do:
                    
                    - 📊 Provide statistics and metrics
                    - 📈 Generate visualizations
                    - 📥 Export data as CSV
                    - 💡 Answer questions about learners, skills, and assessments
                    
                    Try asking a question or check out the example questions above!
                    """)
            
            for idx, message in enumerate(st.session_state.messages):
                role = message["role"]
                content = message.get("content")
                
                if role == "user":
                    with st.chat_message("user"):
                        st.markdown(content)
                
                elif role == "assistant":
                    with st.chat_message("assistant"):
                        st.markdown(content)
                        
                        # Display any data tables
                        if "data" in message and message["data"] is not None:
                            data = message["data"]
                            if isinstance(data, pd.DataFrame) and len(data) > 0:
                                # Show data in expander
                                with st.expander(f"📊 View Data ({len(data)} rows)"):
                                    st.dataframe(data, use_container_width=True)
                                    
                                    # Add download button
                                    csv_buffer = io.StringIO()
                                    data.to_csv(csv_buffer, index=False)
                                    st.download_button(
                                        label="⬇️ Download CSV",
                                        data=csv_buffer.getvalue(),
                                        file_name=f"data_export_{idx}.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                        
                        # Display charts
                        if "chart" in message and message["chart"] is not None:
                            try:
                                st.altair_chart(message["chart"], width="stretch")
                            except Exception as e:
                                st.caption(f"Could not display chart: {str(e)}")
        
        # Chat input at the bottom
        user_input = st.chat_input("Ask a question about the skills data...")
        
        if user_input:
            # Add user message to chat
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Process with agent
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.chat(user_input)
                    
                    # Prepare assistant message
                    assistant_message = {
                        "role": "assistant",
                        "content": response.get("response", "I encountered an error processing your request."),
                        "data": None,
                        "chart": None
                    }
                    
                    # Check if there's data from function calls
                    if response.get("has_data") and "function_results" in response:
                        for func_result in response["function_results"]:
                            result = func_result["result"]
                            
                            if "error" not in result and "data" in result:
                                data = result["data"]
                                metadata = result.get("metadata", {})
                                
                                # Convert to DataFrame if needed
                                if isinstance(data, pd.DataFrame):
                                    assistant_message["data"] = data
                                    
                                    # Check if this was an explicit chart request
                                    explicit_chart = metadata.get("explicit_chart_request", False)
                                    
                                    # Always try to generate a chart for explicit requests or auto-detect
                                    if explicit_chart or len(data) > 0:
                                        chart = generate_chart(data, metadata)
                                        if chart is not None:
                                            assistant_message["chart"] = chart
                                        elif explicit_chart:
                                            # User asked for a chart but we couldn't generate one
                                            # Still show the data, add a note
                                            st.caption("⚠️ Could not auto-generate chart, but data is available below")
                                elif isinstance(data, dict):
                                    # For dict results, show as text (already in response)
                                    pass
                    
                    st.session_state.messages.append(assistant_message)
                    
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ Error: {str(e)}",
                        "data": None,
                        "chart": None
                    })
            
            # Rerun to display new messages
            st.rerun()


def render_chat_interface_main(cached_data: Dict[str, Any]):
    """
    Alternative: Render chat interface in main area (not used currently).
    Kept for potential future use.
    """
    st.header("Skills Analytics Chat")
    
    initialize_chat()
    
    # Initialize agent
    if st.session_state.agent is None:
        st.session_state.agent = SkillsAgent(cached_data)
    
    # Display messages
    for message in st.session_state.messages:
        role = message["role"]
        content = message.get("content")
        
        with st.chat_message(role):
            st.markdown(content)
            
            if role == "assistant":
                # Show data if available
                if "data" in message and message["data"] is not None:
                    st.dataframe(message["data"])
                
                # Show chart if available
                if "chart" in message and message["chart"] is not None:
                    st.altair_chart(message["chart"], width="stretch")
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.chat(prompt)
                st.markdown(response.get("response", "Error occurred."))
                
                # Handle data and charts
                if response.get("has_data") and "function_results" in response:
                    for func_result in response["function_results"]:
                        result = func_result["result"]
                        if "error" not in result and "data" in result:
                            data = result["data"]
                            if isinstance(data, pd.DataFrame):
                                st.dataframe(data)
                                
                                chart = generate_chart(data, result.get("metadata", {}))
                                if chart:
                                    st.altair_chart(chart, width="stretch")
        
        st.rerun()

