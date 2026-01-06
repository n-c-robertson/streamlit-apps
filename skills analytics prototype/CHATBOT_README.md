# Skills Analytics Chatbot

An intelligent chatbot assistant integrated into the Skills Analytics Dashboard that uses OpenAI's function calling to answer questions about learner data, skills, assessments, and more.

## Features

- 🤖 **Agentic AI**: Uses OpenAI function calling to intelligently query the right data based on your questions
- 📊 **Smart Visualizations**: Automatically generates Altair charts matching the dashboard aesthetic
- 📥 **Data Export**: Download query results as CSV files
- 💬 **Natural Language**: Ask questions in plain English
- 🎯 **Context-Aware**: Has access to all cached dashboard data for fast responses

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure OpenAI API Key

Add your OpenAI API key to `settings.py`:

```python
# OpenAI API Key (for chatbot)
openai_api_key = 'sk-your-actual-api-key-here'
```

You can get an API key from [OpenAI Platform](https://platform.openai.com/api-keys).

### 3. Run the App

```bash
streamlit run app.py
```

The chatbot will appear in the sidebar (collapsible).

## Usage

### Example Questions

**General Data Queries:**
- "What are the overall metrics?"
- "Show me skill acquisitions by domain"
- "Which assessments have the highest scores?"
- "How many skills were acquired from Workera assessments?"
- "Show me data for skills acquired in 2024"
- "Which domain has the most skill acquisitions?"

**Explicit Chart Requests:**
- "Make a chart with the top 5 projects in descending order"
- "Create a bar chart showing the top 10 skills by enrollment"
- "Show me a horizontal bar chart of assessment performance"
- "Visualize learning frequency over time as a line chart"
- "Give me a chart of the most needed lessons"

**Recommendations Analysis:**
- "Summarize the Workera recommendations"
- "Which programs are most recommended for learners?"
- "What are the top lessons learners need?"
- "Show me high-priority skill recommendations"

### Features

#### Data Queries
The chatbot can query:
- Skill acquisitions (by domain, source, time period)
- Assessment performance (Udacity and Workera)
- Project completion statistics
- Learning frequency metrics
- Enrollment and graduation data
- Workera lesson recommendations (raw and summarized)
- Learner-specific details

**Recommendation Insights**: The chatbot can intelligently summarize the dense Workera recommendations table to provide:
- Top recommended programs/courses by learner impact
- Most needed individual lessons
- Total learners needing support
- Time investment analysis
- Priority recommendations based on demand

#### Visualizations
Charts are automatically generated when appropriate with **intelligent sorting**:
- Bar charts for categorical data (sorted by value, highest first)
- Line charts for time series (sorted chronologically)
- Donut charts for distributions
- Stacked charts for multi-dimensional data

**Explicit Chart Requests:**
When you explicitly ask for a chart, the chatbot will:
- Use the `create_visualization` function to ensure a chart is generated
- Respect your specified chart type (bar, horizontal bar, line, donut)
- Apply any limits or filters you mention (e.g., "top 5")
- Sort in descending order by default (unless you specify otherwise)

Examples:
- "Make a chart with only the top 5 projects in descending order" ✅
- "Create a horizontal bar chart of the top 10 skills" ✅
- "Show me a line chart of learning frequency over time" ✅

**Intelligent Sorting Features:**
- Time series are sorted chronologically (ascending)
- Bar charts with counts/metrics are sorted descending (highest first)
- Assessment scores are sorted to show top performers first
- Recommendation data is sorted by impact (learners affected)
- Contextual sorting based on data type and field names
- Generic fallback for any data with 1 categorical + 1 numeric column

All charts use the same color scheme as the main dashboard.

#### Data Export
Click the "Download CSV" button in any data response to export the results.

## Architecture

### Components

1. **agent.py**: Core agent system with OpenAI function calling
   - Defines 9 queryable functions
   - Handles conversation history
   - Executes data transformations

2. **chart_helpers.py**: Visualization generation
   - Altair chart templates
   - Automatic chart type inference
   - Consistent styling with main dashboard

3. **chat_ui.py**: Streamlit sidebar interface
   - Message history display
   - Chat input handling
   - Data table and chart rendering
   - Download functionality

### Data Access

The chatbot uses a **hybrid data access strategy**:
- **Cached data**: Uses pre-loaded data from the main app for fast responses
- **Dynamic queries**: Calls data pipeline functions when needed for specific filters or transformations

### Model

Uses **GPT-4o-mini** for cost-effective, fast responses with function calling capabilities.

## Troubleshooting

### "OpenAI API key not configured"

Make sure you've set the `openai_api_key` in `settings.py` and it's not the placeholder value.

### Rate Limit Errors

If you see rate limit errors, you may be hitting OpenAI's usage limits. Wait a moment and try again, or upgrade your OpenAI plan.

### Charts Not Appearing When Requested

If you explicitly ask for a chart but get only a text table:
1. **Be explicit**: Use words like "chart", "graph", "visualize", "plot"
2. **Specify the type**: "bar chart", "horizontal bar chart", "line chart"
3. **Try rephrasing**: "Create a visualization of..." or "Show me a chart with..."

The chatbot now has a dedicated `create_visualization` function that activates when it detects explicit chart requests.

### Chatbot Not Appearing

Make sure all dependencies are installed:
```bash
pip install openai streamlit pandas altair
```

### Import Errors

If you get import errors, ensure all new files are in the same directory:
- `agent.py`
- `chart_helpers.py`
- `chat_ui.py`

## Cost Considerations

The chatbot uses OpenAI's GPT-4o-mini model:
- **Input**: ~$0.15 per 1M tokens
- **Output**: ~$0.60 per 1M tokens

Typical queries cost less than $0.01 each. For high-volume usage, consider implementing caching or using a different model.

## Future Enhancements

Potential improvements:
- [ ] Streaming responses for better UX
- [ ] Conversation persistence across sessions
- [ ] More complex multi-step queries
- [ ] Custom chart configurations
- [ ] Export to Excel with formatting
- [ ] Voice input support
- [ ] Multi-language support

## Support

For issues or questions, refer to the main dashboard documentation or contact the development team.

