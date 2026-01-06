"""
Visualization helpers for generating Altair charts
Matches the styling and color scheme of the main dashboard
"""

import pandas as pd
import altair as alt
from typing import Optional, Dict, Any, Tuple

# Define distinct color palette (matching app.py lines 267-288)
DISTINCT_COLORS = [
    '#2015FF',  # Bright Blue (primary)
    '#00C5A1',  # Teal
    '#BDEA09',  # Bright Lime
    '#B17CEF',  # Purple
    '#6491FC',  # Light Blue
    '#171A53',  # Dark Navy
    '#142580',  # Darker Blue
    '#8B5CF6',  # Medium Purple
    '#10B981',  # Emerald
    '#F59E0B',  # Amber
    '#EF4444',  # Red
    '#EC4899',  # Pink
    '#14B8A6',  # Teal variant
    '#8B5A00',  # Brown
    '#6366F1',  # Indigo
    '#A78BFA',  # Light Purple
    '#34D399',  # Light Green
    '#FBBF24',  # Yellow
    '#F97316',  # Orange
    '#0B0B0B'   # Black
]


def determine_sort_order(
    data: pd.DataFrame,
    x_field: str,
    y_field: str,
    chart_type: str,
    horizontal: bool = False
) -> Tuple[Optional[str], pd.DataFrame]:
    """
    Intelligently determine the best sort order for a chart.
    
    Args:
        data: DataFrame with the data
        x_field: X-axis field name
        y_field: Y-axis field name
        chart_type: Type of chart ('bar', 'line', etc.)
        horizontal: Whether it's a horizontal bar chart
        
    Returns:
        Tuple of (sort_order_string, potentially_sorted_dataframe)
        sort_order_string is for Altair (e.g., '-y', 'x', None)
        sorted_dataframe may be pre-sorted if needed
    """
    df = data.copy()
    
    # Time series should always be sorted chronologically (ascending)
    time_indicators = ['month', 'date', 'year', 'time', 'month_str', 'created_at']
    if any(indicator in x_field.lower() for indicator in time_indicators):
        # For time series, sort the data itself to ensure proper ordering
        df = df.sort_values(by=x_field)
        return (None, df)  # None means use natural order (already sorted)
    
    # For bar charts with numeric values
    if chart_type == 'bar':
        # Check if y_field is numeric (counts, scores, etc.)
        if y_field in df.columns and pd.api.types.is_numeric_dtype(df[y_field]):
            # Determine if we should sort ascending or descending
            
            # Most charts with counts/values look better sorted descending (highest first)
            # Exceptions: scores might want to show lowest first to highlight areas needing improvement
            score_indicators = ['score', 'avg_score', 'average_score', 'performance', 'rate']
            is_score = any(indicator in y_field.lower() for indicator in score_indicators)
            
            # Check metadata or field names for context
            improvement_indicators = ['need', 'weakness', 'gap', 'missing']
            is_improvement_context = any(indicator in y_field.lower() for indicator in improvement_indicators)
            
            if is_improvement_context:
                # Sort ascending for improvement metrics (show worst first)
                sort_order = 'y' if not horizontal else 'x'
            else:
                # Default: Sort descending (show highest first)
                sort_order = '-y' if not horizontal else '-x'
            
            # For horizontal bars, pre-sort the data to control order
            if horizontal:
                ascending = is_improvement_context
                df = df.sort_values(by=y_field, ascending=ascending)
                # Return sort based on x-field (the value axis in horizontal mode)
                return ('-x', df) if not ascending else ('x', df)
            
            return (sort_order, df)
        else:
            # Non-numeric y-field: sort alphabetically by x
            return ('x', df)
    
    # For line charts, preserve order or sort by x
    if chart_type == 'line':
        if x_field in df.columns:
            df = df.sort_values(by=x_field)
        return (None, df)
    
    # Default: no specific sorting
    return (None, df)


def create_bar_chart(
    data: pd.DataFrame,
    x_field: str,
    y_field: str,
    x_title: str,
    y_title: str,
    color: str = '#2015FF',
    height: int = 400,
    title: Optional[str] = None,
    sort_by: Optional[str] = None,
    horizontal: bool = False,
    auto_sort: bool = True
) -> alt.Chart:
    """
    Create a bar chart with consistent styling and intelligent sorting.
    
    Args:
        data: DataFrame with data
        x_field: Column name for x-axis
        y_field: Column name for y-axis
        x_title: Title for x-axis
        y_title: Title for y-axis
        color: Bar color
        height: Chart height in pixels
        title: Optional chart title
        sort_by: Sort order ('-y', '-x', 'y', 'x', or None). If None and auto_sort=True, will determine automatically
        horizontal: If True, create horizontal bars
        auto_sort: If True and sort_by is None, automatically determine best sort order
    """
    # Determine optimal sort order if not specified
    if sort_by is None and auto_sort:
        sort_by, data = determine_sort_order(data, x_field, y_field, 'bar', horizontal)
    
    if horizontal:
        # For horizontal bars, x is the value axis and y is the category axis
        y_encoding = alt.Y(f'{x_field}:N', title=x_title)
        if sort_by:
            y_encoding = alt.Y(f'{x_field}:N', title=x_title, sort=sort_by)
        
        chart = alt.Chart(data).mark_bar(color=color).encode(
            x=alt.X(f'{y_field}:Q', title=y_title),
            y=y_encoding,
            tooltip=[
                alt.Tooltip(f'{x_field}:N', title=x_title),
                alt.Tooltip(f'{y_field}:Q', title=y_title)
            ]
        ).properties(height=height)
    else:
        # For vertical bars, x is the category axis and y is the value axis
        x_encoding = alt.X(f'{x_field}:N', title=x_title, axis=alt.Axis(labelAngle=-45))
        if sort_by:
            x_encoding = alt.X(f'{x_field}:N', title=x_title, axis=alt.Axis(labelAngle=-45), sort=sort_by)
        
        chart = alt.Chart(data).mark_bar(color=color).encode(
            x=x_encoding,
            y=alt.Y(f'{y_field}:Q', title=y_title),
            tooltip=[
                alt.Tooltip(f'{x_field}:N', title=x_title),
                alt.Tooltip(f'{y_field}:Q', title=y_title)
            ]
        ).properties(height=height)
    
    if title:
        chart = chart.properties(title=title)
    
    return chart


def create_stacked_bar_chart(
    data: pd.DataFrame,
    x_field: str,
    y_field: str,
    color_field: str,
    x_title: str,
    y_title: str,
    color_title: str,
    height: int = 400,
    title: Optional[str] = None
) -> alt.Chart:
    """
    Create a stacked bar chart with domain coloring.
    """
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X(f'{x_field}:N', title=x_title, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(f'{y_field}:Q', title=y_title, stack='zero'),
        color=alt.Color(
            f'{color_field}:N',
            title=color_title,
            scale=alt.Scale(range=DISTINCT_COLORS),
            legend=alt.Legend(orient='bottom', columns=5)
        ),
        tooltip=[
            alt.Tooltip(f'{x_field}:N', title=x_title),
            alt.Tooltip(f'{color_field}:N', title=color_title),
            alt.Tooltip(f'{y_field}:Q', title=y_title)
        ]
    ).properties(height=height)
    
    if title:
        chart = chart.properties(title=title)
    
    return chart


def create_line_chart(
    data: pd.DataFrame,
    x_field: str,
    y_field: str,
    x_title: str,
    y_title: str,
    color: str = '#6491FC',
    height: int = 400,
    title: Optional[str] = None,
    show_points: bool = True
) -> alt.Chart:
    """
    Create a line chart with consistent styling.
    """
    chart = alt.Chart(data).mark_line(
        point=show_points,
        color=color
    ).encode(
        x=alt.X(f'{x_field}:N', title=x_title, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(f'{y_field}:Q', title=y_title),
        tooltip=[
            alt.Tooltip(f'{x_field}:N', title=x_title),
            alt.Tooltip(f'{y_field}:Q', title=y_title)
        ]
    ).properties(height=height)
    
    if title:
        chart = chart.properties(title=title)
    
    return chart


def create_donut_chart(
    data: pd.DataFrame,
    value_field: str,
    category_field: str,
    category_title: str,
    height: int = 300,
    title: Optional[str] = None
) -> alt.Chart:
    """
    Create a donut chart for categorical data.
    """
    chart = alt.Chart(data).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field=value_field, type='quantitative'),
        color=alt.Color(
            field=category_field,
            type='nominal',
            title=category_title,
            scale=alt.Scale(range=['#2015FF', '#00C5A1', '#B17CEF', '#BDEA09'])
        ),
        tooltip=[
            alt.Tooltip(f'{category_field}:N', title=category_title),
            alt.Tooltip(f'{value_field}:Q', title='Count')
        ]
    ).properties(height=height)
    
    if title:
        chart = chart.properties(title=title)
    
    return chart


def create_area_chart(
    data: pd.DataFrame,
    x_field: str,
    y_field: str,
    color_field: str,
    x_title: str,
    y_title: str,
    color_title: str,
    height: int = 400,
    title: Optional[str] = None
) -> alt.Chart:
    """
    Create a stacked area chart for cumulative data.
    """
    chart = alt.Chart(data).mark_area(opacity=0.7).encode(
        x=alt.X(f'{x_field}:N', title=x_title, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(f'{y_field}:Q', title=y_title, stack=True),
        color=alt.Color(
            f'{color_field}:N',
            title=color_title,
            scale=alt.Scale(range=DISTINCT_COLORS),
            legend=alt.Legend(orient='bottom', columns=5)
        ),
        order=alt.Order(f'{color_field}:N'),
        tooltip=[
            alt.Tooltip(f'{x_field}:N', title=x_title),
            alt.Tooltip(f'{color_field}:N', title=color_title),
            alt.Tooltip(f'{y_field}:Q', title=y_title)
        ]
    ).properties(height=height)
    
    if title:
        chart = chart.properties(title=title)
    
    return chart


def infer_chart_type(data: pd.DataFrame, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Infer the best chart type based on data structure and metadata.
    Includes intelligent sorting recommendations.
    
    Returns a dict with 'chart_type', 'config', and optionally 'sorted_data' if pre-sorted.
    """
    if data is None or len(data) == 0:
        return None
    
    columns = list(data.columns)
    
    # Check for common patterns
    
    # Time series data (month/date column)
    if any(col in columns for col in ['month', 'date', 'month_str']):
        time_col = next((col for col in ['month', 'date', 'month_str'] if col in columns), None)
        value_col = next((col for col in columns if col in ['count', 'average_active_days', 'passed_projects']), None)
        
        if time_col and value_col:
            # Sort time series chronologically
            sorted_data = data.sort_values(by=time_col)
            return {
                'chart_type': 'line',
                'config': {
                    'x_field': time_col,
                    'y_field': value_col,
                    'x_title': time_col.replace('_', ' ').title(),
                    'y_title': value_col.replace('_', ' ').title()
                },
                'sorted_data': sorted_data
            }
    
    # Categorical with counts
    if 'count' in columns or 'skill_count' in columns:
        count_col = 'count' if 'count' in columns else 'skill_count'
        
        # Check for domain/source breakdown
        if 'domain_name' in columns:
            # Sort by count descending (highest domains first)
            sorted_data = data.sort_values(by=count_col, ascending=False)
            return {
                'chart_type': 'bar',
                'config': {
                    'x_field': 'domain_name',
                    'y_field': count_col,
                    'x_title': 'Domain',
                    'y_title': 'Count',
                    'color': '#2015FF',
                    'sort_by': '-y'
                },
                'sorted_data': sorted_data
            }
        elif 'source' in columns:
            # Donut charts don't need explicit sorting (uses natural order)
            return {
                'chart_type': 'donut',
                'config': {
                    'value_field': count_col,
                    'category_field': 'source',
                    'category_title': 'Source'
                }
            }
        elif 'skill' in columns or 'skill_name' in columns:
            skill_col = 'skill' if 'skill' in columns else 'skill_name'
            # Sort by count descending for horizontal bar
            sorted_data = data.sort_values(by=count_col, ascending=False)
            return {
                'chart_type': 'bar',
                'config': {
                    'x_field': skill_col,
                    'y_field': count_col,
                    'x_title': 'Skill',
                    'y_title': 'Count',
                    'color': '#00C5A1',
                    'horizontal': True,
                    'sort_by': '-x'
                },
                'sorted_data': sorted_data
            }
        elif 'program' in columns:
            # For program recommendations
            sorted_data = data.sort_values(by=count_col, ascending=False)
            return {
                'chart_type': 'bar',
                'config': {
                    'x_field': 'program',
                    'y_field': count_col,
                    'x_title': 'Program',
                    'y_title': 'Learners Affected',
                    'color': '#B17CEF',
                    'horizontal': True,
                    'sort_by': '-x'
                },
                'sorted_data': sorted_data
            }
    
    # Assessment performance
    if 'avg_score' in columns and 'assessment_name' in columns:
        # Sort by score descending (highest scores first)
        sorted_data = data.sort_values(by='avg_score', ascending=False)
        return {
            'chart_type': 'bar',
            'config': {
                'x_field': 'assessment_name',
                'y_field': 'avg_score',
                'x_title': 'Assessment',
                'y_title': 'Average Score',
                'color': '#2015FF',
                'sort_by': '-y'
            },
            'sorted_data': sorted_data
        }
    
    # Projects
    if 'projectName' in columns and 'passed_projects' in columns:
        # Sort by passes descending (most passed first)
        sorted_data = data.sort_values(by='passed_projects', ascending=False)
        return {
            'chart_type': 'bar',
            'config': {
                'x_field': 'projectName',
                'y_field': 'passed_projects',
                'x_title': 'Project',
                'y_title': 'Passes',
                'color': '#BDEA09',
                'horizontal': True,
                'sort_by': '-x'
            },
            'sorted_data': sorted_data
        }
    
    # Learners affected (recommendation summaries)
    if 'learners_affected' in columns:
        if 'program' in columns:
            sorted_data = data.sort_values(by='learners_affected', ascending=False)
            return {
                'chart_type': 'bar',
                'config': {
                    'x_field': 'program',
                    'y_field': 'learners_affected',
                    'x_title': 'Program',
                    'y_title': 'Learners Affected',
                    'color': '#B17CEF',
                    'horizontal': True,
                    'sort_by': '-x'
                },
                'sorted_data': sorted_data
            }
    
    # Lesson recommendations
    if 'learners_need' in columns and 'lesson' in columns:
        sorted_data = data.sort_values(by='learners_need', ascending=False)
        return {
            'chart_type': 'bar',
            'config': {
                'x_field': 'lesson',
                'y_field': 'learners_need',
                'x_title': 'Lesson',
                'y_title': 'Learners Need',
                'color': '#00C5A1',
                'horizontal': True,
                'sort_by': '-x'
            },
            'sorted_data': sorted_data
        }
    
    # Generic fallback: Try to find one categorical and one numeric column
    if len(columns) >= 2:
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            # Use first categorical and first numeric column
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            # Decide if horizontal is better (if many categories or long labels)
            num_categories = data[cat_col].nunique()
            horizontal = num_categories > 7 or data[cat_col].astype(str).str.len().max() > 15
            
            # Sort by numeric value
            sorted_data = data.sort_values(by=num_col, ascending=False)
            
            # Check if it's an explicit chart request
            explicit_chart = metadata.get("explicit_chart_request", False)
            requested_type = metadata.get("requested_chart_type", "bar")
            
            if explicit_chart and requested_type == "horizontal_bar":
                horizontal = True
            
            return {
                'chart_type': 'bar',
                'config': {
                    'x_field': cat_col,
                    'y_field': num_col,
                    'x_title': cat_col.replace('_', ' ').title(),
                    'y_title': num_col.replace('_', ' ').title(),
                    'color': '#2015FF',
                    'horizontal': horizontal,
                    'sort_by': '-x' if horizontal else '-y'
                },
                'sorted_data': sorted_data
            }
    
    return None


def generate_chart(data: pd.DataFrame, metadata: Dict[str, Any]) -> Optional[alt.Chart]:
    """
    Automatically generate an appropriate chart based on data structure.
    Includes intelligent sorting for optimal visualization.
    """
    chart_info = infer_chart_type(data, metadata)
    
    if chart_info is None:
        return None
    
    chart_type = chart_info['chart_type']
    config = chart_info['config']
    
    # Use sorted data if provided
    chart_data = chart_info.get('sorted_data', data)
    
    try:
        if chart_type == 'bar':
            return create_bar_chart(chart_data, **config)
        elif chart_type == 'line':
            return create_line_chart(chart_data, **config)
        elif chart_type == 'donut':
            return create_donut_chart(chart_data, **config)
        elif chart_type == 'stacked_bar':
            return create_stacked_bar_chart(chart_data, **config)
        elif chart_type == 'area':
            return create_area_chart(chart_data, **config)
        else:
            return None
    except Exception as e:
        print(f"Error generating chart: {e}")
        return None

