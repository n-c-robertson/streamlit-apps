"""
Agentic system for Skills Analytics Chatbot
Uses OpenAI function calling to query data and generate responses
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import data_pipeline
import openai
import settings
import streamlit as st

# Set OpenAI API key
try:
    openai.api_key = settings.openai_api_key
    if not openai.api_key or openai.api_key == 'YOUR_OPENAI_API_KEY_HERE':
        raise ValueError("OpenAI API key not configured")
except (AttributeError, ValueError) as e:
    openai.api_key = None
    print(f"Warning: OpenAI API key not properly configured: {e}")

# Define tools (functions) that the agent can call
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_skill_acquisitions_summary",
            "description": "Get a summary of skill acquisitions, optionally filtered by domain, source, or date range. Returns aggregated counts and statistics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domain_name": {
                        "type": "string",
                        "description": "Optional: Filter by specific domain name (e.g., 'Data Science', 'Programming')"
                    },
                    "source": {
                        "type": "string",
                        "enum": ["Project", "Udacity Assessment", "Workera Assessment"],
                        "description": "Optional: Filter by acquisition source"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Optional: Start date in YYYY-MM-DD format"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "Optional: End date in YYYY-MM-DD format"
                    },
                    "group_by": {
                        "type": "string",
                        "enum": ["domain", "subject", "source", "month", "skill"],
                        "description": "How to group the results. 'domain' = top-level categories (broadest), 'subject' = mid-level categories, 'skill' = individual skills (most granular). Default is 'domain'"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_assessment_performance",
            "description": "Get assessment performance data showing average scores and attempt counts by assessment name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "assessment_type": {
                        "type": "string",
                        "enum": ["Udacity", "Workera"],
                        "description": "Type of assessment to analyze"
                    }
                },
                "required": ["assessment_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_projects",
            "description": "Get the most frequently passed projects with their pass counts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of top projects to return. Default is 10."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_learning_frequency",
            "description": "Get learning frequency statistics showing average active days per month for learners.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_enrollment_skills",
            "description": "Get the top skills by enrollment counts, showing which skills learners are most enrolled in.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of top skills to return. Default is 20."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_graduation_skills",
            "description": "Get the top skills by graduation counts, showing which skills learners have successfully completed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of top skills to return. Default is 20."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_workera_recommendations",
            "description": "Get recommended lessons based on Workera assessment results showing skills that need improvement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of recommendations to return. Default is 10."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_workera_recommendations",
            "description": "Get a comprehensive summary and insights from Workera recommendations, including top programs, lessons, total learners affected, and time investment analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": "string",
                        "enum": ["program", "lesson", "both"],
                        "description": "How to group the summary. 'program' groups by parent course/nanodegree, 'lesson' shows individual lessons, 'both' provides comprehensive insights. Default is 'both'."
                    },
                    "min_learners": {
                        "type": "integer",
                        "description": "Minimum number of learners who need a recommendation to include it. Default is 1."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_learner_details",
            "description": "Get detailed information about specific learners, including their enrollments, assessments, and skill acquisitions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "Optional: Email address of specific learner"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_overall_metrics",
            "description": "Get high-level overview metrics including total projects passed, assessments passed, and skill acquisitions.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_visualization",
            "description": "Explicitly create a chart/visualization based on user request. Use this when the user explicitly asks for a chart, graph, or visualization. Supports bar charts, line charts, and donut charts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "line", "donut", "horizontal_bar"],
                        "description": "Type of chart to create. Use 'horizontal_bar' for better readability with many categories or long labels."
                    },
                    "data_function": {
                        "type": "string",
                        "enum": ["get_top_projects", "get_skill_acquisitions_summary", "get_assessment_performance", "get_enrollment_skills", "get_graduation_skills", "get_learning_frequency"],
                        "description": "Which data function to use as the source"
                    },
                    "function_params": {
                        "type": "object",
                        "description": "Parameters to pass to the data function (e.g., {'limit': 5} for top 5)"
                    },
                    "sort_descending": {
                        "type": "boolean",
                        "description": "Whether to sort values in descending order (highest first). Default true."
                    }
                },
                "required": ["chart_type", "data_function"]
            }
        }
    },
]


class SkillsAgent:
    """Agent that uses OpenAI function calling to answer questions about skills data"""
    
    def __init__(self, cached_data: Dict[str, Any]):
        """
        Initialize the agent with cached data from the main app.
        
        Args:
            cached_data: Dictionary containing pre-loaded data:
                - emc_content_data
                - udacity_assessment_data
                - skill_acquisitions_df
                - combined_assessments
                - passed_projects
                - enrollment_skills
                - graduation_skills
                - workera_data
                - recommendations
                - user_progress_data
                - frequency_df
        """
        self.cached_data = cached_data
        self.conversation_history = []
    
    def execute_function(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a function call requested by the AI.
        
        Returns a dict with 'data' (DataFrame or dict) and 'metadata' (info about the data)
        """
        try:
            if function_name == "get_skill_acquisitions_summary":
                return self._get_skill_acquisitions_summary(**arguments)
            elif function_name == "get_assessment_performance":
                return self._get_assessment_performance(**arguments)
            elif function_name == "get_top_projects":
                return self._get_top_projects(**arguments)
            elif function_name == "get_learning_frequency":
                return self._get_learning_frequency(**arguments)
            elif function_name == "get_enrollment_skills":
                return self._get_enrollment_skills(**arguments)
            elif function_name == "get_graduation_skills":
                return self._get_graduation_skills(**arguments)
            elif function_name == "get_workera_recommendations":
                return self._get_workera_recommendations(**arguments)
            elif function_name == "summarize_workera_recommendations":
                return self._summarize_workera_recommendations(**arguments)
            elif function_name == "get_learner_details":
                return self._get_learner_details(**arguments)
            elif function_name == "get_overall_metrics":
                return self._get_overall_metrics(**arguments)
            elif function_name == "create_visualization":
                return self._create_visualization(**arguments)
            else:
                return {"error": f"Unknown function: {function_name}"}
        except Exception as e:
            return {"error": f"Error executing {function_name}: {str(e)}"}
    
    def _get_skill_acquisitions_summary(
        self, 
        domain_name: Optional[str] = None,
        source: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        group_by: str = "domain"
    ) -> Dict[str, Any]:
        """Get skill acquisitions summary with optional filters"""
        df = self.cached_data.get('skill_acquisitions_df')
        if df is None or len(df) == 0:
            return {"error": "No skill acquisition data available"}
        
        df = df.copy()
        
        # Apply filters
        if domain_name:
            df = df[df['domain_name'] == domain_name]
        if source:
            df = df[df['source'] == source]
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        if len(df) == 0:
            return {"error": "No data found matching the filters"}
        
        # Group by requested dimension
        if group_by == "domain":
            result = df[df['domain_name'].notna()].groupby('domain_name').size().reset_index(name='count')
            result = result.sort_values('count', ascending=False)
        elif group_by == "subject":
            result = df[df['subject_name'].notna()].groupby('subject_name').size().reset_index(name='count')
            result = result.sort_values('count', ascending=False)
        elif group_by == "source":
            result = df.groupby('source').size().reset_index(name='count')
            result = result.sort_values('count', ascending=False)
        elif group_by == "month":
            df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
            result = df.groupby('month').size().reset_index(name='count')
            result['month'] = result['month'].dt.strftime('%Y-%m')
            result = result.sort_values('month')
        elif group_by == "skill":
            result = df.groupby('skill_name').size().reset_index(name='count')
            result = result.sort_values('count', ascending=False).head(20)
        else:
            result = df[df['domain_name'].notna()].groupby('domain_name').size().reset_index(name='count')
            result = result.sort_values('count', ascending=False)
        
        return {
            "data": result,
            "metadata": {
                "total_records": len(df),
                "group_by": group_by,
                "filters_applied": {
                    "domain_name": domain_name,
                    "source": source,
                    "start_date": start_date,
                    "end_date": end_date
                }
            }
        }
    
    def _get_assessment_performance(self, assessment_type: str) -> Dict[str, Any]:
        """Get assessment performance data"""
        combined_assessments = self.cached_data.get('combined_assessments')
        if combined_assessments is None or len(combined_assessments) == 0:
            return {"error": "No assessment data available"}
        
        filtered = combined_assessments[
            combined_assessments['assessmentSource'] == assessment_type
        ].copy()
        
        if len(filtered) == 0:
            return {"error": f"No {assessment_type} assessments found"}
        
        # Calculate average score by assessment
        avg_scores = filtered[filtered['score'] > 0].groupby('assessment_name').agg({
            'score': 'mean',
            'email': 'count'
        }).reset_index()
        avg_scores.columns = ['assessment_name', 'avg_score', 'num_attempts']
        avg_scores = avg_scores.sort_values('avg_score', ascending=False)
        avg_scores['avg_score'] = avg_scores['avg_score'].round(2)
        
        return {
            "data": avg_scores,
            "metadata": {
                "assessment_type": assessment_type,
                "total_assessments": len(avg_scores)
            }
        }
    
    def _get_top_projects(self, limit: int = 10) -> Dict[str, Any]:
        """Get top passed projects"""
        passed_projects = self.cached_data.get('passed_projects')
        if passed_projects is None or len(passed_projects) == 0:
            return {"error": "No project data available"}
        
        project_counts = passed_projects.groupby('projectName')['passed_projects'].sum().reset_index()
        project_counts = project_counts.sort_values('passed_projects', ascending=False).head(limit)
        
        return {
            "data": project_counts,
            "metadata": {
                "limit": limit,
                "total_projects": len(passed_projects['projectName'].unique())
            }
        }
    
    def _get_learning_frequency(self) -> Dict[str, Any]:
        """Get learning frequency data"""
        frequency_df = self.cached_data.get('frequency_df')
        if frequency_df is None or len(frequency_df) == 0:
            return {"error": "No learning frequency data available"}
        
        df = frequency_df.copy()
        df['month'] = df['month'].dt.strftime('%Y-%m')
        
        return {
            "data": df,
            "metadata": {
                "total_months": len(df),
                "avg_active_days_overall": df['average_active_days'].mean().round(2),
                "avg_active_learners": df['active_learners'].mean().round(0)
            }
        }
    
    def _get_enrollment_skills(self, limit: int = 20) -> Dict[str, Any]:
        """Get top enrollment skills"""
        enrollment_skills = self.cached_data.get('enrollment_skills')
        if enrollment_skills is None or len(enrollment_skills) == 0:
            return {"error": "No enrollment skills data available"}
        
        skill_counts = enrollment_skills.groupby('skill').size().reset_index(name='skill_count')
        skill_counts = skill_counts.sort_values('skill_count', ascending=False).head(limit)
        
        return {
            "data": skill_counts,
            "metadata": {
                "limit": limit,
                "total_unique_skills": len(enrollment_skills['skill'].unique())
            }
        }
    
    def _get_graduation_skills(self, limit: int = 20) -> Dict[str, Any]:
        """Get top graduation skills"""
        graduation_skills = self.cached_data.get('graduation_skills')
        if graduation_skills is None or len(graduation_skills) == 0:
            return {"error": "No graduation skills data available"}
        
        skill_counts = graduation_skills.groupby('skill').size().reset_index(name='skill_count')
        skill_counts = skill_counts.sort_values('skill_count', ascending=False).head(limit)
        
        return {
            "data": skill_counts,
            "metadata": {
                "limit": limit,
                "total_unique_skills": len(graduation_skills['skill'].unique())
            }
        }
    
    def _get_workera_recommendations(self, limit: int = 10) -> Dict[str, Any]:
        """Get Workera lesson recommendations"""
        recommendations = self.cached_data.get('recommendations')
        if recommendations is None or len(recommendations) == 0:
            return {"error": "No recommendations available"}
        
        return {
            "data": recommendations.head(limit),
            "metadata": {
                "limit": limit,
                "total_recommendations": len(recommendations)
            }
        }
    
    def _summarize_workera_recommendations(
        self, 
        group_by: str = "both",
        min_learners: int = 1
    ) -> Dict[str, Any]:
        """
        Provide comprehensive summary and insights from Workera recommendations.
        
        Analyzes the dense recommendations table to extract:
        - Most recommended programs/courses
        - Most recommended individual lessons
        - Total learners needing support
        - Time investment analysis
        - Priority recommendations
        """
        recommendations = self.cached_data.get('recommendations')
        if recommendations is None or len(recommendations) == 0:
            return {"error": "No recommendations available"}
        
        df = recommendations.copy()
        
        # Filter by minimum learners
        df = df[df['count'] >= min_learners]
        
        if len(df) == 0:
            return {"error": f"No recommendations with at least {min_learners} learners"}
        
        summary = {}
        
        # Overall statistics
        total_unique_lessons = len(df)
        total_learner_needs = df['count'].sum()
        
        # Parse duration from string (if formatted) or use as-is if numeric
        def parse_duration(duration_val):
            """Extract numeric minutes from duration string or value"""
            if pd.isna(duration_val):
                return 0
            if isinstance(duration_val, (int, float)):
                return float(duration_val)
            # If it's a string like "120 minutes" or "2 hours"
            if isinstance(duration_val, str):
                if 'hour' in duration_val.lower():
                    parts = duration_val.lower().split('hour')
                    hours = float(parts[0].strip())
                    minutes = hours * 60
                    if 'and' in duration_val:
                        # e.g., "2 hours and 30 minutes"
                        min_part = duration_val.split('and')[1].strip()
                        if 'minute' in min_part:
                            minutes += float(min_part.split('minute')[0].strip())
                    return minutes
                elif 'minute' in duration_val.lower():
                    return float(duration_val.split('minute')[0].strip())
            return 0
        
        df['duration_minutes'] = df['duration'].apply(parse_duration)
        df['parent_duration_minutes'] = df['parent_duration'].apply(parse_duration)
        
        # Calculate weighted metrics (multiply by count to see total impact)
        df['total_duration_impact'] = df['duration_minutes'] * df['count']
        
        if group_by in ["program", "both"]:
            # Group by parent program/course
            program_summary = df.groupby('parent_title').agg({
                'count': 'sum',
                'parent_duration_minutes': 'first',
                'id': 'count'
            }).reset_index()
            program_summary.columns = ['program', 'learners_affected', 'program_duration_minutes', 'num_lessons']
            program_summary = program_summary.sort_values('learners_affected', ascending=False)
            
            # Format duration for readability
            def format_minutes(mins):
                if mins == 0 or pd.isna(mins):
                    return "N/A"
                hours = int(mins // 60)
                remaining_mins = int(mins % 60)
                if hours == 0:
                    return f"{remaining_mins}m"
                elif remaining_mins == 0:
                    return f"{hours}h"
                else:
                    return f"{hours}h {remaining_mins}m"
            
            program_summary['program_duration'] = program_summary['program_duration_minutes'].apply(format_minutes)
            program_summary = program_summary.drop(columns=['program_duration_minutes'])
            
            summary['top_programs'] = program_summary.head(10)
        
        if group_by in ["lesson", "both"]:
            # Top individual lessons
            lesson_summary = df[['label', 'parent_title', 'count', 'duration_minutes']].copy()
            lesson_summary = lesson_summary.sort_values('count', ascending=False).head(15)
            
            # Format duration
            lesson_summary['duration'] = lesson_summary['duration_minutes'].apply(
                lambda x: f"{int(x//60)}h {int(x%60)}m" if x >= 60 else f"{int(x)}m" if x > 0 else "N/A"
            )
            lesson_summary = lesson_summary.drop(columns=['duration_minutes'])
            lesson_summary.columns = ['lesson', 'program', 'learners_need', 'duration']
            
            summary['top_lessons'] = lesson_summary
        
        # Priority insights
        avg_learners_per_recommendation = df['count'].mean()
        high_priority = df[df['count'] > avg_learners_per_recommendation]
        
        # Time investment analysis
        total_time_if_all_complete = df['total_duration_impact'].sum()
        avg_time_per_learner = total_time_if_all_complete / total_learner_needs if total_learner_needs > 0 else 0
        
        insights = {
            "total_unique_lessons_recommended": total_unique_lessons,
            "total_learner_needs": int(total_learner_needs),
            "high_priority_lessons": len(high_priority),
            "avg_learners_per_lesson": round(avg_learners_per_recommendation, 1),
            "est_avg_time_per_learner_hours": round(avg_time_per_learner / 60, 1),
            "most_needed_lesson": df.loc[df['count'].idxmax(), 'label'] if len(df) > 0 else "N/A",
            "most_needed_lesson_count": int(df['count'].max()) if len(df) > 0 else 0,
            "unique_programs": df['parent_title'].nunique()
        }
        
        # Prepare return data based on group_by
        if group_by == "program":
            return_data = summary.get('top_programs')
        elif group_by == "lesson":
            return_data = summary.get('top_lessons')
        else:  # both
            # Create a summary DataFrame with key insights
            insight_df = pd.DataFrame([insights])
            return_data = insight_df
        
        return {
            "data": return_data,
            "metadata": {
                "group_by": group_by,
                "insights": insights,
                "top_programs": summary.get('top_programs') if group_by == "both" else None,
                "top_lessons": summary.get('top_lessons') if group_by == "both" else None,
                "min_learners_filter": min_learners
            }
        }
    
    def _get_learner_details(self, email: Optional[str] = None) -> Dict[str, Any]:
        """Get learner details"""
        emc_data = self.cached_data.get('emc_content_data')
        skill_acquisitions = self.cached_data.get('skill_acquisitions_df')
        
        if emc_data is None:
            return {"error": "No learner data available"}
        
        learners = emc_data.get('data', {}).get('company', {}).get('learners', {}).get('edges', [])
        
        if email:
            # Find specific learner
            learner_info = None
            for edge in learners:
                node = edge.get('node', {})
                if node.get('email') == email:
                    learner_info = node
                    break
            
            if not learner_info:
                return {"error": f"Learner with email {email} not found"}
            
            # Get their skill acquisitions
            learner_skills = skill_acquisitions[skill_acquisitions['email'] == email] if skill_acquisitions is not None else pd.DataFrame()
            
            return {
                "data": {
                    "email": email,
                    "enrollments": len(learner_info.get('roster', [])),
                    "skills_acquired": len(learner_skills),
                    "skills_by_source": learner_skills.groupby('source').size().to_dict() if len(learner_skills) > 0 else {}
                },
                "metadata": {
                    "email": email
                }
            }
        else:
            # Return summary of all learners
            total_learners = len(learners)
            return {
                "data": {
                    "total_learners": total_learners,
                    "message": "Provide an email address to get specific learner details"
                },
                "metadata": {
                    "total_learners": total_learners
                }
            }
    
    def _get_overall_metrics(self) -> Dict[str, Any]:
        """Get overall dashboard metrics"""
        emc_data = self.cached_data.get('emc_content_data')
        udacity_assessment_data = self.cached_data.get('udacity_assessment_data')
        skill_acquisitions = self.cached_data.get('skill_acquisitions_df')
        
        # Count passed projects
        passed_projects_count = 0
        if emc_data:
            learners = emc_data.get('data', {}).get('company', {}).get('learners', {}).get('edges', [])
            for learner_edge in learners:
                node = learner_edge.get('node', {})
                roster = node.get('roster', [])
                for enrollment in roster:
                    learner_activity = enrollment.get('learnerActivity', {})
                    submissions = learner_activity.get('submissions', [])
                    if submissions:
                        for submission in submissions:
                            if submission.get('status') == 'passed':
                                passed_projects_count += 1
        
        # Count passed assessments
        passed_assessments_count = 0
        if udacity_assessment_data:
            for user_data in udacity_assessment_data:
                attempts = user_data.get('attempts', [])
                if attempts:
                    for attempt in attempts:
                        if attempt.get('status') == 'COMPLETED' and attempt.get('result') == 'PASSED':
                            passed_assessments_count += 1
        
        # Add Workera assessments
        try:
            workera_data = self.cached_data.get('workera_data')
            if workera_data is not None and len(workera_data) > 0:
                workera_data['created_at'] = pd.to_datetime(workera_data['created_at'])
                workera_latest = workera_data.sort_values('created_at', ascending=False).groupby(['user', 'domain_identifier']).first().reset_index()
                workera_passed = workera_latest[workera_latest['score'] > workera_latest['target_score']]
                passed_assessments_count += len(workera_passed)
        except:
            pass
        
        total_skill_acquisitions = len(skill_acquisitions) if skill_acquisitions is not None else 0
        
        return {
            "data": {
                "passed_projects": passed_projects_count,
                "passed_assessments": passed_assessments_count,
                "total_skill_acquisitions": total_skill_acquisitions
            },
            "metadata": {
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
    
    def _create_visualization(
        self,
        chart_type: str,
        data_function: str,
        function_params: Optional[Dict[str, Any]] = None,
        sort_descending: bool = True
    ) -> Dict[str, Any]:
        """
        Explicitly create a visualization based on user request.
        This ensures charts are generated when users explicitly ask for them.
        """
        if function_params is None:
            function_params = {}
        
        # Map function names to methods
        function_map = {
            "get_top_projects": self._get_top_projects,
            "get_skill_acquisitions_summary": self._get_skill_acquisitions_summary,
            "get_assessment_performance": self._get_assessment_performance,
            "get_enrollment_skills": self._get_enrollment_skills,
            "get_graduation_skills": self._get_graduation_skills,
            "get_learning_frequency": self._get_learning_frequency
        }
        
        if data_function not in function_map:
            return {"error": f"Unknown data function: {data_function}"}
        
        # Get the data
        result = function_map[data_function](**function_params)
        
        if "error" in result:
            return result
        
        data = result.get("data")
        if data is None or (isinstance(data, pd.DataFrame) and len(data) == 0):
            return {"error": "No data available for visualization"}
        
        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            return {"error": "Data must be in table format for visualization"}
        
        # Sort if requested
        if sort_descending and len(data) > 0:
            # Find the numeric column to sort by
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                sort_col = numeric_cols[0]
                data = data.sort_values(by=sort_col, ascending=False)
        
        # Add metadata to indicate explicit chart request
        result["metadata"]["explicit_chart_request"] = True
        result["metadata"]["requested_chart_type"] = chart_type
        result["data"] = data
        
        return result
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user message and return a response.
        
        Returns:
            Dict with 'response' (text), 'data' (optional DataFrame), 'chart_config' (optional)
        """
        # Check if API key is configured
        if not openai.api_key or openai.api_key == 'YOUR_OPENAI_API_KEY_HERE':
            return {
                "response": "⚠️ OpenAI API key not configured. Please add your API key to settings.py to use the chatbot.",
                "error": True
            }
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # System prompt
        system_prompt = """You are a helpful data analyst assistant for a skills analytics dashboard. 
You have access to data about learner skill acquisitions, assessments, projects, and recommendations.

When users ask questions:
1. Determine which function(s) to call to get the relevant data
2. Analyze the results and provide clear, actionable insights
3. Suggest visualizations when appropriate
4. Be concise but informative

**IMPORTANT: When users explicitly ask for a chart, graph, or visualization:**
- Use the 'create_visualization' function to ensure a chart is generated
- Specify the chart_type based on the data (bar, horizontal_bar, line, or donut)
- Pass any parameters like limit, filters, etc. in function_params
- Use horizontal_bar for better readability when there are many categories

**IMPORTANT: Skill Hierarchy Structure:**
- Domain (top level, broadest categories) - e.g., "Programming", "Data Science"
- Subject (middle level) - e.g., "Python", "Machine Learning"
- Skill (bottom level, most specific) - e.g., "List Comprehensions", "Neural Networks"

When grouping skill acquisitions, use 'domain' for high-level overview, 'subject' for mid-level detail.

Available data includes:
- Skill acquisitions from projects, Udacity assessments, and Workera assessments
- Assessment performance (scores, attempt counts)
- Project completion data
- Learning frequency (active days per month)
- Enrollment and graduation statistics by skill
- Lesson recommendations based on Workera assessments
- Comprehensive summaries of recommendations (by program or lesson)

For recommendation-related questions, use 'summarize_workera_recommendations' for insights and analysis,
or 'get_workera_recommendations' for the raw detailed table.

Always provide specific numbers and trends when available."""
        
        messages = [{"role": "system", "content": system_prompt}] + self.conversation_history
        
        try:
            # Call OpenAI with function calling
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto"
            )
            
            assistant_message = response.choices[0].message
            
            # Check if the model wants to call functions
            if assistant_message.tool_calls:
                # Execute all function calls
                function_results = []
                
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    result = self.execute_function(function_name, function_args)
                    function_results.append({
                        "tool_call_id": tool_call.id,
                        "result": result
                    })
                    
                    # Add function call and result to conversation
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call.model_dump()]
                    })
                    
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": self._serialize_function_result(result)
                    })
                
                # Get final response after function calls
                messages = [{"role": "system", "content": system_prompt}] + self.conversation_history
                
                final_response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages
                )
                
                final_text = final_response.choices[0].message.content
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_text
                })
                
                # Return response with data from function calls
                return {
                    "response": final_text,
                    "function_results": function_results,
                    "has_data": True
                }
            else:
                # No function calls, just return the text response
                response_text = assistant_message.content
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_text
                })
                
                return {
                    "response": response_text,
                    "has_data": False
                }
                
        except openai.RateLimitError:
            error_msg = "⚠️ OpenAI API rate limit reached. Please wait a moment and try again."
            return {
                "response": error_msg,
                "error": True
            }
        except openai.AuthenticationError:
            error_msg = "⚠️ Invalid OpenAI API key. Please check your settings.py configuration."
            return {
                "response": error_msg,
                "error": True
            }
        except openai.APIConnectionError:
            error_msg = "⚠️ Could not connect to OpenAI API. Please check your internet connection."
            return {
                "response": error_msg,
                "error": True
            }
        except Exception as e:
            error_msg = f"⚠️ Error processing request: {str(e)}"
            return {
                "response": error_msg,
                "error": True
            }
    
    def _serialize_function_result(self, result: Dict[str, Any]) -> str:
        """Convert function result to string for OpenAI API"""
        if "error" in result:
            return json.dumps({"error": result["error"]})
        
        data = result.get("data")
        metadata = result.get("metadata", {})
        
        # Convert DataFrame to dict for serialization
        if isinstance(data, pd.DataFrame):
            data_dict = data.to_dict('records')
            return json.dumps({
                "data": data_dict,
                "metadata": metadata,
                "row_count": len(data)
            })
        else:
            return json.dumps({
                "data": data,
                "metadata": metadata
            })


# =============================================================================
# AI SUMMARY GENERATION FUNCTIONS
# =============================================================================

def generate_overall_insights(cached_data: Dict[str, Any]) -> Optional[str]:
    """
    Generate AI-powered insights summarizing key trends across all dashboard data.
    
    Args:
        cached_data: Dictionary with all cached data from the dashboard
        
    Returns:
        str: Markdown-formatted insights summary, or None if generation fails
    """
    # Check if API key is configured
    if not openai.api_key or openai.api_key == 'YOUR_OPENAI_API_KEY_HERE':
        return None
    
    try:
        # Gather key metrics and summaries
        skill_acquisitions_df = cached_data.get('skill_acquisitions_df')
        combined_assessments = cached_data.get('combined_assessments')
        passed_projects = cached_data.get('passed_projects')
        frequency_df = cached_data.get('frequency_df')
        
        # Build summary data - convert pandas types to native Python types
        summary_data = {}
        
        if skill_acquisitions_df is not None and len(skill_acquisitions_df) > 0:
            summary_data['total_skill_acquisitions'] = int(len(skill_acquisitions_df))
            # Convert value_counts to dict with native Python types
            skills_by_domain = skill_acquisitions_df['domain_name'].value_counts().head(5)
            summary_data['skills_by_domain'] = {str(k): int(v) for k, v in skills_by_domain.items()}
            skills_by_source = skill_acquisitions_df['source'].value_counts()
            summary_data['skills_by_source'] = {str(k): int(v) for k, v in skills_by_source.items()}
            summary_data['unique_learners'] = int(skill_acquisitions_df['email'].nunique() if 'email' in skill_acquisitions_df else 0)
        
        if combined_assessments is not None and len(combined_assessments) > 0:
            summary_data['total_assessments'] = int(len(combined_assessments))
            summary_data['avg_assessment_score'] = float(combined_assessments['score'].mean())
        
        if passed_projects is not None and len(passed_projects) > 0:
            summary_data['total_projects_passed'] = int(passed_projects['passed_projects'].sum())
            top_projects = passed_projects.nlargest(3, 'passed_projects')[['projectName', 'passed_projects']]
            summary_data['top_projects'] = [
                {'projectName': str(row['projectName']), 'passed_projects': int(row['passed_projects'])}
                for _, row in top_projects.iterrows()
            ]
        
        if frequency_df is not None and len(frequency_df) > 0:
            summary_data['avg_active_days_per_month'] = float(frequency_df['average_active_days'].mean())
            summary_data['avg_active_learners'] = float(frequency_df['active_learners'].mean())
        
        # Create prompt for GPT
        prompt = f"""You are a data analyst reviewing a skills analytics dashboard. 

Here's a summary of the key metrics:

{json.dumps(summary_data, indent=2)}

Provide a concise analytical summary (3-4 paragraphs) covering:
1. Overall performance and engagement patterns observed in the data
2. Key areas of concentration (domains, activities, or sources showing the most activity)
3. Notable trends, distributions, or anomalies in the metrics
4. What the data suggests about learner behavior and skill development patterns

Focus on objective observations and data-driven insights. Use specific numbers from the data. Maintain a warm but neutral, professional tone. Avoid prescriptive language—interpret what the data shows rather than recommending specific actions."""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a skilled data analyst providing objective insights from learning analytics data. Focus on what the data shows, not on prescribing solutions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        return None


def generate_recommendations_insights(recommendations_df: pd.DataFrame) -> Optional[str]:
    """
    Generate AI-powered insights about lesson recommendations with multiple analytical angles.
    
    Args:
        recommendations_df: DataFrame with lesson recommendations
        
    Returns:
        str: Markdown-formatted insights about recommendations, or None if generation fails
    """
    # Check if API key is configured
    if not openai.api_key or openai.api_key == 'YOUR_OPENAI_API_KEY_HERE':
        return None
    
    if recommendations_df is None or len(recommendations_df) == 0:
        return None
    
    try:
        # Analyze from multiple angles - convert pandas types to native Python types
        analysis = {}
        
        # 1. Most needed lessons
        top_lessons = recommendations_df.nlargest(10, 'count')[['label', 'count']]
        analysis['top_lessons'] = [
            {'label': str(row['label']), 'count': int(row['count'])}
            for _, row in top_lessons.iterrows()
        ]
        
        # 2. Programs/courses most frequently recommended
        program_counts = recommendations_df.groupby('parent_title')['count'].sum().nlargest(10)
        analysis['top_programs'] = {str(k): int(v) for k, v in program_counts.items()}
        
        # 3. Duration analysis
        # Parse duration if it's a string
        def safe_duration(val):
            if pd.isna(val):
                return 0
            if isinstance(val, (int, float)):
                return float(val)
            # If string, try to extract numbers
            if isinstance(val, str):
                try:
                    # Try to find first number
                    import re
                    numbers = re.findall(r'\d+', val)
                    return float(numbers[0]) if numbers else 0.0
                except:
                    return 0.0
            return 0.0
        
        durations = recommendations_df['duration'].apply(safe_duration)
        weighted_time = (durations * recommendations_df['count']).sum()
        analysis['total_learning_hours_needed'] = float(weighted_time / 60)  # Convert to hours
        analysis['avg_lesson_duration'] = float(durations.mean())
        
        # 4. Learner impact
        analysis['total_learner_needs'] = int(recommendations_df['count'].sum())
        analysis['unique_lessons'] = int(len(recommendations_df))
        analysis['avg_learners_per_lesson'] = float(recommendations_df['count'].mean())
        
        # 5. High-priority items (above average)
        avg_count = recommendations_df['count'].mean()
        high_priority = recommendations_df[recommendations_df['count'] > avg_count]
        analysis['high_priority_lessons'] = int(len(high_priority))
        
        # Create prompt for GPT
        prompt = f"""You are a data analyst reviewing lesson recommendations generated from skills gap assessments.

Here's an analysis of the recommendation data from multiple angles:

{json.dumps(analysis, indent=2)}

Provide an analytical summary (3-4 paragraphs) covering:
1. **Concentration Areas**: Which lessons and topics appear most frequently in the recommendations? What patterns do you observe in the skill gaps?
2. **Program Themes**: What programs or content areas are most represented? What does this distribution suggest about learner needs?
3. **Scale and Scope**: Based on the number of learners affected and time investment required, what does the data indicate about the magnitude of the learning needs?
4. **Key Observations**: What does the distribution of recommendations—high-priority items, program concentration, lesson variety—reveal about the current skill landscape?

Focus on interpreting what the data shows rather than prescribing solutions. Use specific numbers and be objective. Maintain a warm but neutral, professional tone."""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analyst providing objective insights from skills gap analysis. Focus on interpreting patterns and trends in the data rather than recommending solutions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error generating recommendation insights: {str(e)}")
        return None

