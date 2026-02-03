# ─────────────────────────────────────────────────────────────────────────────
# AI Agent for Skills Challenge Analysis
# ─────────────────────────────────────────────────────────────────────────────

import json
import pandas as pd
import numpy as np
from openai import OpenAI
from settings import OPENAI_API_KEY


def json_serializer(obj):
    """Custom JSON serializer for numpy types."""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def to_json(data):
    """Convert data to JSON string, handling numpy types."""
    return json.dumps(data, default=json_serializer)


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definitions
# ─────────────────────────────────────────────────────────────────────────────

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_summary_metrics",
            "description": "Get high-level summary metrics: average score, number of users, enterprise benchmarks, score range",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_subdomain_coverage",
            "description": "Get coverage percentages for each subdomain (strong vs weak skills). Use this to understand broad areas of strength and weakness.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_topic_coverage",
            "description": "Get detailed coverage percentages for each topic within subdomains. More granular than subdomain coverage.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_skill_gaps",
            "description": "Get the top skill gaps (skills marked as needing improvement) across the cohort. Essential for understanding training priorities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Number of top gaps to return", "default": 20}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_strengths",
            "description": "Get the top strengths (skills marked as strong) across the cohort. Useful for identifying what the organization does well.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Number of top strengths to return", "default": 20}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_score_distribution",
            "description": "Get score distribution statistics: min, max, mean, median, quartiles, and counts by skill level band (Beginner/Developing/Advanced)",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recommended_courses",
            "description": "Get the top recommended courses/programs based on skill gaps. Requires recommendations to have been generated first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Number of courses to return", "default": 10}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_details",
            "description": "Get detailed information about specific users including their scores, strong skills, and weak skills. Can filter by score band.",
            "parameters": {
                "type": "object",
                "properties": {
                    "score_filter": {
                        "type": "string",
                        "enum": ["all", "low", "medium", "high"],
                        "description": "Filter users by score band: low (<100), medium (100-200), high (>200)"
                    },
                    "limit": {"type": "integer", "description": "Number of users to return", "default": 10}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_assessment_info",
            "description": "Get information about the current assessment being analyzed including name, user count, and data availability",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "analyze_learning_plan",
            "description": "Analyze and suggest a learning plan structure with test-out rules based on score thresholds. Uses recommended courses and score distribution to create actionable plans.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_test_out_rules": {
                        "type": "boolean",
                        "description": "Whether to include suggested test-out score thresholds",
                        "default": True
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_to_benchmark",
            "description": "Compare the cohort's performance to enterprise benchmarks. Useful for contextualizing results.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_skills_by_subdomain",
            "description": "Get all skills within a specific subdomain, with their strength/weakness counts",
            "parameters": {
                "type": "object",
                "properties": {
                    "subdomain": {"type": "string", "description": "The subdomain name to filter by"}
                },
                "required": ["subdomain"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "identify_critical_gaps",
            "description": "Identify critical skill gaps - skills where more than a threshold percentage of users need improvement",
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold_pct": {
                        "type": "number",
                        "description": "Minimum percentage of cohort that must have this gap (default 30)",
                        "default": 30
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "segment_users_by_performance",
            "description": "Segment users into performance tiers and provide summary statistics for each tier",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# Tool Execution
# ─────────────────────────────────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_args: dict, session_data: dict) -> str:
    """Execute an agent tool and return the result as a JSON string."""
    
    attempts_df = session_data.get('attempts_df')
    weak_df = session_data.get('weak_df')
    strong_df = session_data.get('strong_df')
    taxonomy_skills = session_data.get('taxonomy_skills')
    selected_assessment = session_data.get('selected_assessment', 'Unknown')
    all_recommendations = session_data.get('all_recommendations', [])
    
    # ─────────────────────────────────────────────────────────────────
    # get_summary_metrics
    # ─────────────────────────────────────────────────────────────────
    if tool_name == "get_summary_metrics":
        if attempts_df is None or len(attempts_df) == 0:
            return to_json({"error": "No data loaded"})
        
        scores = attempts_df['score'].dropna()
        result = {
            "assessment": selected_assessment,
            "total_users": len(attempts_df),
            "average_score": round(scores.mean(), 1) if len(scores) > 0 else None,
            "median_score": round(scores.median(), 1) if len(scores) > 0 else None,
            "min_score": round(scores.min(), 1) if len(scores) > 0 else None,
            "max_score": round(scores.max(), 1) if len(scores) > 0 else None,
            "std_deviation": round(scores.std(), 1) if len(scores) > 0 else None,
        }
        
        if 'enterprise_average_score' in attempts_df.columns:
            ent_avg = attempts_df['enterprise_average_score'].max()
            result["enterprise_average"] = round(ent_avg, 1) if pd.notna(ent_avg) else None
        if 'enterprise_percentile_75_score' in attempts_df.columns:
            ent_75 = attempts_df['enterprise_percentile_75_score'].max()
            result["enterprise_75th_percentile"] = round(ent_75, 1) if pd.notna(ent_75) else None
            
        return to_json(result)
    
    # ─────────────────────────────────────────────────────────────────
    # get_subdomain_coverage
    # ─────────────────────────────────────────────────────────────────
    elif tool_name == "get_subdomain_coverage":
        if weak_df is None and strong_df is None:
            return to_json({"error": "No skill data loaded"})
        
        weak_counts = weak_df.Subdomain.value_counts().to_dict() if weak_df is not None and len(weak_df) > 0 else {}
        strong_counts = strong_df.Subdomain.value_counts().to_dict() if strong_df is not None and len(strong_df) > 0 else {}
        
        all_subdomains = set(list(weak_counts.keys()) + list(strong_counts.keys()))
        coverage = []
        for sd in all_subdomains:
            weak = weak_counts.get(sd, 0)
            strong = strong_counts.get(sd, 0)
            total = weak + strong
            coverage.append({
                "subdomain": sd,
                "weak_count": weak,
                "strong_count": strong,
                "coverage_pct": round(strong / total * 100, 1) if total > 0 else 0
            })
        
        coverage.sort(key=lambda x: x['coverage_pct'], reverse=True)
        return to_json({"subdomain_coverage": coverage})
    
    # ─────────────────────────────────────────────────────────────────
    # get_topic_coverage
    # ─────────────────────────────────────────────────────────────────
    elif tool_name == "get_topic_coverage":
        if weak_df is None and strong_df is None:
            return to_json({"error": "No skill data loaded"})
        
        weak_counts = weak_df.groupby(['Subdomain', 'Topic']).size().to_dict() if weak_df is not None and len(weak_df) > 0 else {}
        strong_counts = strong_df.groupby(['Subdomain', 'Topic']).size().to_dict() if strong_df is not None and len(strong_df) > 0 else {}
        
        all_topics = set(list(weak_counts.keys()) + list(strong_counts.keys()))
        coverage = []
        for (sd, topic) in all_topics:
            weak = weak_counts.get((sd, topic), 0)
            strong = strong_counts.get((sd, topic), 0)
            total = weak + strong
            coverage.append({
                "subdomain": sd,
                "topic": topic,
                "weak_count": weak,
                "strong_count": strong,
                "coverage_pct": round(strong / total * 100, 1) if total > 0 else 0
            })
        
        coverage.sort(key=lambda x: (x['subdomain'], -x['coverage_pct']))
        return to_json({"topic_coverage": coverage})
    
    # ─────────────────────────────────────────────────────────────────
    # get_top_skill_gaps
    # ─────────────────────────────────────────────────────────────────
    elif tool_name == "get_top_skill_gaps":
        if weak_df is None or len(weak_df) == 0:
            return to_json({"error": "No weak skills data"})
        
        limit = tool_args.get('limit', 20)
        skill_gaps = weak_df.groupby(['Subdomain', 'Topic', 'Skill']).size().reset_index(name='count')
        skill_gaps = skill_gaps.sort_values('count', ascending=False).head(limit)
        
        gaps = []
        for _, row in skill_gaps.iterrows():
            gaps.append({
                "skill": row['Skill'],
                "topic": row['Topic'],
                "subdomain": row['Subdomain'],
                "users_affected": int(row['count']),
                "pct_of_cohort": round(row['count'] / len(attempts_df) * 100, 1) if attempts_df is not None else 0
            })
        
        return to_json({"top_skill_gaps": gaps, "total_gaps_found": len(gaps)})
    
    # ─────────────────────────────────────────────────────────────────
    # get_top_strengths
    # ─────────────────────────────────────────────────────────────────
    elif tool_name == "get_top_strengths":
        if strong_df is None or len(strong_df) == 0:
            return to_json({"error": "No strong skills data"})
        
        limit = tool_args.get('limit', 20)
        skill_strengths = strong_df.groupby(['Subdomain', 'Topic', 'Skill']).size().reset_index(name='count')
        skill_strengths = skill_strengths.sort_values('count', ascending=False).head(limit)
        
        strengths = []
        for _, row in skill_strengths.iterrows():
            strengths.append({
                "skill": row['Skill'],
                "topic": row['Topic'],
                "subdomain": row['Subdomain'],
                "users_with_strength": int(row['count']),
                "pct_of_cohort": round(row['count'] / len(attempts_df) * 100, 1) if attempts_df is not None else 0
            })
        
        return to_json({"top_strengths": strengths})
    
    # ─────────────────────────────────────────────────────────────────
    # get_user_score_distribution
    # ─────────────────────────────────────────────────────────────────
    elif tool_name == "get_user_score_distribution":
        if attempts_df is None or len(attempts_df) == 0:
            return to_json({"error": "No data loaded"})
        
        scores = attempts_df['score'].dropna()
        
        # Score bands
        beginner = len(scores[scores < 100])
        developing = len(scores[(scores >= 100) & (scores < 200)])
        advanced = len(scores[scores >= 200])
        
        result = {
            "total_users": len(scores),
            "min": round(scores.min(), 1),
            "max": round(scores.max(), 1),
            "mean": round(scores.mean(), 1),
            "median": round(scores.median(), 1),
            "std_dev": round(scores.std(), 1),
            "25th_percentile": round(scores.quantile(0.25), 1),
            "75th_percentile": round(scores.quantile(0.75), 1),
            "score_bands": {
                "beginner_0_100": {"count": beginner, "pct": round(beginner / len(scores) * 100, 1)},
                "developing_100_200": {"count": developing, "pct": round(developing / len(scores) * 100, 1)},
                "advanced_200_300": {"count": advanced, "pct": round(advanced / len(scores) * 100, 1)}
            }
        }
        return to_json(result)
    
    # ─────────────────────────────────────────────────────────────────
    # get_recommended_courses
    # ─────────────────────────────────────────────────────────────────
    elif tool_name == "get_recommended_courses":
        if not all_recommendations:
            return to_json({"error": "No recommendations generated yet. User needs to click 'Generate Recommendations' in the app first."})
        
        limit = tool_args.get('limit', 10)
        recs_df = pd.DataFrame(all_recommendations)
        
        course_counts = recs_df.groupby(['parent_key', 'parent_title']).agg(
            total_matches=('skill', 'count'),
            unique_users=('user', 'nunique'),
            skills_list=('skill', lambda x: list(set(x))[:10])
        ).reset_index()
        course_counts = course_counts.sort_values('total_matches', ascending=False).head(limit)
        
        courses = []
        for _, row in course_counts.iterrows():
            courses.append({
                "course_title": row['parent_title'],
                "course_key": row['parent_key'],
                "total_skill_matches": int(row['total_matches']),
                "unique_users_matched": int(row['unique_users']),
                "pct_of_cohort": round(row['unique_users'] / len(attempts_df) * 100, 1) if attempts_df is not None else 0,
                "skills_addressed": row['skills_list']
            })
        
        return to_json({"recommended_courses": courses})
    
    # ─────────────────────────────────────────────────────────────────
    # get_user_details
    # ─────────────────────────────────────────────────────────────────
    elif tool_name == "get_user_details":
        if attempts_df is None or len(attempts_df) == 0:
            return to_json({"error": "No data loaded"})
        
        score_filter = tool_args.get('score_filter', 'all')
        limit = tool_args.get('limit', 10)
        
        df = attempts_df.copy()
        if score_filter == 'low':
            df = df[df['score'] < 100]
        elif score_filter == 'medium':
            df = df[(df['score'] >= 100) & (df['score'] < 200)]
        elif score_filter == 'high':
            df = df[df['score'] >= 200]
        
        df = df.head(limit)
        
        users = []
        for _, row in df.iterrows():
            users.append({
                "email": row['workera_user_email'],
                "score": round(row['score'], 1) if pd.notna(row['score']) else None,
                "num_strong_skills": len(row['strong_skills']) if row['strong_skills'] else 0,
                "num_weak_skills": len(row['needs_improvement_skills']) if row['needs_improvement_skills'] else 0,
                "strong_skills_sample": row['strong_skills'][:5] if row['strong_skills'] else [],
                "weak_skills_sample": row['needs_improvement_skills'][:5] if row['needs_improvement_skills'] else []
            })
        
        return to_json({"users": users, "filter_applied": score_filter, "users_in_filter": len(df)})
    
    # ─────────────────────────────────────────────────────────────────
    # get_assessment_info
    # ─────────────────────────────────────────────────────────────────
    elif tool_name == "get_assessment_info":
        return to_json({
            "assessment_name": selected_assessment,
            "total_users": len(attempts_df) if attempts_df is not None else 0,
            "has_taxonomy": taxonomy_skills is not None,
            "has_recommendations": len(all_recommendations) > 0,
            "num_recommendations": len(all_recommendations),
            "score_scale": "0-300 (Beginner: 0-100, Developing: 100-200, Advanced: 200-300)"
        })
    
    # ─────────────────────────────────────────────────────────────────
    # analyze_learning_plan
    # ─────────────────────────────────────────────────────────────────
    elif tool_name == "analyze_learning_plan":
        if not all_recommendations:
            return to_json({"error": "No recommendations generated yet. Cannot create learning plan without course recommendations."})
        
        include_test_out = tool_args.get('include_test_out_rules', True)
        
        recs_df = pd.DataFrame(all_recommendations)
        course_counts = recs_df.groupby(['parent_key', 'parent_title']).agg(
            total_matches=('skill', 'count'),
            unique_users=('user', 'nunique'),
            skills_list=('skill', lambda x: list(set(x)))
        ).reset_index()
        course_counts = course_counts.sort_values('total_matches', ascending=False)
        
        # Get score distribution for context
        scores = attempts_df['score'].dropna() if attempts_df is not None else pd.Series()
        median_score = scores.median() if len(scores) > 0 else 150
        pct_below_150 = len(scores[scores < 150]) / len(scores) * 100 if len(scores) > 0 else 50
        
        learning_plan = []
        for idx, row in course_counts.head(5).iterrows():
            plan_item = {
                "order": len(learning_plan) + 1,
                "course": row['parent_title'],
                "course_key": row['parent_key'],
                "skills_addressed": row['skills_list'][:10],
                "users_who_need_this": int(row['unique_users']),
                "pct_of_cohort": round(row['unique_users'] / len(attempts_df) * 100, 1) if attempts_df is not None else 0,
                "priority": "High" if row['unique_users'] > len(attempts_df) * 0.5 else "Medium" if row['unique_users'] > len(attempts_df) * 0.25 else "Standard"
            }
            
            if include_test_out:
                # Suggest test-out thresholds - higher thresholds for later courses
                base_threshold = 150 if len(learning_plan) < 2 else 175 if len(learning_plan) < 4 else 200
                plan_item["test_out_rule"] = {
                    "score_threshold": base_threshold,
                    "description": f"Test out if Workera assessment score >= {base_threshold}/300",
                    "rationale": f"Users scoring {base_threshold}+ have demonstrated foundational competency in these skills"
                }
            
            learning_plan.append(plan_item)
        
        return to_json({
            "learning_plan": learning_plan,
            "cohort_context": {
                "total_users": len(attempts_df) if attempts_df is not None else 0,
                "median_score": round(median_score, 1),
                "pct_below_150": round(pct_below_150, 1),
                "recommendation": "Start with foundational courses for users below 150, offer test-out for higher performers"
            }
        })
    
    # ─────────────────────────────────────────────────────────────────
    # compare_to_benchmark
    # ─────────────────────────────────────────────────────────────────
    elif tool_name == "compare_to_benchmark":
        if attempts_df is None or len(attempts_df) == 0:
            return to_json({"error": "No data loaded"})
        
        scores = attempts_df['score'].dropna()
        cohort_avg = scores.mean()
        
        result = {
            "cohort_average": round(cohort_avg, 1),
            "cohort_median": round(scores.median(), 1),
        }
        
        if 'enterprise_average_score' in attempts_df.columns:
            ent_avg = attempts_df['enterprise_average_score'].max()
            if pd.notna(ent_avg):
                result["enterprise_average"] = round(ent_avg, 1)
                result["vs_enterprise_avg"] = round(cohort_avg - ent_avg, 1)
                result["vs_enterprise_avg_pct"] = round((cohort_avg - ent_avg) / ent_avg * 100, 1)
                result["performance_vs_enterprise"] = "above" if cohort_avg > ent_avg else "below" if cohort_avg < ent_avg else "at"
        
        if 'enterprise_percentile_75_score' in attempts_df.columns:
            ent_75 = attempts_df['enterprise_percentile_75_score'].max()
            if pd.notna(ent_75):
                result["enterprise_75th_percentile"] = round(ent_75, 1)
                result["pct_above_ent_75th"] = round(len(scores[scores >= ent_75]) / len(scores) * 100, 1)
        
        return to_json(result)
    
    # ─────────────────────────────────────────────────────────────────
    # get_skills_by_subdomain
    # ─────────────────────────────────────────────────────────────────
    elif tool_name == "get_skills_by_subdomain":
        subdomain = tool_args.get('subdomain')
        if not subdomain:
            return to_json({"error": "subdomain parameter is required"})
        
        if weak_df is None and strong_df is None:
            return to_json({"error": "No skill data loaded"})
        
        # Get skills in this subdomain
        weak_in_sd = weak_df[weak_df['Subdomain'] == subdomain] if weak_df is not None else pd.DataFrame()
        strong_in_sd = strong_df[strong_df['Subdomain'] == subdomain] if strong_df is not None else pd.DataFrame()
        
        weak_skill_counts = weak_in_sd.groupby('Skill').size().to_dict() if len(weak_in_sd) > 0 else {}
        strong_skill_counts = strong_in_sd.groupby('Skill').size().to_dict() if len(strong_in_sd) > 0 else {}
        
        all_skills = set(list(weak_skill_counts.keys()) + list(strong_skill_counts.keys()))
        
        skills = []
        for skill in all_skills:
            weak = weak_skill_counts.get(skill, 0)
            strong = strong_skill_counts.get(skill, 0)
            total = weak + strong
            skills.append({
                "skill": skill,
                "weak_count": weak,
                "strong_count": strong,
                "coverage_pct": round(strong / total * 100, 1) if total > 0 else 0
            })
        
        skills.sort(key=lambda x: x['coverage_pct'])
        
        return to_json({
            "subdomain": subdomain,
            "skills": skills,
            "total_skills": len(skills)
        })
    
    # ─────────────────────────────────────────────────────────────────
    # identify_critical_gaps
    # ─────────────────────────────────────────────────────────────────
    elif tool_name == "identify_critical_gaps":
        if weak_df is None or len(weak_df) == 0:
            return to_json({"error": "No weak skills data"})
        
        threshold = tool_args.get('threshold_pct', 30)
        total_users = len(attempts_df) if attempts_df is not None else 1
        min_users = total_users * threshold / 100
        
        skill_gaps = weak_df.groupby(['Subdomain', 'Topic', 'Skill']).size().reset_index(name='count')
        critical = skill_gaps[skill_gaps['count'] >= min_users].sort_values('count', ascending=False)
        
        gaps = []
        for _, row in critical.iterrows():
            gaps.append({
                "skill": row['Skill'],
                "topic": row['Topic'],
                "subdomain": row['Subdomain'],
                "users_affected": int(row['count']),
                "pct_of_cohort": round(row['count'] / total_users * 100, 1)
            })
        
        return to_json({
            "critical_gaps": gaps,
            "threshold_used": threshold,
            "num_critical_gaps": len(gaps)
        })
    
    # ─────────────────────────────────────────────────────────────────
    # segment_users_by_performance
    # ─────────────────────────────────────────────────────────────────
    elif tool_name == "segment_users_by_performance":
        if attempts_df is None or len(attempts_df) == 0:
            return to_json({"error": "No data loaded"})
        
        scores = attempts_df['score'].dropna()
        
        segments = {
            "struggling": {
                "range": "0-99",
                "count": len(scores[scores < 100]),
                "avg_score": round(scores[scores < 100].mean(), 1) if len(scores[scores < 100]) > 0 else None,
                "recommendation": "Require full learning path, no test-outs"
            },
            "developing": {
                "range": "100-149",
                "count": len(scores[(scores >= 100) & (scores < 150)]),
                "avg_score": round(scores[(scores >= 100) & (scores < 150)].mean(), 1) if len(scores[(scores >= 100) & (scores < 150)]) > 0 else None,
                "recommendation": "Targeted skill-building, some test-out options"
            },
            "proficient": {
                "range": "150-199",
                "count": len(scores[(scores >= 150) & (scores < 200)]),
                "avg_score": round(scores[(scores >= 150) & (scores < 200)].mean(), 1) if len(scores[(scores >= 150) & (scores < 200)]) > 0 else None,
                "recommendation": "Advanced topics, test-out of fundamentals"
            },
            "advanced": {
                "range": "200-300",
                "count": len(scores[scores >= 200]),
                "avg_score": round(scores[scores >= 200].mean(), 1) if len(scores[scores >= 200]) > 0 else None,
                "recommendation": "Specialized/advanced content, mentor others"
            }
        }
        
        total = len(scores)
        for seg in segments.values():
            seg["pct_of_cohort"] = round(seg["count"] / total * 100, 1) if total > 0 else 0
        
        return to_json({"segments": segments, "total_users": total})
    
    # ─────────────────────────────────────────────────────────────────
    # Unknown tool
    # ─────────────────────────────────────────────────────────────────
    return to_json({"error": f"Unknown tool: {tool_name}"})


# ─────────────────────────────────────────────────────────────────────────────
# Agent Conversation Handler
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert learning & development analyst helping interpret Workera Skills Challenge assessment data.

You have access to tools that can query the assessment data. Use them to provide data-driven insights.

Key context:
- Workera scores range from 0-300: Beginner (0-100), Developing (100-200), Advanced (200-300)
- "Coverage" means the % of skills marked as strengths vs needs improvement
- You're helping L&D professionals understand their workforce's skill gaps and plan training

When asked to:
- **Summarize insights**: Use get_summary_metrics, get_subdomain_coverage, get_top_skill_gaps, and compare_to_benchmark
- **Recommend what to learn**: Use get_top_skill_gaps, get_recommended_courses, and identify_critical_gaps
- **Create narratives**: Combine multiple data sources to tell a compelling story about strengths and gaps
- **Design learning plans**: Use analyze_learning_plan and segment_users_by_performance to create actionable, personalized plans with test-out rules

Be concise but insightful. Use specific numbers from the data. Format responses clearly with headers and bullet points when appropriate.

When creating learning plans with test-out rules:
- Lower performers (score <150) should complete full learning paths
- Mid-range performers (150-199) can test out of fundamentals
- High performers (200+) can test out of most content, focus on advanced topics
"""


def run_conversation(user_message: str, session_data: dict, chat_history: list) -> str:
    """Run a conversation turn with the AI agent."""
    
    if not OPENAI_API_KEY:
        return "Please add your OpenAI API key to settings.py to use the AI assistant."
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add chat history (last 10 messages for context)
    for msg in chat_history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": user_message})
    
    try:
        # First API call
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=AGENT_TOOLS,
            tool_choice="auto",
            temperature=0.7
        )
        
        assistant_message = response.choices[0].message
        
        # Handle tool calls iteratively
        while assistant_message.tool_calls:
            messages.append(assistant_message)
            
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                
                # Execute the tool
                tool_result = execute_tool(tool_name, tool_args, session_data)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
            
            # Get next response
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=AGENT_TOOLS,
                tool_choice="auto",
                temperature=0.7
            )
            assistant_message = response.choices[0].message
        
        return assistant_message.content
        
    except Exception as e:
        return f"Error communicating with AI: {str(e)}"

