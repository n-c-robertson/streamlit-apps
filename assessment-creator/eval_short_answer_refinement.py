"""Generate a local before/after comparison for SHORT_ANSWER refinement.

Run from this directory:
    python eval_short_answer_refinement.py

The script makes no mutations outside the generated JSON report. It uses the
same configured model as assessment generation and requires the usual API key.
"""

import json
import os
import tomllib
from pathlib import Path

from openai import OpenAI
import prompts


SECRETS_PATH = Path(__file__).parent / ".streamlit" / "secrets.toml"


SEEDS = [
    ("Prompt Engineering", "Explain prompt engineering and why it matters for LLM outputs."),
    ("SQL Aggregation", "Explain how GROUP BY and aggregate functions work together in SQL."),
    ("Model Evaluation", "Explain the difference between precision and recall and when each matters."),
    ("Python Testing", "Explain why unit tests are useful and what makes a good unit test."),
]


def call(messages):
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=messages,
    )
    return json.loads(response.choices[0].message.content.replace("```json", "").replace("```", ""))


def main():
    global client, model
    if not SECRETS_PATH.exists():
        raise SystemExit(
            f"Create {SECRETS_PATH} from .streamlit/secrets.toml.example "
            "and add your local credentials."
        )
    with SECRETS_PATH.open("rb") as secrets_file:
        secrets = tomllib.load(secrets_file)
    api_key = secrets.get("openai_api_key")
    if not api_key:
        raise SystemExit(f"Add openai_api_key to {SECRETS_PATH}.")
    model = secrets.get("openai_model", "gpt-4o")
    client = OpenAI(api_key=api_key)

    report = []
    for skill, content in SEEDS:
        old = call([
            {
                "role": "system",
                "content": (
                    "Generate a SHORT_ANSWER assessment item. Return JSON with "
                    "question.content and question.rubric. The rubric should have "
                    "2-4 criteria with names, descriptions, points, and a "
                    "passingThreshold."
                ),
            },
            {"role": "user", "content": f"Skill: {skill}\nQuestion: {content}"},
        ])
        old_question = old.get("question", old)
        refined = call(prompts.get_short_answer_refinement_prompt(
            skill,
            [],
            {
                "content": old_question.get("content", content),
                "rubric": old_question.get("rubric"),
            },
        ))
        report.append({
            "skill": skill,
            "before": old_question,
            "after": refined.get("question", refined),
        })

    output = Path("short_answer_refinement_eval.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    for item in report:
        print(f"\n## {item['skill']}\nBEFORE: {item['before'].get('content')}\nAFTER:  {item['after'].get('content')}")


if __name__ == "__main__":
    main()
