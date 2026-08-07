# Project Rubric → Task List

A Streamlit app that takes a Udacity cd/nd key, loads the program from
classroom-content, fetches every project and its rubric, lets you review them,
then uses OpenAI to synthesize a sequential task list (rubric-first, supporting
content second) that you can refine in chat.

## Important: where rubrics live

Project (mentor-review) rubrics are **not** in `assessments-api`. They live in
`reviews-api` and are proxied by classroom-content's `Project.rubric` field.
This app fetches them via classroom-content's nested `Project.rubric`, with a
reviews-api REST fallback.

## Setup

```bash
cd project-rubric-tasklist
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit secrets.toml: OPENAI_API_KEY, UDACITY_JWT
streamlit run app.py
```

## Secrets (`.streamlit/secrets.toml`)

```toml
OPENAI_API_KEY = "sk-..."
UDACITY_JWT    = "eyJ..."   # staff or enrolled-learner JWT (Hoth)
```

## Endpoints (production)

- classroom-content GraphQL: `https://api.udacity.com/api/classroom-content/v1/graphql`
- reviews-api REST (rubric fallback): `https://api.udacity.com/api/reviews/v1/rubrics/{id}?projection=with_project_and_contents`

Auth header on every call: `Authorization: Bearer <UDACITY_JWT>`.

## How it works

1. Enter a cd/nd key in the sidebar.
2. The app queries classroom-content (`nanodegree(key:)` or `course(key:)`) for
   the full program tree, including every lesson's `project { rubric_id, rubric { ... } }`
   and the concept/atom text.
3. Review screen: expand each part/module/lesson; see each project's rubric items.
4. Click "Synthesize task list" on a project. The app builds a corpus from the
   rubric plus nearby lesson/concept text, embeds it with OpenAI, and prompts the
   LLM (rubric = primary signal, retrieved content = secondary) for an ordered
   task list whose steps cite rubric criteria.
5. Refine the result in chat; the LLM edits the task list in place.

## Notes / limitations

- classroom-content requires a JWT and checks enrollment or staff/editor role.
  Use a **staff JWT** to read any program without enrolling.
- Semantic search is a local in-memory index (OpenAI `text-embedding-3-small` +
  numpy cosine), rebuilt per session and cached by program key. It does not use
  the production Pinecone indexes that `ai-workers`/`marvin-api` use.
