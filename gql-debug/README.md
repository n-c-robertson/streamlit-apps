# gql-debug

Disposable Streamlit app for inspecting the classroom-content GraphQL
queries the assessment-creator runs. Delete this entire folder when done.

## Run locally

```bash
cd ~/streamlit-apps/gql-debug
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then paste a JWT in the sidebar.

## What it does

For an input `nd*` or `cd*` key + locale, it walks every query the
assessment-creator uses and renders the raw GraphQL response at each
step:

1. `components(key:)` cross-locale enumeration (not locale-gated).
2. `nanodegree(key:, locale:)` shallow shape (matches the old crosswalk).
3. `nanodegree(key:, locale:)` deep shape (mirrors current `query_nd_full`
   plus the `parts.branch.component.metadata` traversal so you can see
   what that path really returns).
4. For each part: `component(key:, locale:)` + `node(id:)`.

Each step shows HTTP status, GraphQL `errors[]`, and the raw `data`
section. Nothing is persisted.
