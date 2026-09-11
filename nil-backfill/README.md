# nil-backfill app

A Streamlit app that lets an operator run the three Workera nil-backfill operations
while the owner is out on leave. Consolidates the three `nil_backfill_*.ipynb`
notebooks into one UI with a **preview-then-confirm** safety gate before any
production PATCH is sent.

## Operations

| Operation | Endpoint | Modes | Notes |
|---|---|---|---|
| Score Override | `PATCH /attempts/{id}/score` | single, CSV | Overwrites `attempts.score` (0–300) and sets `score_overridden` so Workera sync can't revert. One request per attempt. |
| Step-ID Backfill | `PATCH /attempts/step-id/batch` | single | Only writes when `step_id IS NULL` — never overwrites. |
| User-ID Backfill | `PATCH /attempts/user-id/batch` | single, CSV | Only writes when `user_id IS NULL` — never overwrites. Batched at 1000. |

All operations authenticate via the `_jwt` cookie (not the `Authorization`
header) and send a browser `User-Agent` so Cloudflare doesn't block the
request.

## Run locally

```bash
cd nil-backfill-app
pip install -r requirements.txt
streamlit run app.py
```

## Secrets

App-level config lives in `.streamlit/secrets.toml` (gitignored). Copy the
template:

```bash
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
```

The only key is `default_environment` (`"prod"` or `"staging"`), which sets the
default of the environment radio button.

**Staff JWTs are NOT stored in secrets.** They are entered per-session in the
sidebar (`type="password"`) because they are user-specific and short-lived.

### Streamlit Cloud

When deploying to Streamlit Cloud, paste the same key/values into
**app settings → Secrets** (e.g. `default_environment = "prod"`). Operators
still paste their own JWT into the sidebar at run time.

## Safety model

1. Operator picks environment + operation + inputs.
2. **Preview** validates every row (UUID format, score range, CSV columns) and
   shows exactly what will be sent, with a red warning banner. No network call
   is made yet.
3. **Confirm & Run** is only enabled after a clean preview. It sends the
   PATCH(es), shows a progress bar, then a results table + CSV download.
4. Score Override gets an extra-strong warning — it is the only operation that
   overwrites existing data and is not reversible.

## Auth / RBAC

The app itself does no auth. The Workera API enforces staff / `company_admin`
permissions via the JWT. Anyone with a valid staff JWT can run any operation.
