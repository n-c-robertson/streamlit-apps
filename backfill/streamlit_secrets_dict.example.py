"""
Recommended Streamlit secrets for the J&J backfill app.

The application reads **only** ``streamlit.runtime.secrets.StreamlitSecrets`` (from
``.streamlit/secrets.toml`` locally or **App settings → Secrets** on Streamlit Cloud).
It does **not** import this module.

Use this file in two ways:

1. **Copy-paste TOML** — Use the ``PASTE_READY_SECRETS_TOML`` string below in
   ``.streamlit/secrets.toml`` or in the Cloud secrets editor.
2. **Local dictionary** — Copy this file to ``streamlit_secrets_dict.py`` (gitignored),
   fill ``STREAMLIT_SECRETS``, then transcribe values into TOML when deploying.

---

PASTE_READY_SECRETS_TOML
------------------------
"""

# Outer ''' allows TOML to use """ for multiline JSON (see TOML spec / Streamlit secrets).
PASTE_READY_SECRETS_TOML = r'''
# =============================================================================
# J&J backfill — Streamlit secrets (.streamlit/secrets.toml or Cloud UI)
# Replace every YOUR_* placeholder. Use either service_account_json OR
# [google_sheets.service_account], not both (service_account_json is easiest).
# =============================================================================

[google_sheets]
spreadsheet_id = "YOUR_SPREADSHEET_ID"
source_sheet_gid = 0
output_sheet_gid = 1198426749

# Paste the full contents of your Google Cloud service account JSON key file.
service_account_json = """
{
  "type": "service_account",
  "project_id": "YOUR_PROJECT_ID",
  "private_key_id": "YOUR_PRIVATE_KEY_ID",
  "private_key": "-----BEGIN PRIVATE KEY-----\nYOUR_KEY_BODY\n-----END PRIVATE KEY-----\n",
  "client_email": "YOUR_SERVICE_ACCOUNT@YOUR_PROJECT.iam.gserviceaccount.com",
  "client_id": "YOUR_CLIENT_ID",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/YOUR_SERVICE_ACCOUNT%40YOUR_PROJECT.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
"""

# --- Alternative: nested table instead of service_account_json (omit the block above) ---
# [google_sheets.service_account]
# type = "service_account"
# project_id = "YOUR_PROJECT_ID"
# private_key = """-----BEGIN PRIVATE KEY-----
# ...
# -----END PRIVATE KEY-----"""
# client_email = "YOUR_SERVICE_ACCOUNT@YOUR_PROJECT.iam.gserviceaccount.com"
# ...

# Optional: POST a JSON backup of the dataframe on each submit.
[google_form]
form_id = "YOUR_GOOGLE_FORM_ID"
entry_payload = "entry.1234567890"

# Required: gate the entire app until this matches (see app.py _require_password).
password = "YOUR_APP_PASSWORD"
'''

# Mirror of keys the app reads (placeholders only — safe to commit).
RECOMMENDED_STREAMLIT_SECRETS: dict = {
    "google_sheets": {
        "spreadsheet_id": "YOUR_SPREADSHEET_ID",
        "source_sheet_gid": 0,
        "output_sheet_gid": 1198426749,
        "service_account_json": "<entire JSON key as one string, OR omit and use nested service_account>",
        "service_account": {
            "type": "service_account",
            "project_id": "YOUR_PROJECT_ID",
            "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
            "client_email": "YOUR_SERVICE_ACCOUNT@YOUR_PROJECT.iam.gserviceaccount.com",
        },
    },
    "google_form": {
        "form_id": "YOUR_GOOGLE_FORM_ID",
        "entry_payload": "entry.1234567890",
    },
    "password": "YOUR_APP_PASSWORD",
}
