import streamlit as st
import requests
import json
from io import BytesIO
from fpdf import FPDF

st.set_page_config(page_title="Classroom Content Fetcher", layout="wide")

# GraphQL query
CLASSROOM_CONTENT_QUERY = """
query classroomContent($key: String!) {
  node(key: $key) {
    ... on Nanodegree {
      key
      locale
      version
      semantic_type
      title
      parts {
        key
        locale
        version
        semantic_type
        title
        modules {
          key
          locale
          version
          semantic_type
          title
          lessons {
            key
            id
            locale
            version
            semantic_type
            title
            concepts {
              key
              locale
              version
              semantic_type
              title
              progress_key
              atoms {
                ... on TextAtom {
                  key
                  locale
                  version
                  semantic_type
                  title
                  text
                }
                ... on VideoAtom {
                  key
                  locale
                  version
                  semantic_type
                  title
                  video {
                    vtt_url
                    transcript
                  }
                }
                ... on RadioQuizAtom {
                  semantic_type
                  question {
                    prompt
                    answers {
                      is_correct
                      text
                    }
                  }
                }
                ... on CheckboxQuizAtom {
                  semantic_type
                  question {
                    prompt
                    correct_feedback
                    answers {
                      is_correct
                      text
                    }
                  }
                }
                ... on MatchingQuizAtom {
                  semantic_type
                  question {
                    answers_label
                    concepts_label
                    concepts {
                      text
                      correct_answer {
                        text
                      }
                    }
                    complex_prompt {
                      text
                    }
                    answers {
                      text
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    ... on Course {
      key
      locale
      version
      semantic_type
      title
      lessons {
        key
        locale
        version
        semantic_type
        title
        concepts {
          key
          locale
          version
          semantic_type
          title
          progress_key
          atoms {
            ... on TextAtom {
              key
              locale
              version
              semantic_type
              title
              text
            }
            ... on VideoAtom {
              key
              locale
              version
              semantic_type
              title
              video {
                vtt_url
                transcript
              }
            }
            ... on RadioQuizAtom {
              semantic_type
              question {
                prompt
                answers {
                  is_correct
                  text
                }
              }
            }
            ... on CheckboxQuizAtom {
              semantic_type
              question {
                prompt
                correct_feedback
                answers {
                  is_correct
                  text
                }
              }
            }
            ... on MatchingQuizAtom {
              semantic_type
              question {
                answers_label
                concepts_label
                concepts {
                  text
                  correct_answer {
                    text
                  }
                }
                complex_prompt {
                  text
                }
                answers {
                  text
                }
              }
            }
          }
        }
      }
    }
    ... on Part {
      key
      locale
      version
      semantic_type
      title
      modules {
        key
        locale
        version
        semantic_type
        title
        lessons {
          key
          locale
          version
          semantic_type
          title
          concepts {
            key
            locale
            version
            semantic_type
            title
            progress_key
            atoms {
              ... on TextAtom {
                key
                locale
                version
                semantic_type
                title
                text
              }
              ... on VideoAtom {
                key
                locale
                version
                semantic_type
                title
                video {
                  vtt_url
                  transcript
                }
              }
              ... on RadioQuizAtom {
                semantic_type
                question {
                  prompt
                  answers {
                    is_correct
                    text
                  }
                }
              }
              ... on CheckboxQuizAtom {
                semantic_type
                question {
                  prompt
                  correct_feedback
                  answers {
                    is_correct
                    text
                  }
                }
              }
              ... on MatchingQuizAtom {
                semantic_type
                question {
                  answers_label
                  concepts_label
                  concepts {
                    text
                    correct_answer {
                      text
                    }
                  }
                  complex_prompt {
                    text
                  }
                  answers {
                    text
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
"""


def query_node(program_key: str, jwt: str) -> dict:
    """Fetch classroom content for a program key."""
    payload = {
        "query": CLASSROOM_CONTENT_QUERY,
        "variables": {"key": program_key}
    }
    headers = {
        'Authorization': f'Bearer {jwt}',
        'Content-Type': 'application/json'
    }
    response = requests.post(
        "https://api.udacity.com/api/classroom-content/v1/graphql",
        headers=headers,
        json=payload
    )
    return response.json()['data']['node']


def content_to_pdf_bytes(content: dict, program_key: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"Classroom Content Export: {program_key}", ln=True)
    pdf.ln(4)

    pdf.set_font("Courier", size=8)

    content_str = json.dumps(content, indent=2, ensure_ascii=False)
    safe_text = content_str.encode("latin-1", "replace").decode("latin-1")

    pdf.multi_cell(0, 4.5, safe_text)

    # ✅ fpdf2 >= 2.2
    return bytes(pdf.output())




# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    .main-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        color: #00d9ff;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
    }
    .subtitle {
        font-family: 'JetBrains Mono', monospace;
        color: #8892b0;
        text-align: center;
        margin-bottom: 2rem;
    }
    div[data-testid="stTextInput"] label {
        color: #ccd6f6 !important;
        font-family: 'JetBrains Mono', monospace;
    }
    .stButton > button {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    .action-buttons {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
</style>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">📚 Classroom Content Fetcher</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Query Udacity classroom content via GraphQL</p>', unsafe_allow_html=True)

# Input fields
col1, col2 = st.columns(2)
with col1:
    jwt_key = st.text_input("🔑 JWT Key", type="password", placeholder="Enter your JWT token")
with col2:
    program_key = st.text_input("📦 Program Key", placeholder="e.g., nd001")

# Fetch button
fetch_clicked = st.button("🚀 Fetch Content", type="primary", use_container_width=True)

# Session state for storing results
if 'content' not in st.session_state:
    st.session_state.content = None
if 'program_key' not in st.session_state:
    st.session_state.program_key = None

# Fetch content
if fetch_clicked:
    if not jwt_key or not program_key:
        st.error("Please provide both JWT key and Program key.")
    else:
        with st.spinner("Fetching classroom content..."):
            try:
                content = query_node(program_key, jwt_key)
                st.session_state.content = content
                st.session_state.program_key = program_key
                st.success("Content fetched successfully!")
            except Exception as e:
                st.error(f"Error fetching content: {str(e)}")

# Display results and action buttons
if st.session_state.content:
    content = st.session_state.content
    content_json = json.dumps(content, indent=2)
    
    st.markdown("---")
    
    # Action buttons at the top
    btn_col1, btn_col2, btn_col3, _ = st.columns([1, 1, 1, 3])
    
    
    with btn_col1:
        # Download JSON
        st.download_button(
            label="📥 Download .JSON",
            data=content_json,
            file_name=f"{st.session_state.program_key}_content.json",
            mime="application/json",
            use_container_width=True
        )
    
    with btn_col2:
        # Download PDF
        pdf_bytes = content_to_pdf_bytes(content, st.session_state.program_key)
        st.download_button(
            label="📄 Download PDF",
            data=pdf_bytes,
            file_name=f"{st.session_state.program_key}_content.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    
    st.markdown("### 📄 Content Result")
    
    # Display in code block
    st.code(content_json, language="json")

