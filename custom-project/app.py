# Import required packages.
import streamlit as st
import json
from pydantic import BaseModel
import openai
import time
client = openai.OpenAI(api_key=st.secrets["OpenAI_key"])

# Backend

#Creates the prompt for ChatGPT.
#
#   Args:
#        content: the content that needs metadata.
#    Returns:
#        A structured prompt for OpenAI.    

def prompt(content):
    return [
        {'role': 'system', 
         'content': f"""You are a project generator for an elearning website called Udacity.com. 
         You generate a variety of technical projects for learners based on their needs, including 
         detailed descriptions, rubrics, and project files to get started. You are able to tailor 
         the project to cover the skills and scenarios they care about. The project files must be real -- you 
         can't just return a string value called 'dataset.csv', or whatever. You need to either link out to a 
         dataset that can be used on the web, or you need to provide the code that can generate the code you need 
         to start - as well as some starter code with relevant hints, if needed, for how the learner should proceed."""},
        {'role': 'user', 
         'content': f"""Create a project based on this criterion: {content}"""}
            ]


#A format constraint that is passed to OpenAI.

response_format = {
    'type': 'json_schema',
    'json_schema': {
        'name': 'UdacityProject',
        'schema': {
            'type': 'object',
            'properties': {
                "Project Title": {
                    'type': 'string',
                    'description': 'The name of the project.'
                },
                "Project Description": {
                    'type': 'string',
                    'description': 'A 200-300 word description of the project, and the challenge the student will need to solve.'
                },
                'Skills': {
                    'type': 'array',
                    'items': {
                        'type': 'string'
                    },
                    'description': 'A list of skills the project teaches.'
                },
                'Difficulty': {
                    'type': 'string',
                    'description': 'One of the following: Introductory, Novice, Intermediate, Advanced, Expert.'
                },
                'Industry': {
                    'type': 'string',
                    'description': 'Which industry is the project relevant to?'
                },
                'Rubric': {
                    'type': 'array',
                    'description': 'A list of rubric items grouped by category, describing the criteria and how they relate to the project submission.',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'Category': {
                                'type': 'string',
                                'description': 'The category this rubric item falls under (e.g., Code Quality, Presentation, etc.).'
                            },
                            'Criteria': {
                                'type': 'array',
                                'description': 'A list of criteria for this category.',
                                'items': {
                                    'type': 'object',
                                    'properties': {
                                        'Description': {
                                            'type': 'string',
                                            'description': 'A detailed description of the criterion.'
                                        },
                                        'Relation to Project': {
                                            'type': 'string',
                                            'description': 'How this criterion relates to the overall project submission.'
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                'Suggestions': {
                    'type': 'array',
                    'description': 'A list of one-sentence suggestions for how to go above and beyond in the project.',
                    'items': {
                        'type': 'string'
                    }      
                },
                'Assets': {
                    'type': 'array',
                    'description': 'A list of project assets required to complete the project (e.g., datasets, code files, Jupyter notebooks).',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'Asset Type': {
                                'type': 'string',
                                'description': 'The type of asset (e.g., Public Dataset, Code File, Jupyter Notebook).'
                            },
                            'Data': {
                                'type': 'string',
                                'description': 'code that will generate a dataset of 10000 records that will have relevant fields for the project, and underlying distributions that are interesting and realistic.'
                            },
                            'Description': {
                                'type': 'string',
                                'description': 'A detailed description of the asset, including its purpose in the project.'
                            },
                            #'License': {
                            #    'type': 'string',
                            #    'description': 'The license information for public datasets.'
                            #},
                            'Starter Code': {
                                'type': 'string',
                                'description': '''Starter code to get started on the project. 
                                Either (1) actual code with documented headers where further coding is needed or 
                                (2) code to generate a relevant dataset.''',
                            },
                            'Documentation': {
                                'type': 'string',
                                'description': 'Any additional documentation needed to use the asset.'
                            },
                            #'url': {
                            #    'type': 'string',
                            #    'description': 'url to where this asset can be found on the web.'
                            #}
                        }
                    }
                }
            }
        }
    }
}


#Handles calling OpenAI.
#
#    Args:
#        message: The prompt generated by `prompt`.
#        model: The model being used (in this case, a specific snapshot of 4o that supports structured outputs.
#    
#    Returns:
#        OpenAI's results.

def chatgpt(message,model='gpt-4o-2024-08-06'):
    while True:
        try:
            result = client.beta.chat.completions.parse(
                      model=model,
                      messages=message,
                      response_format=response_format)

            return json.loads(result.choices[0].message.content)
        
        # Lazy exception error handling to back off and try the API again.
        except Exception as e:
            print(e)
            time.sleep(5)
# Frontend

# Title
st.title('Demo: Custom Project Generator')

# Input box for pasting content.
st.markdown('### Input')
sampleInput = """I want a basic project that teaches me how to do unsupervised machine clustering in a healthcare scenario. I want a realistic project, with a dataset that is a little messy that I'll need to clean up to show I have data wrangling skills."""
content = st.text_area('Input prompt and click "Generate Project"',sampleInput, height=150)

# Button to trigger job.
if st.button('Generate Project'):

	# Processing task.
	with st.spinner('Generating project...'):

		# Generate output.
		response = chatgpt(message=prompt(content))

		# Pretty formatting of results for output.
		json_formatted_str = json.dumps(response, indent=4)

		# Render output that was given.
		st.markdown('### Output')
		st.code(json_formatted_str,language="json")

		st.markdown('### Starter Code')
		st.markdown(response['Asset']['Documentation'])
		st.code(response['Asset'][0]['Starter Code'])
		st.markdown('### Data')
		st.code(response['Asset'][0]['Data'])
		
