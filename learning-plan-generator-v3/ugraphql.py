# Import packages.
import requests
import time

# Read in supporting functions from the same directory.
from keys import jwt_token

# GraphQL endpoints (only classroom-content is used for now).
classroomContentUrl = "https://api.udacity.com/api/classroom-content/v1/graphql"

# Headers for API calls. TODO: replace jwt_token with a service token that doesn't
# periodically expire.
headers = {
	"Authorization": f"Bearer {jwt_token()}",
	"Content-Type": "application/json"
	}

def fetchData(query, url, headers=headers):
    """
    Fetches data from a GraphQL API using a POST request.
    
    Args:
        query: The GraphQL query string.
        url: The API endpoint URL.
        headers: Headers for the request.
        
    Returns:
        dict: The JSON response from the API.
    """

    # If no headers are provided, assume empty headers.
    if headers is None:
        headers = {}

	# Attempt to fetch the data.    
    try:
    	# Request + time logging.
        print(f"Sending request to {url}...")  # Log the URL
        start_time = time.time()  # Start timing
        response = requests.post(url, headers=headers, json={'query': query}, timeout=10)  # Set a timeout
        elapsed_time = time.time() - start_time  # Calculate time taken for request
        print(f"Request completed in {elapsed_time:.2f} seconds.")
        
        # If there is an error, print error and return None.
        if response.status_code != 200:
            print(f"HTTP Error: {response.status_code}")
            print(f"Response Text: {response.text}")
            return None
        
        # Attempt to format the data into a JSON structure.
        try:
            data = response.json()
            return data

        # If it values, error and return None.
        except ValueError as ve:
            return None

    # If there is a time out, error and return None.
    except requests.exceptions.Timeout:
        print("The request timed out.")
        return None

    # If there is a requests exception, error and return None.
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the request: {e}")
        return None

def queryCourseStructure(key):
	"""
	Format query for gathering data on courses ("parts").

		Args:
			key: the program key of the content offering.

		Returns:
			Returns query pointed toward that desired program key.
	"""

	query = '''{{
	    part (key: "{0}", locale: "en-us") {{
	        title
	        key
	        is_default
	        id
	        is_optional
	        is_career
	        instructors {{
	            first_name
	            image_url
	        }}
	        is_extracurricular
	        modules {{
	            title
	            lessons {{
	                title
	                is_project_lesson
	                project {{
	                    title
	                }}
	                lab {{
	                    title
	                }}
	            }}
	        }}
	    }}
	}}
	'''.format(key)

	return query

def queryNanodegreeStructure(key):
	"""
	Format query for gathering data on Nanodegrees.

		Args:
			key: the program key of the content offering.

		Returns:
			Returns query pointed toward that desired program key.
	"""
	query = """{{
	  nanodegrees (key: "{0}", locale: "en-us") {{
		title
		key
		is_default
		version
		locale
		id
		parts {{
		  title
		  key
		  id
		  is_optional
		  is_career
		  instructors {{
			first_name
			image_url
		  }}
		  is_extracurricular
		  modules {{
		   title
		   lessons {{
			title
			is_project_lesson
			project {{
			  title
			}}
			lab {{
			  title
			}}
		  }}
		}}
	  }}
	}}
	}}  
	""".format(key)
	return query

def queryFreeCourseStructure(key):
	"""
	Format query for gathering data on free courses.

		Args:
			key: the program key of the content offering.

		Returns:
			Returns query pointed toward that desired program key.
	"""
	query = """{{
		  courses(key: "{0}") {{
		    id
		    key
		    title
		    summary
		    is_default
		    lessons {{
		      id
		      key
		      title
		      duration
		      project {{
		        id
		        title
		      }}
		      lab {{
		        id
		        title
		      }}
		    }}
		  }}
		}}""".format(key)
	return query

