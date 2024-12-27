# Import packages.
import pandas as pd
from io import StringIO
import PyPDF2

def extract_text_from_pdf(pdf_file):
	"""
	Extract the text from a PDF file.

		Args:
			pdf_file: a PDF file.

		Returns:
			The text elements from the PDF file.
	"""
	pdf_reader = PyPDF2.PdfReader(pdf_file)
	text = ""
	for page_num in range(len(pdf_reader.pages)):
		text += pdf_reader.pages[page_num].extract_text()
	return text

def extract_text_from_docx(docx_file):
	"""
	Extract the text for a docx file.

		Args:
			docx_file: a docx file.

		Returns:
			The text from a docx file.
	"""
	doc = Document(docx_file)
	string = "\n".join([para.text for para in doc.paragraphs])
	return string

def extract_text_from_txt(txt_file):
	"""
	Extract the text for a txt file.

		Args:
			docx_file: a txt file.

		Returns:
			The text from a txt file.
	"""
	string = txt_file.read().decode("utf-8")
	return string

def extract_data_from_csv(csv_file):
	"""
	Extract the text for a csv file.

		Args:
			docx_file: a csv file.

		Returns:
			The text from a csv file.
	"""
	csv = pd.read_csv(csv_file)
	string = csv.to_string()
	return string

# Helper function to extract data from Excel
def extract_data_from_excel(excel_file):
	"""
	Extract the text for an excel file.

		Args:
			docx_file: an excel file.

		Returns:
			The text from an excel file.
	"""
	df = pd.read_excel(excel_file)
	string = df.to_string()
	return string


def process_file(uploaded_file):
	"""
	Extract text after determining the right way to get
	text from the file type.

		Args:
			uploaded_file: The file that was uploaded.

		Returns:
			The text from the file.
	"""
	file_type = uploaded_file.name.split('.')[-1]
	
	if file_type == 'pdf':
		return extract_text_from_pdf(uploaded_file)
	elif file_type == 'docx':
		return extract_text_from_docx(uploaded_file)
	elif file_type == 'txt':
		return extract_text_from_txt(uploaded_file)
	elif file_type == 'csv':
		return extract_data_from_csv(uploaded_file)
	elif file_type == 'xlsx':
		return extract_data_from_excel(uploaded_file)
	else:
		return None

def upload_and_merge_files(files):

	"""
	Extract text from multiple files across different file types.

		Args:
			files: A list of files that were uploaded.

		Returns:
			The text from all of the files.
	"""

	if files is not None:
		unified_file = ''
		for f in files:
			file = process_file(f)
			unified_file += file

		return unified_file
	else:
		return 'No Support Assets Provided.'

def df_to_string_csv(df):
	"""
	Convert a DataFrame to a string. This is most often used when we need to save a DataFrame as
	context for OpenAI in a way that it can make sense of the data.

		Args:
			df: a pandas DataFrame.

		Returns:
			A string representation of the pandas DataFrame.
	"""
	csv_string = StringIO()
	df.to_csv(csv_string, index=False)

	# Get the string value
	csv_result = csv_string.getvalue()

	return csv_result