import PyPDF2
import re
from nltk.tokenize import sent_tokenize
import nltk

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text



############################################
#           Data Preprocessing 
##############################################

def clean_text(text):
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,]', '', text)
    return text


