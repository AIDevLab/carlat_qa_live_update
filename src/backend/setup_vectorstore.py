import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import re
import sys
import docx
import streamlit as st


# Get the absolute path to the main directory (my_project)
main_directory = os.path.abspath(os.path.dirname(__file__))
# Add main directory and its subdirectories to sys.path
sys.path.append(main_directory)
client = OpenAI(api_key=st.secrets["api_key"])



# Function to check if a file is loaded
def is_file_loaded(file):
    return file is not None


def get_doc_string(doc):
    doc = docx.Document(doc)
    # Iterate through paragraphs and append text to the entire content
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)

    # Print the entire content
    return '\n'.join(fullText)


def get_text_from_dir(dir_path):
    file = os.listdir(dir_path)
    raw_text = get_doc_string(os.path.join(dir_path, file[0]))
    return raw_text


def separte_speakers(raw_text):
    """
    get the interviewee quotes using regular expression

    Args:
    - raw_text (str): transcript


    Returns:
    - interviewee_quotes (list) : list of interviewee quotes
    - qa_formatted_text (list): list of Q/A pairs extracted from the transcript
    """
    formatted_text = []
    splitted_text = raw_text.split("\n") 

    for i in range(len(splitted_text)):
            chunk = splitted_text[i]
            # if the following chunk is a new speaker
            if re.match(r"[\w.]+(?:\s[\w.]+){0,2}\s*:", chunk):
                formatted_text.append(chunk)
            else:
                if (len(formatted_text) != 0):
                    # concat to the previous speaker's text 
                    formatted_text[-1] = formatted_text[-1] + "\n" + chunk

    qa_formatted_text = []
    interviewee_quotes = []
    for i in range(0, len(formatted_text)-2, 2):
        qa_formatted_text.append(formatted_text[i]+"\n"+formatted_text[i+1])
        interviewee_quotes.append(formatted_text[i+1])

    return qa_formatted_text, interviewee_quotes


def text_splitter(raw_text):
    """
    get Q/A pairs extracted from the transcript using regular expressions

    Args:
    - raw_text (str): transcript


    Returns:
    - interviewee_quotes (list) : list of interviewee quotes
    - qa_formatted_text (list): list of Q/A pairs extracted from the transcript
    """
    formatted_text = []
    splitted_text = raw_text.split("\n")

    for i in range(len(splitted_text)):
        chunk = splitted_text[i]
        # if the following chunk is a new speaker
        if re.match(r"[\w.]+(?:\s[\w.]+){0,2}\s*:", chunk):
            formatted_text.append(chunk)
        else:
            if (len(formatted_text) != 0):
                # concat to the previous speaker's text 
                formatted_text[-1] = formatted_text[-1] + "\n\n" + chunk

    qa_formatted_text = []
    for i in range(0, len(formatted_text)-2, 2):
        qa_formatted_text.append(formatted_text[i]+"\n"+formatted_text[i+1])

    return qa_formatted_text


def create_embedding(docs): 
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=st.secrets["api_key"])
    docsearch = FAISS.from_texts(docs, embeddings)
    return docsearch


if __name__ == '__main__':
    raw_text = get_text_from_dir('documents')
    docs = text_splitter(raw_text)
    docsearch = create_embedding(docs)
    docsearch.save_local('vectorstore')
