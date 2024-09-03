import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import docx
import nltk 
import sys
nltk.download('punkt')
import streamlit as st

# sys.path.append("C:\\Users\\dell\\Documents\\GitHub\\carlat-qa-editor-dev-env\\src")
# sys.path.append("C:\\Users\\dell\\Documents\\GitHub\\carlat-qa-editor-dev-env\\src\\backend")

sys.path.append("/mount/src/carlat_qa_live_update/src")
sys.path.append("/mount/src/carlat_qa_live_update/src/backend")

# Get the absolute path to the main directory (my_project)
main_directory = os.path.abspath(os.path.dirname(__file__))
# Add main directory and its subdirectories to sys.path
sys.path.append(main_directory)

from setup_vectorstore import separte_speakers
client = OpenAI(api_key=st.secrets["api_key"])



def load_vectorstore(): 
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=st.secrets["api_key"])
    docsearch = FAISS.load_local("//mount//src//caralt-qa-editor-adl//src//vectorstore", embeddings, allow_dangerous_deserialization= True)
    return docsearch


def similarity_search(query, docsearch, k=4):
    results = docsearch.similarity_search(query, k=k)

    docs = [doc.page_content for doc in results]
    return docs


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


def similarity_search_score(query, docsearch, threshold=0.5):
    """
    Perform similarity search and select documents based on a threshold.

    Args:
        query: The query for similarity search.
        docsearch: The document search object.
        threshold (float): The similarity threshold for document selection.

    Returns:
        List[str]: Selected document contents.
    """
    results = docsearch.similarity_search_with_score(query, k=7)
    docs = []
    scores = []
    min_val, max_val = float('inf'), float('-inf')  # Initialize min and max

    for doc, score in results:
        docs.append(doc.page_content)
        scores.append(score)
        min_val = min(min_val, score)
        max_val = max(max_val, score)

    # Normalize scores using list comprehension
    normalized_scores = [(score - min_val) / (max_val - min_val) for score in scores]

    # Select documents based on the threshold
    selected_docs = [doc for doc, score in zip(docs, normalized_scores) if score < threshold]

    return selected_docs


def get_quotes(topic):
    vect_store = load_vectorstore()
    # get the document chunks that sementically relate to the topic.
    docs = similarity_search_score(topic, vect_store, threshold=0.70)

    # remove the interviewer's quotes
    _, interviewee_quotes = separte_speakers("\n".join(docs))
    return  interviewee_quotes