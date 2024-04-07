import streamlit as st
from langgraph_crag_Main import *
import os
from langchain.document_loaders import UnstructuredPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
import pprint
from streamlit_chat import message
import shutil
import pickle
import dill

## delete pdfs directory if it exists
if os.path.exists('pdfs'):
    shutil.rmtree('pdfs')


def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["conversation_history"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))


def get_retriever():
    pdf_dir = "pdfs"
    loader = PyPDFDirectoryLoader(pdf_dir, glob='**/*.pdf')
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=750, chunk_overlap=50)
    doc_splits = text_splitter.split_documents(docs)

    return doc_splits

def save_retriever(retriever,directory='retriever_storage'):
    ## if directory exists, delete it
    if os.path.exists(directory):
        shutil.rmtree(directory)
    
    os.makedirs(directory)
    retriever_path = os.path.join(directory, 'retriever.pkl')

    with open(retriever_path, 'wb') as f:
        dill.dump(retriever, f)
    

st.title("Langchain Multi-Agent Workflow")

# Initialize vector store
if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = None

# Add option of uploading multiple PDFs
st.write("Upload PDFs")
uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True)

retriever = None

if uploaded_files:
    # Save uploaded files to the "pdfs" directory
    pdf_dir = "pdfs"
    os.makedirs(pdf_dir, exist_ok=True)
    for file in uploaded_files:
        file_path = os.path.join(pdf_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.write("PDFs uploaded successfully")

    retriever = get_retriever()
    save_retriever(retriever)
    
    st.write("Vector store created/updated successfully")

if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = []


question = st.text_input("Ask a question", key='input')



if question:
    inputs = {"keys": {"question": question}}
    for output in app.stream(inputs,{
        "recursion_limit": 40,
    }):
        print(output)
        for key, value in output.items():
            pprint.pprint("\n---\n")
    answer = value["keys"]["generation"]
    st.session_state['conversation_history'].append(question)
    response = answer
    print(answer)
    st.session_state['generated'].append(response)

if st.session_state['generated']:
    display_conversation(st.session_state)

