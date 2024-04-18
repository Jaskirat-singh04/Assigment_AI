import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate



GOOGLE_API_KEY = ''  # Your Google API key
prompt_template = """
    You are an AI assistant. Your primary goal is to provide support and assistance to users based on extracted context    
    Context : \n {context} \n    
    Question: \n {question} \n
    Answer:
    """
# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                google_api_key = GOOGLE_API_KEY,
                                content=text_chunks,
                                task_type="retrieval_document",
                                title="Document Title")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# Function to query the language model chain
def query_llm_chain(question, vector_store):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
  # retrive the relevant documents
 
    retriever = vector_store.as_retriever()
    relevant_docs = retriever.get_relevant_documents(question)
  # set up chain
    chain = LLMChain(llm = model, prompt=prompt, verbose = True)
  # run chain
    response = chain({"context": relevant_docs, "question": question}, return_only_outputs=True)
    return response["text"]
    # chain = LLMChain(llm=model, prompt=prompt_template, verbose=True)
    # response = chain({"context": vector_store.get_relevant_documents(question), "question": question}, return_only_outputs=True)
    # return response["text"]

# Set up Streamlit app
st.set_page_config("Automated QNA Generation")

# Sidebar for uploading PDF files
st.sidebar.header("Upload your Documents")
pdf_docs = st.sidebar.file_uploader("Upload PDF files", accept_multiple_files=True)

# Process uploaded PDF files
if pdf_docs:
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vector_store = get_vector_store(text_chunks)
    st.success("PDFs processed successfully!")

# Input field for user's question
user_question = st.text_input("Ask a Question")

# Query language model and display response
if user_question and pdf_docs:
    response = query_llm_chain(user_question, vector_store)
    st.write("Answer:", response)
elif user_question:
    st.warning("Please upload PDF files first.")
