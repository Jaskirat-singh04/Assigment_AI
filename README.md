# AI_Assignment

## Automated QNA Generation
This project is an AI-powered question answering system that generates answers to user questions based on the content of uploaded PDF documents.

## Features
PDF Document Ingestion: The system can process multiple PDF documents, extracting the text content from them.
Text Chunking: The extracted text is split into smaller, manageable chunks to improve the efficiency of the question answering process.
Vector Store Creation: The text chunks are used to create a vector store, which allows for efficient retrieval of relevant information to answer user questions.

## Language Model Integration: The system leverages a large language model (ChatGoogleGenerativeAI) to generate relevant and coherent answers to user questions.

## Streamlit-based UI: The application is built using the Streamlit framework, providing a user-friendly interface for uploading PDF documents and asking questions.

## Usage
Upload PDF Documents: Use the sidebar file uploader to upload one or more PDF documents that you want the system to process.
Ask a Question: In the main input field, enter a question that you would like the system to answer based on the uploaded PDF content.
View the Response: The system will process the question, retrieve the relevant information from the vector store, and display the generated answer.

## Dependencies
The project relies on the following Python libraries:

PyPDF2: for extracting text from PDF documents
tiktoken: for text tokenization and chunking
Gemini : for language model inference
streamlit: for the user interface
faiss: for creating and querying the vector store

To install the required dependencies, you can use the following command:

```pip install -r requirements.txt```

Configuration
The application requires a Google API key to be set as an environment variable (GOOGLE_API_KEY). Make sure to obtain a valid API key and set it accordingly.

Future Improvements
Multilingual Support: Extend the system to handle documents and questions in multiple languages.
Summarization: Provide a summarized version of the answer, in addition to the full response.
Contextual Awareness: Improve the language model's understanding of the document context to generate more relevant and coherent answers.
Iterative Refinement: Allow users to provide feedback on the generated answers and continuously improve the system's performance.
