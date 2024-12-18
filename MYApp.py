import os
import tempfile
import faiss
import requests
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
import replicate
import fitz  # PyMuPDF for PDF handling

# Configure paths and models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Sentence Transformer model for embeddings
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")  # Ensure this environment variable is set securely

# Initialize models
embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Replicate Llama 2 model integration
def llama2_generate(prompt, top_p=1, temperature=0.75, max_new_tokens=800):
    replicate.Client(api_token=REPLICATE_API_TOKEN)
    input = {
        "top_p": top_p,
        "prompt": prompt,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens
    }
    response = ""
    for event in replicate.stream("meta/llama-2-7b-chat", input=input):
        response += event
    return response

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(pdf_file) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to fetch text from a URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL content: {e}")
        return ""

# Main Streamlit Interface
def main():
    # Streamlit UI
    st.title("Document-based QA with Llama 2 and FAISS")
    st.sidebar.header("Upload Documents or Provide a Web Link")

    uploaded_files = st.sidebar.file_uploader("Upload files (PDF/TXT)", type=["txt", "pdf"], accept_multiple_files=True)
    web_link = st.sidebar.text_input("Or provide a web link (URL):", "")

    documents = []

    # Handle uploaded files
    if uploaded_files:
        temp_dir = tempfile.TemporaryDirectory()

        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir.name, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            # Process text or PDF files
            if uploaded_file.name.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    documents.append(f.read())
            elif uploaded_file.name.endswith(".pdf"):
                documents.append(extract_text_from_pdf(file_path))

        temp_dir.cleanup()

    # Handle URL input
    if web_link:
        documents.append(extract_text_from_url(web_link))

    if documents:
        # Split documents into chunks
        chunk_size = 500
        chunks = [doc[i:i+chunk_size] for doc in documents for i in range(0, len(doc), chunk_size)]

        # Generate embeddings
        embeddings = embeddings_model.embed_documents(chunks)

        # Create FAISS index
        dimension = len(embeddings[0])
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings)

        # Wrap FAISS with LangChain Vectorstore
        vector_store = FAISS(embeddings, chunks, faiss_index)

        # Set up RetrievalQA chain
        retriever = vector_store.as_retriever()
        prompt_template = PromptTemplate(template="Answer the question based on the provided context: {context}\n\nQuestion: {question}", input_variables=["context", "question"])

        # QA Interface
        st.header("Ask a Question")
        question = st.text_input("Your Question:")

        if st.button("Get Answer") and question:
            # Retrieve context and get answer from Llama 2
            context = retriever.retrieve(question)
            full_prompt = f"Answer the question based on the provided context: {context}\n\nQuestion: {question}"
            answer = llama2_generate(full_prompt)
            st.write("Answer:", answer)

    else:
        st.write("Upload documents or provide a web link to get started.")

if __name__ == "__main__":
    main()