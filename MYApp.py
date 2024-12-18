import os
import tempfile
import numpy as np
import faiss
import requests
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
import fitz  # PyMuPDF for PDF handling
from docx import Document as DocxDocument
from pptx import Presentation
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.docstore.document import Document as LangchainDocument

# Load environment variables
load_dotenv()

# Set environment variables
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not REPLICATE_API_TOKEN or not HUGGINGFACE_API_TOKEN:
    raise ValueError("API tokens for Replicate and Hugging Face are not set in environment variables.")

# Initialize models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Define utility functions
def extract_text_from_file(uploaded_file):
    """Extract text from uploaded files based on file type."""
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension == "pdf":
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == "docx":
        return extract_text_from_docx(uploaded_file)
    elif file_extension == "pptx":
        return extract_text_from_pptx(uploaded_file)
    elif file_extension in ["xls", "xlsx"]:
        return extract_text_from_xlsx(uploaded_file)
    else:
        return None

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file_path = tmp_file.name

    text = ""
    try:
        with fitz.open(tmp_file_path) as doc:
            for page in doc:
                text += page.get_text()
    finally:
        os.unlink(tmp_file_path)  # Clean up the temporary file
    return text

def extract_text_from_docx(docx_file):
    """Extract text from a DOCX file."""
    doc = DocxDocument(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pptx(pptx_file):
    """Extract text from a PPTX file."""
    presentation = Presentation(pptx_file)
    text = ""
    for slide in presentation.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                text += shape.text + "\n"
    return text

def extract_text_from_xlsx(xlsx_file):
    """Extract text from an XLSX file."""
    df = pd.read_excel(xlsx_file)
    return df.to_string()

def llama3_1_generate(prompt, model="meta-llama/Llama-3.1-405B-Instruct", top_p=0.9, temperature=0.7, max_tokens=800):
    """Generate text using Llama 3.1."""
    input = {
        "top_p": top_p,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # Replace this mock with actual API call or inference logic for Llama 3.1
    return f"Generated response: {prompt}"

# Main Streamlit application
def main():
    st.title("Document-Based Question Answering System")

    # Upload documents
    uploaded_file = st.file_uploader("Upload your document (PDF, DOCX, PPTX, XLSX)", type=["pdf", "docx", "pptx", "xlsx"])

    # Ensure context is initialized
    context = ""
    if uploaded_file:
        context = extract_text_from_file(uploaded_file)
        if not context:
            st.error("Unsupported file type. Please upload a PDF, DOCX, PPTX, or XLSX file.")
            return
    else:
        context = "No document uploaded. Please upload a document to provide context."

    # Initialize chat history if not already done
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Question input
    question = st.text_input("Enter your question:")
    format_choice = st.radio("Select the format of the answer:", ["Bullet Points", "Summary", "Specific Length"])
    word_limit = st.number_input("Word limit (if applicable):", min_value=1, value=50) if format_choice == "Specific Length" else None

    if st.button("Get Answer") and question:
        # Prepare chat history for prompt
        history_context = " ".join(
            [f"Q: {entry['question']} A: {entry['answer']}" for entry in st.session_state.chat_history]
        )

        # Build the prompt
        prompt = f"Context: {context}\n\nChat History: {history_context}\n\nQuestion: {question}"

        # Format prompt based on user choice
        if format_choice == "Bullet Points":
            prompt += "\nPlease provide the answer as bullet points."
        elif format_choice == "Summary":
            prompt += "\nPlease summarize concisely."
        elif format_choice == "Specific Length":
            prompt += f"\nLimit the answer to {word_limit} words."

        # Generate the answer
        answer = llama3_1_generate(prompt)

        # Save chat history and display answer
        st.session_state.chat_history.append({"question": question, "answer": answer})
        st.markdown(f"**Q: {question}**")
        st.markdown(f"**A:** {answer}")

    # Display full chat history
    if st.session_state.chat_history:
        st.header("Chat History")
        for entry in st.session_state.chat_history:
            st.write(f"**Q:** {entry['question']}\n**A:** {entry['answer']}\n")

if __name__ == "__main__":
    main()
