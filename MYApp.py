import os
import json
import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, TextLoader
from tempfile import NamedTemporaryFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Set environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Hugging Face API Token is not set in environment variables.")

# Initialize Llama 3.1 (Hugging Face Model)
model_name = "meta-llama/Llama-3.1-405B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)

# Initialize Sentence Transformer Model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use another model here

# Token estimation function
def estimate_tokens(text):
    words = text.split()
    return int(len(words) / 0.75)  # Rough token estimate per document

# Process Uploaded Files
def process_uploaded_files(uploaded_files):
    """Process and load documents from uploaded files."""
    document_list = []
    for uploaded_file in uploaded_files:
        # Save the uploaded file to a temporary file
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        with NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name

        # Choose loader based on file type
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == ".txt":
            loader = TextLoader(temp_file_path, encoding="utf-8")  # Use UTF-8 encoding for text files
        else:
            st.warning(f"Unsupported file format: {file_extension}. Skipping {uploaded_file.name}")
            continue

        # Load documents and extend the document list
        try:
            documents = loader.load()
            document_list.extend(documents)
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
            continue

        # Optionally clean up the temporary file after processing
        os.remove(temp_file_path)

    return document_list

# Split Documents into Chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20, max_tokens=2000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    # Ensure chunks do not exceed token limit
    valid_chunks = []
    for chunk in chunks:
        token_count = estimate_tokens(chunk.page_content)
        if token_count <= max_tokens:
            valid_chunks.append(chunk)
        else:
            st.warning(f"Skipping chunk due to token limit: {chunk.page_content[:100]}... (tokens: {token_count})")

    return valid_chunks

# Generate embeddings for documents using Sentence Transformers
def generate_embeddings(documents):
    embeddings = embedding_model.encode([doc.page_content for doc in documents], convert_to_tensor=True)
    return embeddings.cpu().detach().numpy()

# Create FAISS index from documents
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Embedding dimension
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance (Euclidean)
    index.add(embeddings)
    return index

# Generate Answer with Llama 3.1
def generate_answer_with_llama(query, retrieved_docs):
    context = "\n---\n".join([f"Document {i+1}:\n{doc['content']}" for i, doc in enumerate(retrieved_docs)])

    # Tokenize input for Llama
    inputs = tokenizer.encode(query + context, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=3)

    # Decode the generated response
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Create or update FAISS index with new documents
def update_index_with_new_files(uploaded_files, faiss_index_path, metadata_path):
    if uploaded_files:
        # Process newly uploaded documents
        new_docs = process_uploaded_files(uploaded_files)
        new_docs = split_docs(new_docs, chunk_size=1000, chunk_overlap=20, max_tokens=2000)

        if new_docs:
            # Generate embeddings for new documents
            new_embeddings = generate_embeddings(new_docs)

            # Create or update FAISS index
            index = create_faiss_index(new_embeddings)

            # Load existing metadata if it exists
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    existing_metadata = json.load(f)
            else:
                existing_metadata = []

            # Update metadata with new documents
            metadata = [{'doc_id': len(existing_metadata) + i, 'content': doc.page_content}
                        for i, doc in enumerate(new_docs)]
            existing_metadata.extend(metadata)

            # Save updated FAISS index and metadata
            faiss.write_index(index, faiss_index_path)
            with open(metadata_path, 'w') as f:
                json.dump(existing_metadata, f)

            return index, existing_metadata
    return None, None

# Perform FAISS search and generate an answer
def search_similar_docs_with_faiss_and_generate_answer(query, faiss_index, metadata, embeddings, k=3, max_context_tokens=2000):
    query_embedding = embedding_model.encode([query], convert_to_tensor=True).cpu().detach().numpy()
    distances, indices = faiss_index.search(query_embedding, k)

    if indices[0].size > 0 and np.any(indices[0] != -1):  # Ensure valid retrieval
        retrieved_docs = [(metadata[idx], score) for idx, score in zip(indices[0], distances[0])]

        # Sort documents by relevance (ascending distance means higher relevance)
        retrieved_docs.sort(key=lambda x: x[1])

        # Generate answer with Llama using retrieved documents as context
        answer = generate_answer_with_llama(query, [doc for doc, _ in retrieved_docs])
        return answer
    else:
        return "No relevant documents found for the query."

# Main Streamlit Interface
def main():
    st.title("Document-based Question Answering System")

    # Upload Documents
    uploaded_files = st.file_uploader(
        "Upload your documents (.pdf or .txt):",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    # FAISS index and metadata paths
    faiss_index_path = "faiss_index.index"
    metadata_path = "metadata.json"

    # Check if FAISS index exists and load or create
    if os.path.exists(faiss_index_path) and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        faiss_index = faiss.read_index(faiss_index_path)
    else:
        faiss_index = None
        metadata = None

    # Update FAISS index and metadata if new files are uploaded
    new_index, new_metadata = update_index_with_new_files(uploaded_files, faiss_index_path, metadata_path)
    if new_index and new_metadata:
        faiss_index = new_index
        metadata = new_metadata

    # Query System
    query = st.text_input("Enter your question:")

    if query:
        if faiss_index and metadata:
            answer = search_similar_docs_with_faiss_and_generate_answer(query, faiss_index, metadata, embedding_model)
            st.write(f"### Generated Answer: {answer}")
        else:
            st.write("No documents available for querying. Please upload files.")

if __name__ == "__main__":
    main()
