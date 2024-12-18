import os
import json
import faiss
import numpy as np
from tempfile import NamedTemporaryFile
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import streamlit as st
from retrying import retry

def estimate_tokens(text):
    words = text.split()
    return int(len(words) / 0.75)  # Rough token estimate per document

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    import fitz  # PyMuPDF
    text = ""
    with fitz.open(pdf_file) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_txt(txt_file):
    """Extract text from a TXT file."""
    return txt_file.read().decode('utf-8')

def process_uploaded_files(uploaded_files):
    document_list = []
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == ".pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif file_extension == ".txt":
            text = extract_text_from_txt(uploaded_file)
        else:
            st.warning(f"Unsupported file format: {file_extension}. Skipping {uploaded_file.name}")
            continue

        document_list.append(Document(page_content=text))
    return document_list

def split_docs(documents, chunk_size=1000, chunk_overlap=20, max_tokens=2000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    valid_chunks = []
    for chunk in chunks:
        token_count = estimate_tokens(chunk.page_content)
        if token_count <= max_tokens:
            valid_chunks.append(chunk)
        else:
            st.warning(f"Skipping chunk due to token limit: {chunk.page_content[:100]}...")

    return valid_chunks

@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000)
def embed_documents_with_retry(texts, embeddings):
    return embeddings.encode(texts, show_progress_bar=True)

def create_faiss_vectorstore_from_docs(docs, embeddings, faiss_index_path, metadata_path):
    texts = [doc.page_content for doc in docs]
    embeddings_matrix = embed_documents_with_retry(texts, embeddings)

    embeddings_array = np.array(embeddings_matrix).astype('float32')
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    if os.path.exists(faiss_index_path):
        os.remove(faiss_index_path)
    faiss.write_index(index, faiss_index_path)

    metadata = [{'doc_id': i, 'content': doc.page_content} for i, doc in enumerate(docs)]
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

    return index

def load_faiss_index(faiss_index_path):
    try:
        return faiss.read_index(faiss_index_path)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None

def load_metadata(metadata_path):
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return []

def generate_answer_with_llm(query, retrieved_docs, model):
    context = "\n---\n".join([f"Document {i+1}:\n{doc['content']}" for i, doc in enumerate(retrieved_docs)])
    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""
            You are a helpful assistant. Use the context below to answer the question. If the answer cannot be found, say so.

            Context:
            {context}

            Question:
            {question}

            Answer:
        """
    )
    llm_chain = LLMChain(prompt=prompt, llm=model)
    return llm_chain.run({"question": query, "context": context})

def search_similar_docs_with_faiss_and_generate_answer(query, index, metadata, embeddings, model, k=3, max_context_tokens=2000):
    query_embedding = embeddings.encode([query], show_progress_bar=False)[0]
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)

    distances, indices = index.search(query_embedding, k)

    if indices[0].size > 0 and np.any(indices[0] != -1):
        retrieved_docs = [metadata[idx] for idx in indices[0] if idx != -1]
        context = "\n---\n".join([doc['content'] for doc in retrieved_docs])

        if len(context.split()) > max_context_tokens:
            st.warning("Context too large; summarizing...")
            context = "\n---\n".join(retrieved_docs[:max_context_tokens])

        return generate_answer_with_llm(query, retrieved_docs, model)

    return "No relevant documents found."

def main():
    st.title("Document-based Question Answering System")

    uploaded_files = st.file_uploader("Upload your documents (.pdf or .txt):", type=["pdf", "txt"], accept_multiple_files=True)

    faiss_index_path = "faiss_index.index"
    metadata_path = "metadata.json"

    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if uploaded_files:
        docs = process_uploaded_files(uploaded_files)
        chunks = split_docs(docs)

        if chunks:
            index = create_faiss_vectorstore_from_docs(chunks, embedding_model, faiss_index_path, metadata_path)
        else:
            index, metadata = None, None
    else:
        index = load_faiss_index(faiss_index_path)
        metadata = load_metadata(metadata_path)

    query = st.text_input("Enter your question:")

    if query:
        if index and metadata:
            answer = search_similar_docs_with_faiss_and_generate_answer(query, index, metadata, embedding_model, model="meta-llama/Llama-3.1-405B-Instruct")
            st.write(f"Answer: {answer}")
        else:
            st.warning("No documents available for querying. Please upload files.")

if __name__ == "__main__":
    main()
