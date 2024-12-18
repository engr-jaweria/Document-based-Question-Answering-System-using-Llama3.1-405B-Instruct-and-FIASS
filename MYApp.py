import os
import numpy as np
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, PDFMinerLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# FAISS Index and metadata loading and saving
def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def load_metadata(metadata_path):
    with open(metadata_path, "rb") as f:
        return pickle.load(f)

def save_faiss_index(index, index_path):
    faiss.write_index(index, index_path)

def save_metadata(metadata, metadata_path):
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

def update_index_with_new_files(uploaded_files, faiss_index_path, metadata_path, embeddings):
    documents = []
    metadata = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            loader = PDFMinerLoader(uploaded_file)
        elif uploaded_file.type == "text/plain":
            loader = TextLoader(uploaded_file)
        else:
            st.error("Unsupported file format!")
            continue
        document = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(document)
        documents.extend(docs)
        metadata.extend([{"content": doc.page_content} for doc in docs])
    
    if documents:
        # Create embeddings for the documents
        doc_embeddings = embeddings.embed_documents([doc.page_content for doc in documents])
        # Create a FAISS index
        index = faiss.IndexFlatL2(len(doc_embeddings[0]))
        faiss.normalize_L2(doc_embeddings)
        index.add(np.array(doc_embeddings).astype('float32'))
        save_faiss_index(index, faiss_index_path)
        save_metadata(metadata, metadata_path)
        return index, metadata
    else:
        return None, None

# Summarize Long Context to Fit Token Limit
def summarize_long_context(retrieved_docs, model, max_context_tokens=2000):
    """Summarize documents to fit within the token limit."""
    summaries = []
    prompt = PromptTemplate(
        input_variables=["content"],
        template="""Summarize the following document to capture the key points in 200 words or less:

        {content}

        Summary:"""
    )
    for doc in retrieved_docs:
        context = prompt.format(content=doc['content'])
        summaries.append(model(context))
    return "\n".join(summaries)

# Perform Retrieval and Generate Answer
def search_similar_docs_with_faiss_and_generate_answer(query, index, metadata, embeddings, model, k=3, max_context_tokens=2000):
    """Retrieve and synthesize information from multiple documents, prioritizing by relevance score."""
    query_embedding = embeddings.embed_query(query)
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)

    # Perform FAISS search
    distances, indices = index.search(query_embedding, k)

    if indices[0].size > 0 and np.any(indices[0] != -1):  # Ensure valid retrieval
        # Pair retrieved document indices with their relevance scores (distances)
        retrieved_docs = []
        for idx, score in zip(indices[0], distances[0]):
            if idx != -1:
                retrieved_docs.append((metadata[idx], score))

        # Deduplicate the retrieved documents
        unique_docs = {doc['content']: (doc, score) for doc, score in retrieved_docs}.values()
        retrieved_docs = list(unique_docs)

        # Sort documents by relevance (ascending distance means higher relevance)
        retrieved_docs.sort(key=lambda x: x[1])

        st.write(f"Retrieved {len(retrieved_docs)} document(s) for the query, sorted by relevance:")
        for i, (doc, score) in enumerate(retrieved_docs):  # Display snippets and scores
            st.write(f"Document {i + 1} | Score: {score:.4f} | Snippet: {doc['content'][:200]}...")

        # Concatenate the most relevant context for the LLM
        context = "\n---\n".join(
            [f"Document {i + 1} (Score: {score:.4f}):\n{doc['content']}" for i, (doc, score) in enumerate(retrieved_docs)]
        )

        # Summarize context if it exceeds token limit
        if len(context.split()) > max_context_tokens:
            st.write("Context too large; summarizing the top documents...")
            context = summarize_long_context([doc for doc, _ in retrieved_docs], model, max_context_tokens)

        if not context.strip():
            st.write("No sufficient context retrieved to answer the query.")
            return "I cannot determine this from the provided information."

        # Generate the answer
        answer = model(context)
        st.write(f"### Generated Answer: {answer}")
        return answer
    else:
        st.write("No relevant documents found for the query.")
        return "I cannot determine this from the provided information."

# Main Streamlit Interface
def main():
    st.title("🦙 Document-based Question Answering System")

    # Upload Documents
    uploaded_files = st.file_uploader(
        "Upload your documents (.pdf or .txt):",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    # FAISS index and metadata paths (hidden)
    faiss_index_path = "faiss_index.index"
    metadata_path = "metadata.pkl"

    # Initialize Llama 3.1 8B model
    model = HuggingFaceHub(repo_id="meta-llama/Llama-3.1-8B", token=os.getenv('HF_ACCESS_TOKEN'))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Check if FAISS index exists and load or create
    if os.path.exists(faiss_index_path) and os.path.exists(metadata_path):
        st.write("FAISS index exists. Loading FAISS index...")
        metadata = load_metadata(metadata_path)
        index = load_faiss_index(faiss_index_path)
    else:
        st.write("No FAISS index found. Please upload documents to create the index.")
        metadata, index = None, None

    # Update FAISS index and metadata if new files are uploaded
    new_index, new_metadata = update_index_with_new_files(
        uploaded_files, faiss_index_path, metadata_path, embeddings
    )
    if new_index and new_metadata:
        index = new_index
        metadata = new_metadata

    # Display chat interface for user interaction
    st.set_page_config(
        page_title="Chat with Doc",
        page_icon="📄",
        layout="centered"
    )

    st.title("🦙 Chat with Doc - LLAMA 3.1")

    # Initialize chat history in Streamlit session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input for asking questions
    user_input = st.chat_input("Ask Llama...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            if query:
                search_similar_docs_with_faiss_and_generate_answer(user_input, index, metadata, embeddings, model)
                assistant_response = "Answer based on your query"
            st.markdown(assistant_response)
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

if __name__ == "__main__":
    main()
