import os
import re
import hashlib
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss

# ============================================================
# CONFIGURATION
# ============================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is missing. "
        "Please add it to your .env file."
    )
    st.stop()

client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.3-70b-versatile"

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI PDF RAG Assistant",
    page_icon="",
    layout="wide"
)

# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

# ============================================================
# SESSION STATE
# ============================================================
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "metadata" not in st.session_state:
    st.session_state.metadata = []

if "index" not in st.session_state:
    st.session_state.index = None

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============================================================
# CSS
# ============================================================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 700;
    }
    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }
    .source-box {
        padding: 12px;
        border: 1px solid #ddd;
        border-radius: 8px;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# TITLE
# ============================================================
st.markdown(
    '<div class="main-title">'
    'AI PDF RAG Question Answering System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload documents and ask questions using Retrieval-Augmented Generation.'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# PDF EXTRACTION
# ============================================================
def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    documents = []
    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):
        text = page.extract_text()
        if text and text.strip():
            text = re.sub(
                r"\s+",
                " ",
                text
            ).strip()
            documents.append(
                {
                    "text": text,
                    "page": page_number,
                    "source": uploaded_file.name
                }
            )
    return documents

# ============================================================
# TEXT CHUNKING
# ============================================================
def create_chunks(
    documents,
    chunk_size=700,
    overlap=100
):
    chunks = []
    metadata = []
    for document in documents:
        text = document["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(
                    chunk.strip()
                )
                metadata.append(
                    {
                        "source": document["source"],
                        "page": document["page"]
                    }
                )
            start += chunk_size - overlap
    return chunks, metadata

# ============================================================
# EMBEDDINGS
# ============================================================
def create_embeddings(chunks):
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.astype(
        "float32"
    )

# ============================================================
# FAISS INDEX
# ============================================================
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(
        dimension
    )
    index.add(
        embeddings
    )
    return index

# ============================================================
# PROCESS DOCUMENTS
# ============================================================
def process_documents(uploaded_files):
    all_documents = []
    for uploaded_file in uploaded_files:
        documents = extract_pdf_text(
            uploaded_file
        )
        all_documents.extend(
            documents
        )
    if not all_documents:
        raise ValueError(
            "No readable text was found in the uploaded PDFs."
        )
    chunks, metadata = create_chunks(
        all_documents
    )
    if not chunks:
        raise ValueError(
            "Unable to create text chunks."
        )
    embeddings = create_embeddings(
        chunks
    )
    index = create_faiss_index(
        embeddings
    )
    return (
        chunks,
        metadata,
        index
    )

# ============================================================
# SEARCH DOCUMENTS
# ============================================================
def search_documents(
    question,
    top_k=5
):
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    question_embedding = question_embedding.astype(
        "float32"
    )
    scores, indices = st.session_state.index.search(
        question_embedding,
        top_k
    )
    results = []
    for score, index in zip(
        scores[0],
        indices[0]
    ):
        if index == -1:
            continue
        results.append(
            {
                "text": st.session_state.chunks[index],
                "metadata": st.session_state.metadata[index],
                "score": float(score)
            }
        )
    return results

# ============================================================
# GENERATE ANSWER
# ============================================================
def generate_answer(
    question,
    search_results
):
    context_parts = []
    for i, result in enumerate(
        search_results,
        start=1
    ):
        metadata = result["metadata"]
        context_parts.append(
            f"SOURCE {i}\n"
            f"File: {metadata['source']}\n"
            f"Page: {metadata['page']}\n"
            f"Content:\n"
            f"{result['text']}\n"
        )
    context = "\n".join(
        context_parts
    )
    prompt = f"""
    You are a document question-answering assistant.
    Answer the user's question using ONLY the
    provided document context.

    Important rules:
    1. Do not invent information.
    2. Do not use outside knowledge.
    3. If the answer is not available in the
    provided context, say:
    "I could not find this information in
    the uploaded documents."
    4. Give a clear and concise answer.
    5. Mention relevant source file and page.
    6. If multiple documents contain relevant
    information, combine them carefully.
    7. Do not claim something is present if it
    is not supported by the context.

    DOCUMENT CONTEXT:
    {context}

    USER QUESTION:
    {question}

    Return the answer in this format:
    ANSWER:
    <answer>

    SOURCES:
    - <filename>, Page <number>
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content":
                "You are a reliable RAG assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1
    )
    return response.choices[0].message.content

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )
    process_button = st.button(
        "Process Documents",
        use_container_width=True
    )
    st.divider()
    st.subheader("RAG Pipeline")
    st.write(
        "PDF\n"
        "↓\n"
        "Text Extraction\n"
        "↓\n"
        "Chunking\n"
        "↓\n"
        "Embeddings\n"
        "↓\n"
        "FAISS\n"
        "↓\n"
        "Semantic Search\n"
        "↓\n"
        "Groq LLM"
    )
    st.divider()
    if st.session_state.documents_processed:
        st.success(
            f"{len(st.session_state.chunks)} chunks indexed"
        )

# ============================================================
# PROCESS BUTTON
# ============================================================
if process_button:
    if not uploaded_files:
        st.warning(
            "Please upload at least one PDF."
        )
    else:
        with st.spinner(
            "Processing documents..."
        ):
            try:
                chunks, metadata, index = process_documents(
                    uploaded_files
                )
                st.session_state.chunks = chunks
                st.session_state.metadata = metadata
                st.session_state.index = index
                st.session_state.documents_processed = True
                st.session_state.chat_history = []
                st.success(
                    f"Successfully processed "
                    f"{len(uploaded_files)} PDF(s). "
                    f"Created {len(chunks)} chunks."
                )
            except Exception as e:
                st.error(
                    f"Processing error: {e}"
                )

# ============================================================
# DOCUMENT STATUS
# ============================================================
if st.session_state.documents_processed:
    st.header("Document Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Documents",
            len(uploaded_files)
            if uploaded_files
            else 0
        )
    with col2:
        st.metric(
            "Text Chunks",
            len(st.session_state.chunks)
        )
    with col3:
        st.metric(
            "Vector Dimension",
            embedding_model.get_sentence_embedding_dimension()
        )

# ============================================================
# QUESTION AREA
# ============================================================
st.divider()
st.header("Ask Questions")

if not st.session_state.documents_processed:
    st.info(
        "Upload PDF files from the sidebar and "
        "click 'Process Documents' first."
    )
else:
    question = st.text_input(
        "Enter your question",
        placeholder=
        "Example: What is inheritance in Python?"
    )
    ask_button = st.button(
        "Ask AI",
        type="primary",
        use_container_width=True
    )

    if ask_button:
        if not question.strip():
            st.warning(
                "Please enter a question."
            )
        else:
            with st.spinner(
                "Searching documents..."
            ):
                try:
                    search_results = search_documents(
                        question,
                        top_k=5
                    )
                    answer = generate_answer(
                        question,
                        search_results
                    )
                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "answer": answer,
                            "results": search_results
                        }
                    )
                except Exception as e:
                    st.error(
                        f"Error: {e}"
                    )

# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================
if st.session_state.chat_history:
    st.divider()
    st.header("Conversation")
    for chat in reversed(
        st.session_state.chat_history
    ):
        st.markdown(
            f"**Question:** {chat['question']}"
        )
        st.markdown(
            chat["answer"]
        )
        with st.expander(
            "View Retrieved Sources"
        ):
            for i, result in enumerate(
                chat["results"],
                start=1
            ):
                metadata = result["metadata"]
                st.markdown(
                    f"**Source {i}**\n"
                    f"File: `{metadata['source']}`\n"
                    f"Page: `{metadata['page']}`\n"
                    f"Similarity Score:\n"
                    f"`{result['score']:.4f}`\n"
                )
                st.write(
                    result["text"]
                )
        st.divider()

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "AI PDF RAG Assistant | "
    "Python + Streamlit + FAISS + Groq"
)
