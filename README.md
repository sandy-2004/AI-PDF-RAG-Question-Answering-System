# 📚 AI PDF RAG Question Answering System

An AI-powered web application that allows users to upload PDF documents and ask natural language questions about their content. This project leverages **Retrieval-Augmented Generation (RAG)** to provide highly accurate answers backed by source citations from the uploaded documents, preventing AI hallucinations.

## ✨ Features
- **PDF Text Extraction:** Seamlessly read and extract text from single or multiple PDF files.
- **Smart Chunking & Embeddings:** Processes documents into manageable chunks and converts them into searchable vectors.
- **Semantic Search:** Uses FAISS vector database to find the most relevant document sections based on the user's question.
- **Powered by Groq:** Utilizes the blazing-fast Llama-3.3-70b-versatile model via Groq API to generate human-like answers.
- **Source Citations:** Transparently displays exactly which file and page the AI used to generate its answer.

## 🛠️ Tech Stack
- **Frontend / UI:** [Streamlit](https://streamlit.io/)
- **LLM:** [Groq](https://groq.com/) (Llama-3.3-70b-versatile)
- **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
- **Embeddings:** [Sentence-Transformers](https://huggingface.co/sentence-transformers) (`all-MiniLM-L6-v2`)
- **PDF Parsing:** `pypdf`

## 🚀 Local Setup & Installation

### 1. Get the Code
Clone the repository or download the project files:
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
```
Activate it:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up Environment Variables
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_actual_api_key_here
```
*(You can get a free API key from the [Groq Console](https://console.groq.com/))*

### 5. Run the Application
```bash
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.

## ☁️ Deployment (Streamlit Community Cloud)

**⚠️ SECURITY WARNING: NEVER upload your `.env` file to GitHub.**

1. Push your code to a GitHub repository (Ensure `.env` is listed in your `.gitignore`).
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and click **New App**.
3. Select your repository, branch, and set `app.py` as the main file path.
4. Click **Advanced Settings** before deploying.
5. Add your API key to the **Secrets** section:
   ```toml
   GROQ_API_KEY="gsk_your_actual_api_key_here"
   ```
6. Click **Deploy!**

## 🧠 How it Works (The RAG Pipeline)
1. **Ingest:** PDFs are uploaded and parsed into raw text.
2. **Chunk:** Text is split into overlapping 700-character chunks to retain context.
3. **Embed:** Chunks are converted into numerical vectors using Hugging Face embeddings.
4. **Store:** Vectors are stored in a local FAISS index for rapid searching.
5. **Retrieve:** When a user asks a question, it is embedded, and FAISS finds the top 5 most similar document chunks.
6. **Generate:** The retrieved context + the question are sent to the Groq LLM to generate a factual, source-backed answer.
