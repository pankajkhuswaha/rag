# RAG Knowledge Base & AI Chat Assistant

An advanced Retrieval-Augmented Generation (RAG) assistant built with **LangGraph**, **ChromaDB**, **Ollama**, and **Sentence-Transformers (Cross-Encoder)**, wrapped in a modern **Streamlit** user interface.

---

## 🌟 Key Features

1. **Dynamic Knowledge Base Ingestion**:
   - Upload new documents (**PDF**, **TXT**, **Markdown**) through the UI at any time to instantly expand the AI's knowledge base.
   - Text is parsed, chunked with `RecursiveCharacterTextSplitter`, embedded using `nomic-embed-text`, and indexed into ChromaDB.
   - Replace or update documents without having to re-index the entire database.

2. **Knowledge Base Management**:
   - Visual dashboard showing the list of currently indexed documents and chunk counts.
   - One-click deletion of specific documents from ChromaDB.

3. **Two-Stage Reranked Retrieval with Hallucination Defense**:
   - **Stage 1**: Broad candidate retrieval via ChromaDB similarity search.
   - **Stage 2**: High-precision semantic reranking using `BAAI/bge-reranker-base` cross-encoder.
   - **Relevance Cutoff Guardrail**: Chunks below the score cutoff (e.g. `< 0.25`) are discarded. If no relevant chunks exist in the knowledge base, the system deterministically refuses to answer rather than guessing.

4. **Conversational Memory**:
   - Tracks chat history per session thread.
   - Standalone query rewriter resolves follow-up questions and pronouns (e.g., "What about its eligibility?" -> "What is the eligibility for B.Tech CSE?").

5. **Transparency & Source Citations**:
   - Every answer includes an expandable **"📚 Retrieved Sources & Excerpts"** accordion showing the source file name, page number, relevance score, and the exact text passage used.

6. **Anti-Hallucination Controls**:
   - **Strict Grounding Mode**: Enforces that answers must be derived 100% from `<reference_facts>`.
   - **Cutoff Slider**: Adjust the Cross-Encoder score cutoff directly from the sidebar.
   - **Low Temperature**: Defaulted to `0.1` for factual precision.

7. **Dual Inference Engine (Local Ollama vs Cloud Groq)**:
   - **Local Mode (Ollama)**: 100% offline, private inference with `llama3.1:8b`.
   - **Cloud Mode (Groq)**: Lightning-fast production inference using Groq LPU with `llama-3.3-70b-versatile` or `llama-3.1-8b-instant`.
   - Seamlessly toggle between Local and Groq directly from the UI sidebar or via `.env`.

8. **Customizable Personas & Settings**:
   - Features **Ami**, Admissions Counselor at **Amity University**, with natural conversational tone and intelligent off-topic handling.
   - Sliders for LLM temperature, initial candidates fetched ($k$), and final reranked chunks ($top\_k$).

---

## 🚀 Quick Start

### 1. Prerequisites
- **For Local Inference (Ollama)**:
  Ensure [Ollama](https://ollama.com/) is running:
  ```bash
  ollama serve
  ollama pull llama3.1:8b
  ollama pull nomic-embed-text
  ```
- **For Cloud Inference (Groq)**:
  Get a free API key at [console.groq.com/keys](https://console.groq.com/keys). You can enter it directly in the UI sidebar or set it in a `.env` file:
  ```bash
  cp .env.example .env
  # Add your GROQ_API_KEY inside .env
  ```

### 2. Launch the Streamlit UI Wrapper
You can launch the web interface in one of two ways:

**Option A (Windows Shortcut):**
Double-click `run_ui.bat` in the project root.

**Option B (Command Line):**
```powershell
.\.venv\Scripts\activate
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

### 3. Alternative: Run the CLI
If you prefer the terminal interface:
```powershell
.\.venv\Scripts\python.exe main.py
```

---

## 📂 Project Structure

```
rag/
├── agent/
│   ├── __init__.py         # Package exports
│   ├── chain.py            # LangGraph workflow (query rewriter + answer generator)
│   ├── config.py           # Model names, storage paths, chunk sizes
│   ├── ingestion.py        # Loaders (PDF, TXT, MD), chunking & ChromaDB operations
│   ├── memory.py           # InMemorySaver thread checkpointer
│   └── retriever.py        # Chroma retriever + CrossEncoder reranker
├── chroma_db/              # Persistent Chroma vector store
├── documents/              # Local storage for indexed documents
├── app.py                  # Streamlit Web UI wrapper
├── main.py                 # CLI interface
├── requirements.txt        # Key dependencies
├── run_ui.bat              # 1-click Windows launcher
└── run_ui.ps1              # PowerShell launcher
```
