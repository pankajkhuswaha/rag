import os
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from agent.config import (
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_AVAILABLE_MODELS,
    RERANKER_MODEL,
    INITIAL_FETCH_K,
    FINAL_TOP_K,
)
from agent.ingestion import (
    get_vectorstore,
    ingest_document,
    get_indexed_documents_summary,
    delete_document_from_vectorstore,
)
from agent.retriever import RerankedRetriever
from agent.chain import (
    build_rag_chain,
    DEFAULT_COUNSELOR_PROMPT,
    GENERAL_ASSISTANT_PROMPT,
)
from agent.memory import clear_session_history

# Page configuration
st.set_page_config(
    page_title="Amity Admissions Assistant (Ami)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E88E5;
    }
    .sub-title {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
        margin-right: 0.4rem;
        background-color: #E3F2FD;
        color: #0D47A1;
    }
    .source-box {
        border-left: 3px solid #1E88E5;
        padding-left: 12px;
        margin-top: 8px;
        margin-bottom: 8px;
        background-color: rgba(30, 136, 229, 0.05);
        border-radius: 0 4px 4px 0;
    }
    .card-stat {
        background-color: #f8f9fa;
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_base_rag_components():
    """Initializes and caches the persistent Chroma vector store."""
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    vectorstore = get_vectorstore()
    return vectorstore


# Initialize session state variables
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state or "rag_app" not in st.session_state:
    st.session_state.needs_chain_rebuild = True

# Load vectorstore
vectorstore = load_base_rag_components()

# --- SIDEBAR: Knowledge Base & Configuration ---
with st.sidebar:
    st.title("📂 Knowledge Base")

    # Document Upload Section
    st.markdown("### 📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or MD to extend knowledge base",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Files will be parsed, chunked, and embedded into ChromaDB.",
    )

    replace_existing = st.checkbox(
        "Replace existing version if re-uploaded",
        value=True,
        help="If checked, removes old chunks with the same filename before adding new ones.",
    )

    if st.button("🚀 Ingest Uploaded Documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select at least one document to upload.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_added_chunks = 0

            upload_error = None
            for i, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                saved_path = os.path.join(DOCUMENTS_DIR, filename)

                status_text.text(f"Saving and indexing {filename}...")

                try:
                    # Save file locally
                    with open(saved_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Ingest document
                    chunks_added = ingest_document(
                        file_path=saved_path,
                        vectorstore=vectorstore,
                        original_filename=filename,
                        replace_existing=replace_existing,
                    )
                    total_added_chunks += chunks_added
                except Exception as exc:
                    upload_error = str(exc)
                    st.error(f"❌ Failed to index '{filename}': {exc}")
                    if "Ollama" in upload_error or "connect" in upload_error.lower():
                        st.info("💡 Ensure Ollama is running (`ollama serve`) and model `nomic-embed-text` is pulled.")
                    break
                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.empty()
            progress_bar.empty()
            if not upload_error:
                st.success(f"Indexed {len(uploaded_files)} file(s) into {total_added_chunks} chunks!")
                st.session_state.needs_chain_rebuild = True
                st.rerun()

    st.markdown("---")

    # Knowledge Base Inventory
    st.markdown("### 📚 Indexed Documents")
    doc_summary = get_indexed_documents_summary(vectorstore)
    total_chunks = sum(item["chunks"] for item in doc_summary)

    col1, col2 = st.columns(2)
    col1.metric("Documents", len(doc_summary))
    col2.metric("Total Chunks", total_chunks)

    if doc_summary:
        with st.expander("View Indexed Files", expanded=False):
            for doc in doc_summary:
                st.write(f"📄 **{doc['filename']}** ({doc['chunks']} chunks)")

        # Document Removal Option
        st.markdown("#### Manage Files")
        selected_file_to_delete = st.selectbox(
            "Select document to remove",
            options=["None"] + [doc["filename"] for doc in doc_summary],
        )
        if selected_file_to_delete != "None":
            if st.button(f"🗑️ Delete '{selected_file_to_delete}'", use_container_width=True):
                deleted_chunks = delete_document_from_vectorstore(vectorstore, selected_file_to_delete)
                st.success(f"Removed {deleted_chunks} chunks for {selected_file_to_delete}.")
                st.session_state.needs_chain_rebuild = True
                st.rerun()
    else:
        st.info("Knowledge base is currently empty. Upload documents above to begin!")

    st.markdown("---")

    # LLM Backend Provider Switch
    st.markdown("### 🤖 LLM Engine")
    provider_option = st.radio(
        "Inference Backend",
        options=["🏠 Local (Ollama)", "⚡ Cloud (Groq)"],
        index=1 if LLM_PROVIDER == "groq" else 0,
        horizontal=True,
        help="Switch seamlessly between local offline Ollama inference and lightning-fast Groq Cloud inference.",
    )

    is_groq = "Groq" in provider_option
    current_provider = "groq" if is_groq else "ollama"

    if is_groq:
        groq_api_key_input = st.text_input(
            "Groq API Key",
            value=st.session_state.get("groq_api_key", GROQ_API_KEY),
            type="password",
            placeholder="gsk_...",
            help="Free key available at https://console.groq.com/keys",
        )
        st.session_state.groq_api_key = groq_api_key_input

        selected_model = st.selectbox(
            "Groq Model",
            options=GROQ_AVAILABLE_MODELS,
            index=0,
            help="llama-3.3-70b-versatile delivers exceptional reasoning speed and accuracy.",
        )
        if not groq_api_key_input:
            st.warning("⚠️ Enter a Groq API Key to enable Groq cloud inference.")
    else:
        groq_api_key_input = None
        selected_model = st.text_input(
            "Ollama Model",
            value=LLM_MODEL,
            help="Local model name installed in Ollama (e.g. llama3.1:8b).",
        )

    st.markdown("---")

    # Model & Retrieval Settings
    st.markdown("### ⚙️ RAG Settings")

    persona_choice = st.selectbox(
        "Assistant Persona",
        options=[
            "Amity Admissions Counselor (Ami)",
            "General Knowledge Assistant",
            "Custom Persona",
        ],
        index=0,
    )

    if persona_choice == "Amity Admissions Counselor (Ami)":
        system_prompt = DEFAULT_COUNSELOR_PROMPT
    elif persona_choice == "General Knowledge Assistant":
        system_prompt = GENERAL_ASSISTANT_PROMPT
    else:
        system_prompt = st.text_area(
            "Custom System Prompt (include {context})",
            value=GENERAL_ASSISTANT_PROMPT,
            height=150,
        )

    # Anti-Hallucination Guardrail Controls
    st.markdown("#### 🛡️ Factual Grounding")
    strict_mode = st.toggle(
        "Stop Hallucinations (Strict Guardrail)",
        value=True,
        help="If enabled, the AI deterministically refuses to answer when information is not present in the uploaded documents.",
    )

    score_threshold = st.slider(
        "Relevance Score Cutoff",
        min_value=0.05,
        max_value=0.80,
        value=0.25,
        step=0.05,
        help="Chunks with CrossEncoder relevance scores below this cutoff are considered irrelevant and ignored.",
    )

    top_k = st.slider(
        "Final Top K Chunks (Reranked)",
        min_value=1,
        max_value=10,
        value=FINAL_TOP_K,
        help="Number of most relevant chunks passed to the LLM.",
    )

    initial_k = st.slider(
        "Initial Candidate Chunks (ChromaDB)",
        min_value=3,
        max_value=25,
        value=INITIAL_FETCH_K,
        help="Initial pool fetched from Chroma before cross-encoder reranking.",
    )

    temperature = st.slider(
        "LLM Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(LLM_TEMPERATURE),
        step=0.05,
        help="Keep at 0.0 - 0.1 for strict factual precision without creative hallucination.",
    )

    if st.button("🔄 Reset Conversation History", use_container_width=True):
        clear_session_history(st.session_state.thread_id)
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.toast("Conversation memory cleared!", icon="🧹")
        st.rerun()

    # System Status info
    st.markdown("---")
    backend_display = f"⚡ Groq (`{selected_model}`)" if is_groq else f"🏠 Ollama (`{selected_model}`)"
    embed_name = getattr(vectorstore._embedding_function, "model_name", EMBEDDING_MODEL)
    st.caption(
        f"**Engine:** {backend_display}  \n"
        f"**Embeddings:** `{embed_name}`  \n"
        f"**Reranker:** `{RERANKER_MODEL.split('/')[-1]}`  \n"
        f"**Cutoff:** `>= {score_threshold}`  \n"
        f"**Thread ID:** `{st.session_state.thread_id[:8]}...`"
    )

# --- REBUILD RAG CHAIN IF PARAMETERS OR DATA CHANGED ---
current_config_key = (
    f"{current_provider}_{selected_model}_{groq_api_key_input}_{persona_choice}_{top_k}_"
    f"{initial_k}_{temperature}_{total_chunks}_{strict_mode}_{score_threshold}"
)
if st.session_state.get("last_config_key") != current_config_key or st.session_state.needs_chain_rebuild:
    retriever = RerankedRetriever(
        vectorstore=vectorstore,
        initial_k=initial_k,
        top_k=top_k,
        score_threshold=score_threshold,
        reranker_model=RERANKER_MODEL,
    )
    try:
        rag_app = build_rag_chain(
            retriever=retriever,
            system_prompt=system_prompt,
            provider=current_provider,
            model_name=selected_model,
            temperature=temperature,
            groq_api_key=groq_api_key_input,
            strict_mode=strict_mode,
        )
        st.session_state.retriever = retriever
        st.session_state.rag_app = rag_app
        st.session_state.last_config_key = current_config_key
        st.session_state.needs_chain_rebuild = False
    except Exception as build_err:
        st.sidebar.error(f"Failed to initialize {current_provider.upper()}: {build_err}")

# --- MAIN AREA: Chat Interface ---
badge_provider = f"⚡ Groq Cloud ({selected_model})" if is_groq else f"🏠 Local Ollama ({selected_model})"
st.markdown('<div class="main-title">🎓 Amity Admissions Assistant (Ami)</div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="sub-title">
        <span class="badge">{badge_provider}</span>
        <span class="badge">Amity Admissions</span>
        <span class="badge">CrossEncoder Reranker</span>
        <span class="badge">Anti-Hallucination Guardrail</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Display empty chat onboarding
if not st.session_state.messages:
    if persona_choice == "Amity Admissions Counselor (Ami)":
        st.info(
            "👋 **Hi, I'm Ami!** I'm your Admissions Counselor at Amity University. Ask me about courses, "
            "eligibility criteria, admission procedures, campus facilities, or scholarships. You can also upload new documents in the sidebar to extend my knowledge!"
        )
    else:
        st.info(
            "👋 **Welcome!** Ask any question based on your uploaded documents. "
            "Upload more PDFs, TXT, or Markdown files from the sidebar at any time."
        )

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display cited sources if available
        if msg.get("sources"):
            with st.expander(f"📚 Retrieved Sources & Excerpts ({len(msg['sources'])} chunks)", expanded=False):
                for i, doc in enumerate(msg["sources"], 1):
                    src_file = doc.get("filename") or doc.get("source", "Document")
                    page_info = f" • Page {doc.get('page')}" if doc.get("page") else ""
                    score_info = f" • Relevance: {doc.get('rerank_score')}" if doc.get("rerank_score") is not None else ""
                    st.markdown(f"**Reference {i}:** `{src_file}`{page_info}{score_info}")
                    st.markdown(f"> {doc.get('content', '').strip()}")
                    st.divider()
        elif msg.get("no_sources_flag"):
            st.caption("ℹ️ *General guidance response (no matching prospectus chunks).*")

# Chat Input Box
user_query = st.chat_input("Ask a question about your knowledge base...")

if user_query:
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Invoke RAG App
    with st.chat_message("assistant"):
        with st.spinner("Consulting knowledge base and thinking..."):
            try:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = st.session_state.rag_app.invoke(
                    {"messages": [HumanMessage(content=user_query)]},
                    config=config,
                )

                response_text = result["messages"][-1].content
                retrieved_docs = result.get("retrieved_docs", [])

                # Format serializable sources for session state
                sources_data = [
                    {
                        "content": doc.page_content,
                        "filename": doc.metadata.get("filename") or os.path.basename(doc.metadata.get("source", "")),
                        "source": doc.metadata.get("source", ""),
                        "page": doc.metadata.get("page", 1),
                        "rerank_score": doc.metadata.get("rerank_score", None),
                    }
                    for doc in retrieved_docs
                ]

                # Render response
                st.markdown(response_text)

                # Render sources
                if sources_data:
                    with st.expander(f"📚 Retrieved Sources & Excerpts ({len(sources_data)} chunks)", expanded=False):
                        for i, doc in enumerate(sources_data, 1):
                            src_file = doc.get("filename") or doc.get("source", "Document")
                            page_info = f" • Page {doc.get('page')}" if doc.get("page") else ""
                            score_info = f" • Relevance: {doc.get('rerank_score')}" if doc.get("rerank_score") is not None else ""
                            st.markdown(f"**Reference {i}:** `{src_file}`{page_info}{score_info}")
                            st.markdown(f"> {doc.get('content', '').strip()}")
                            st.divider()
                elif strict_mode:
                    st.caption("ℹ️ *General guidance response (no matching prospectus chunks).*")

                # Save assistant response to session state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response_text,
                        "sources": sources_data,
                        "no_sources_flag": (not sources_data and strict_mode),
                    }
                )

            except Exception as e:
                error_msg = f"⚠️ An error occurred while generating response: {str(e)}"
                st.error(error_msg)
                st.info("Tip: Ensure Ollama is running and model `llama3.1:8b` is pulled (`ollama run llama3.1:8b`).")
