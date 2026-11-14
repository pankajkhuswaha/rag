import os
from typing import Iterator, List, Optional
from collections import Counter
from pypdf import PdfReader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from agent.config import PDF_PATH, DB_DIR, DOCUMENTS_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


class LocalPDFLoader(BaseLoader):
    """Lightweight PDF loader using pypdf without langchain-community."""

    def __init__(self, file_path: str, display_name: Optional[str] = None):
        self.file_path = file_path
        self.display_name = display_name or os.path.basename(file_path)

    def lazy_load(self) -> Iterator[Document]:
        reader = PdfReader(self.file_path)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                yield Document(
                    page_content=text,
                    metadata={
                        "source": self.file_path,
                        "filename": self.display_name,
                        "page": page_num + 1,
                    },
                )


class LocalTextLoader(BaseLoader):
    """Lightweight text/markdown document loader."""

    def __init__(self, file_path: str, display_name: Optional[str] = None):
        self.file_path = file_path
        self.display_name = display_name or os.path.basename(file_path)

    def lazy_load(self) -> Iterator[Document]:
        with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        if text.strip():
            yield Document(
                page_content=text,
                metadata={
                    "source": self.file_path,
                    "filename": self.display_name,
                    "page": 1,
                },
            )


def load_document(file_path: str, display_name: Optional[str] = None) -> List[Document]:
    """Loads a document based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = LocalPDFLoader(file_path, display_name=display_name)
    elif ext in {".txt", ".md", ".markdown", ".json", ".csv", ".py"}:
        loader = LocalTextLoader(file_path, display_name=display_name)
    else:
        loader = LocalTextLoader(file_path, display_name=display_name)
    return loader.load()


def get_vectorstore() -> Chroma:
    """Loads existing ChromaDB from disk or indexes the initial documents on first run."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )

    try:
        existing_data = vectorstore.get(limit=1)
        has_existing_data = bool(existing_data.get("ids"))
    except Exception:
        has_existing_data = False

    if not has_existing_data:
        if os.path.exists(PDF_PATH):
            print(f"Indexing initial document {PDF_PATH}...")
            ingest_document(PDF_PATH, vectorstore=vectorstore)
        else:
            print("Vectorstore initialized (empty). Upload documents via the UI to populate.")
    else:
        print("Loaded existing vector database.")

    return vectorstore


def ingest_document(
    file_path: str,
    vectorstore: Chroma,
    original_filename: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    replace_existing: bool = True,
) -> int:
    """Chunks and adds a document to the Chroma vectorstore. Returns number of chunks indexed."""
    display_name = original_filename or os.path.basename(file_path)
    if replace_existing:
        delete_document_from_vectorstore(vectorstore, display_name)

    docs = load_document(file_path, display_name=display_name)
    if not docs:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = splitter.split_documents(docs)
    if not splits:
        return 0

    for i, split in enumerate(splits):
        split.metadata["filename"] = display_name
        split.metadata["chunk_index"] = i

    vectorstore.add_documents(documents=splits)
    print(f"Indexed {len(splits)} chunks from {display_name} into ChromaDB.")
    return len(splits)


def get_indexed_documents_summary(vectorstore: Chroma) -> List[dict]:
    """Returns a list of all indexed documents with their chunk counts."""
    try:
        data = vectorstore.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        if not metadatas:
            return []

        counter = Counter()
        for meta in metadatas:
            if meta:
                name = meta.get("filename") or os.path.basename(meta.get("source", "Unknown"))
                counter[name] += 1

        return [
            {"filename": name, "chunks": count}
            for name, count in sorted(counter.items(), key=lambda x: x[0].lower())
        ]
    except Exception as e:
        print(f"Error fetching document summary: {e}")
        return []


def delete_document_from_vectorstore(vectorstore: Chroma, filename: str) -> int:
    """Deletes all chunks associated with a specific filename/source from the vectorstore."""
    try:
        data = vectorstore.get(include=["metadatas"])
        ids_to_delete = []
        for doc_id, meta in zip(data.get("ids", []), data.get("metadatas", [])):
            if meta:
                name = meta.get("filename") or os.path.basename(meta.get("source", ""))
                src = meta.get("source", "")
                if name == filename or src == filename or os.path.basename(src) == filename:
                    ids_to_delete.append(doc_id)

        if ids_to_delete:
            vectorstore.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} chunks for {filename}.")
        return len(ids_to_delete)
    except Exception as e:
        print(f"Error deleting document from vectorstore: {e}")
        return 0
