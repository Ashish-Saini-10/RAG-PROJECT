"""
ingest.py - Document Ingestion Pipeline
========================================
Phase 2: PDF Loading & Metadata Extraction
Phase 3: Recursive Character Text Chunking
Phase 4: Embedding + ChromaDB Vector Storage (via FastEmbed / ONNX)
"""

import os
# Force protobuf to use pure-Python implementation to avoid descriptor conflicts
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
from pathlib import Path
from typing import List

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding

load_dotenv()

# Configuration
DATA_DIR      = "./data"
CHROMA_DIR    = "./chroma_db"
COLLECTION    = "research_docs"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50


# Phase 2 - PDF Loading & Metadata Extraction
def load_pdfs(data_dir: str = DATA_DIR) -> List:
    documents = []
    pdf_files = list(Path(data_dir).glob("**/*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{data_dir}'. "
            "Add at least one PDF before ingesting."
        )

    print(f"\n[+] Found {len(pdf_files)} PDF file(s) in '{data_dir}'")

    for pdf_path in pdf_files:
        print(f"    Loading: {pdf_path.name} ...", end=" ", flush=True)
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()

            for doc in docs:
                doc.metadata["filename"]  = pdf_path.name
                doc.metadata["file_path"] = str(pdf_path.resolve())
                # PyPDFLoader is 0-indexed; convert to 1-indexed
                doc.metadata["page"]      = doc.metadata.get("page", 0) + 1

            documents.extend(docs)
            print(f"OK ({len(docs)} pages)")
        except Exception as exc:
            print(f"ERROR: {exc}")

    print(f"\n    -> Total pages loaded: {len(documents)}")
    return documents


# Phase 3 - Deterministic Text Chunking
def chunk_documents(documents: List) -> List:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    print(f"\n[+] Chunking complete:")
    print(f"    -> {len(documents)} pages -> {len(chunks)} chunks")
    print(f"    -> Chunk size: {CHUNK_SIZE} chars | Overlap: {CHUNK_OVERLAP} chars")

    return chunks


# Phase 4 - Embedding + ChromaDB Storage
def build_vector_store(chunks: List, client=None) -> None:
    print(f"\n[+] Loading FastEmbed ONNX model: '{EMBED_MODEL}' ...")
    print("    (First run downloads ~25MB model — subsequent runs are instant)")

    # FastEmbed: pure ONNX, no PyTorch, works on Windows
    embed_model = TextEmbedding(model_name=EMBED_MODEL)
    print("    -> Embedding model ready.")

    if client is not None:
        print("    -> Using provided Chroma client.")
    else:
        # Wipe and recreate ChromaDB collection
        print(f"\n[+] Building ChromaDB index (Ephemeral) ...")
        client = chromadb.EphemeralClient()

    existing = [c.name for c in client.list_collections()]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
        print("    -> Cleared previous collection.")

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare batch data
    texts     = [c.page_content for c in chunks]
    ids       = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "filename": c.metadata.get("filename", "unknown"),
            "file_path": c.metadata.get("file_path", ""),
            "page": int(c.metadata.get("page", 1)),
        }
        for c in chunks
    ]

    # Embed all chunks using FastEmbed (batched internally)
    print(f"    -> Embedding {len(texts)} chunks ...")
    embeddings = list(embed_model.embed(texts))
    embeddings_list = [e.tolist() for e in embeddings]
    print(f"    -> Embedding complete. Storing in ChromaDB ...")

    # Batch-insert into ChromaDB
    batch_size = 100
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings_list[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"    -> {len(chunks)} vectors written to disk.")
    print(f"    -> Persistent store: {Path(CHROMA_DIR).resolve()}")


# Entry Point
def main(data_dir: str = DATA_DIR, client=None) -> dict:
    print("=" * 60)
    print("  CiteMind - Document Ingestion Pipeline")
    print("=" * 60)

    os.makedirs(data_dir, exist_ok=True)

    documents = load_pdfs(data_dir)
    chunks    = chunk_documents(documents)
    build_vector_store(chunks, client=client)

    num_docs = len(set(d.metadata["filename"] for d in documents))

    print("\n" + "=" * 60)
    print("  [DONE] Ingestion complete!")
    print(f"     Documents : {num_docs}")
    print(f"     Pages     : {len(documents)}")
    print(f"     Chunks    : {len(chunks)}")
    print(f"     DB Path   : {Path(CHROMA_DIR).resolve()}")
    print("=" * 60 + "\n")

    return {
        "status"    : "success",
        "num_docs"  : num_docs,
        "num_pages" : len(documents),
        "num_chunks": len(chunks),
    }


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result["status"] == "success" else 1)
