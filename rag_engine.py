"""
rag_engine.py - Query-Similarity Retrieval Engine
===================================================
Phase 5: Cosine-distance vector search, returning Top-K chunks + scores
Phase 6: Zero-knowledge system prompt with mandatory inline citations
"""

import os
from typing import Generator, List, Tuple

import chromadb
from dotenv import load_dotenv
from fastembed import TextEmbedding
from groq import Groq

load_dotenv()

# Configuration
CHROMA_DIR    = "./chroma_db"
COLLECTION    = "research_docs"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL    = "llama-3.1-8b-instant"
DEFAULT_TOP_K = 3

# Phase 6 - Zero-Knowledge System Prompt / Guardrails
SYSTEM_PROMPT_TEMPLATE = """You are an expert academic research assistant operating under STRICT VERIFICATION MODE.

=======================================================
 ABSOLUTE RULES - DO NOT DEVIATE UNDER ANY CIRCUMSTANCE
=======================================================

1. ZERO EXTERNAL KNOWLEDGE - You MUST answer exclusively using the document
   excerpts provided in the EVIDENCE BLOCK below. Do NOT draw on any knowledge
   from your pre-training or make any assumptions beyond the provided text.

2. MANDATORY INLINE CITATIONS - Every factual sentence you write MUST end with
   an inline citation formatted exactly as: [Filename, Page X]
   Example: "The study found a 32% improvement in accuracy. [paper.pdf, Page 4]"

3. ZERO-KNOWLEDGE DEFAULT - If the provided excerpts do NOT contain enough
   information to answer the question, you MUST respond with ONLY this statement:
   "I cannot find sufficient information in the provided documents to answer
   this question. Please upload relevant documents or rephrase your query."

4. NO FABRICATION - Never invent statistics, citations, authors, dates, or
   conclusions not explicitly stated in the evidence block.

5. RESPONSE FORMAT - Structure your answer with:
   - A concise opening statement (1-2 sentences, cited)
   - Supporting paragraphs with inline citations on every factual claim
   - A brief concluding summary if warranted

=======================================================
 EVIDENCE BLOCK (your ONLY permitted source of truth)
=======================================================

{context}

=======================================================
 END OF EVIDENCE BLOCK
=======================================================

Now answer the user's question using ONLY the evidence above, with full inline citations.
"""

# Lazy-loaded singletons
_embed_model   = None
_chroma_client = None
_collection    = None


def _get_embed_model() -> TextEmbedding:
    global _embed_model
    if _embed_model is None:
        _embed_model = TextEmbedding(model_name=EMBED_MODEL)
    return _embed_model


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def reset_singletons():
    """Call after re-ingestion to force fresh reload."""
    global _embed_model, _chroma_client, _collection
    _embed_model   = None
    _chroma_client = None
    _collection    = None


# Phase 5 - Similarity Retrieval Engine
def retrieve_chunks(query: str, k: int = DEFAULT_TOP_K) -> List[dict]:
    """
    Embeds the user query via FastEmbed (ONNX) and performs cosine
    similarity search against the ChromaDB index.
    Returns list of dicts: text, filename, page, score_pct, raw_score
    """
    embed_model = _get_embed_model()
    collection  = _get_collection()

    # Embed query — FastEmbed returns a generator, take first result
    query_vector = list(embed_model.embed([query]))[0].tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results and results["documents"] and results["documents"][0]:
        for i, doc_text in enumerate(results["documents"][0]):
            dist = results["distances"][0][i] if results["distances"] else 0
            meta = results["metadatas"][0][i] if results["metadatas"] else {}

            # cosine distance in [0,2]; with normalized vecs approx [0,1]
            similarity_pct = max(0.0, (1.0 - float(dist))) * 100.0
            chunks.append({
                "text"     : doc_text,
                "filename" : meta.get("filename", "Unknown Document"),
                "page"     : meta.get("page", 1),
                "score_pct": round(similarity_pct, 1),
                "raw_score": round(float(dist), 4),
            })

    chunks.sort(key=lambda c: c["score_pct"], reverse=True)
    return chunks


# Context Builder
def build_context(chunks: List[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = (
            f"[EXCERPT {i}]  Source: {chunk['filename']}  |  "
            f"Page: {chunk['page']}  |  "
            f"Similarity: {chunk['score_pct']}%"
        )
        parts.append(f"{header}\n{chunk['text'].strip()}")
    return "\n\n" + "\n\n---\n\n".join(parts) + "\n"


# Phase 6 + Groq - Streaming Query
def query_rag_stream(
    query: str,
    k: int = DEFAULT_TOP_K,
    api_key: str | None = None,
) -> Tuple[List[dict], Generator]:

    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            "Vector database not found. Please ingest documents first."
        )

    resolved_key = api_key or os.getenv("GROQ_API_KEY", "")
    if not resolved_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to .env or the sidebar."
        )

    chunks = retrieve_chunks(query, k=k)
    if not chunks:
        raise RuntimeError(
            "No matching chunks found. Try rephrasing your question."
        )

    context    = build_context(chunks)
    sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    client = Groq(api_key=resolved_key)
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": query},
        ],
        temperature=0.05,
        max_tokens=1500,
        stream=True,
    )

    def _token_generator() -> Generator:
        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta

    return chunks, _token_generator()


# Sync Wrapper (CLI / testing)
def query_rag(
    query: str,
    k: int = DEFAULT_TOP_K,
    api_key: str | None = None,
) -> Tuple[str, List[dict]]:
    chunks, stream = query_rag_stream(query, k=k, api_key=api_key)
    answer = "".join(stream)
    return answer, chunks


# CLI Entry Point
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Usage: python rag_engine.py "your question here"')
        sys.exit(1)

    user_query = " ".join(sys.argv[1:])
    print(f"\nQuery: {user_query}\n")
    answer, sources = query_rag(user_query)

    print("-" * 60)
    print("ANSWER:\n")
    print(answer)
    print("\n" + "-" * 60)
    print(f"\nSOURCE CHUNKS ({len(sources)} retrieved):\n")
    for i, s in enumerate(sources, 1):
        print(f"  [{i}] {s['filename']} | Page {s['page']} | {s['score_pct']}% similarity")
