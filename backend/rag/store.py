"""Embedding model + Chroma vector store, both loaded once as singletons."""
import uuid
import chromadb
from sentence_transformers import SentenceTransformer

from .. import config

_embedder = None
_chroma_client = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print(f"[store] Loading embedding model '{config.EMBED_MODEL_NAME}'...")
        _embedder = SentenceTransformer(config.EMBED_MODEL_NAME)
        print("[store] Embedding model ready.")
    return _embedder


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return _chroma_client


def get_or_create_collection(collection_name: str):
    client = get_chroma_client()
    return client.get_or_create_collection(name=collection_name)


def embed_texts(texts):
    embedder = get_embedder()
    return embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()


def add_chunks(collection_name: str, chunks, source_name: str):
    """Embed chunks and store them in the named collection. Returns chunk ids."""
    collection = get_or_create_collection(collection_name)
    embeddings = embed_texts(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]
    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    return ids


def query_similar(collection_name: str, question: str, top_k: int = None):
    top_k = top_k or config.TOP_K
    collection = get_or_create_collection(collection_name)
    q_embedding = embed_texts([question])[0]
    results = collection.query(query_embeddings=[q_embedding], n_results=top_k)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return list(zip(docs, metas))


def sample_chunks_for_overview(collection_name: str, n: int = 6):
    """Grab a spread of chunks (start/middle/end) to seed suggested-question generation."""
    collection = get_or_create_collection(collection_name)
    total = collection.count()
    if total == 0:
        return []
    all_docs = collection.get()  # small collections only; fine for this app's scale
    docs = all_docs.get("documents", [])
    if len(docs) <= n:
        return docs
    step = max(1, len(docs) // n)
    return docs[::step][:n]
