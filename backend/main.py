import os
import re
import shutil
import uuid
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .rag import chunking, store, llm as llm_module


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the LLM and embedder ONCE, at server startup, not per-request.
    print("[startup] Warming up embedder and LLM...")
    store.get_embedder()
    llm_module.warmup()
    print("[startup] Ready.")
    yield


app = FastAPI(title="Local Document Q&A (RAG)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the simple frontend at /
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


class AskRequest(BaseModel):
    doc_id: str
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list
    elapsed_seconds: float


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int
    suggested_questions: list
    elapsed_seconds: float


def _safe_collection_name(doc_id: str) -> str:
    # Chroma collection names must be simple; doc_id is already a uuid hex.
    return f"doc_{re.sub(r'[^a-zA-Z0-9_]', '', doc_id)}"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    start = time.time()

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".pdf", ".txt", ".md"):
        raise HTTPException(400, f"Unsupported file type: {ext}")

    doc_id = uuid.uuid4().hex
    save_path = os.path.join(config.UPLOAD_DIR, f"{doc_id}{ext}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        text = chunking.extract_text(save_path)
    except Exception as e:
        raise HTTPException(400, f"Failed to extract text: {e}")

    if not text.strip():
        raise HTTPException(400, "No extractable text found in the file.")

    chunks = chunking.chunk_text(text)
    if not chunks:
        raise HTTPException(400, "Document produced no usable chunks.")

    collection_name = _safe_collection_name(doc_id)
    store.add_chunks(collection_name, chunks, source_name=file.filename)

    # Generate recommended questions from a spread of chunks right after ingest.
    sample = store.sample_chunks_for_overview(collection_name, n=6)
    try:
        suggested = llm_module.generate_suggested_questions(sample)
    except Exception as e:
        print(f"[upload] suggestion generation failed: {e}")
        suggested = []

    elapsed = round(time.time() - start, 2)
    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        num_chunks=len(chunks),
        suggested_questions=suggested,
        elapsed_seconds=elapsed,
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest):
    start = time.time()
    collection_name = _safe_collection_name(payload.doc_id)

    results = store.query_similar(collection_name, payload.question)
    if not results:
        raise HTTPException(404, "No indexed content for this doc_id. Upload it first.")

    context_chunks = [doc for doc, _meta in results]
    sources = [meta for _doc, meta in results]

    answer = llm_module.answer_question(payload.question, context_chunks)
    elapsed = round(time.time() - start, 2)

    return AskResponse(answer=answer, sources=sources, elapsed_seconds=elapsed)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
