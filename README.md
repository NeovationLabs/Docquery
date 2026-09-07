# Local Document Q&A (RAG) — FastAPI + llama.cpp

A local RAG app: upload a PDF/txt/md file, it gets chunked and embedded into
ChromaDB, the LLM proposes recommended questions right after ingestion, and
you can then ask your own questions grounded in the document.

## Project layout

```
ragapp/
├── backend/
│   ├── main.py          # FastAPI app: /upload, /ask, /health
│   ├── config.py        # all tunables (model path, threads, chunk size...)
│   └── rag/
│       ├── chunking.py  # PDF/text extraction + overlapping chunker
│       ├── store.py     # embeddings (sentence-transformers) + ChromaDB
│       └── llm.py       # llama-cpp-python model, loaded ONCE at startup
├── frontend/
│   └── index.html       # single-page UI (upload, suggested Qs, chat)
└── storage/             # created automatically: uploads/ + chroma/
```

## Setup

```bash
cd ragapp
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
```

Set the path to your GGUF model (or edit the default in `backend/config.py`):

```bash
export LLM_MODEL_PATH="/absolute/path/to/Meta-Llama-3-8B-Instruct-IQ3_M.gguf"
```

Optional tuning env vars (see `config.py` for defaults):

```bash
export LLM_N_THREADS=6          # set to your PHYSICAL core count
export LLM_N_GPU_LAYERS=-1      # only if your llama-cpp-python build has GPU support
export LLM_N_CTX=3072
```

## Run

From the `ragapp` directory:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The model loads once at startup (you'll see "[startup] Warming up..." in the
logs) — this is what makes per-request answers fast, since you're no longer
paying model-load time on every question.

Open **http://localhost:8000/app/** in your browser for the UI.

## API

- `POST /upload` (multipart file) → chunks the doc, indexes it, and returns
  `doc_id`, chunk count, and `suggested_questions` generated from a spread of
  chunks across the document.
- `POST /ask` `{ "doc_id": "...", "question": "..." }` → retrieves the top
  matching chunks and answers grounded in them.
- `GET /health`

## Getting under 3–4 seconds per answer

An 8B model on CPU alone will usually **not** hit 3-4s consistently for
RAG-style prompts (they include a few paragraphs of retrieved context, which
is more tokens to process than a bare question). Things that actually move
the needle, roughly in order of impact:

1. **Load the model once, not per request** — already done here via the
   `lifespan` startup hook. This was your biggest cost before.
2. **GPU offload** (`LLM_N_GPU_LAYERS`) — if you have any NVIDIA/Apple Silicon
   GPU, install a `llama-cpp-python` build with CUDA/Metal support and set
   this to `-1`. This is usually the single biggest speed lever after #1.
3. **Smaller model** — swap to Llama-3.2-3B-Instruct or Phi-3-mini (both have
   GGUF quantized versions) if CPU-only. An 8B model is noticeably slower
   than a 3B one for marginal quality gain on simple Q&A.
4. **Threads** — set `LLM_N_THREADS` to your physical core count, not logical
   core count (hyperthreads don't help llama.cpp much).
5. **Shorter answers** — lower `ANSWER_MAX_TOKENS` in `config.py` if you don't
   need long responses; generation time scales with tokens produced.
6. **Fewer/smaller retrieved chunks** — lower `TOP_K` or `CHUNK_SIZE` in
   `config.py` so less context has to be processed per question.

## Notes

- `store.sample_chunks_for_overview` pulls the whole collection via
  `collection.get()` to pick a spread of chunks for the suggested-questions
  step. That's fine for typical single-document use but would need paging
  for very large documents (thousands of chunks).
- ChromaDB is persisted to `storage/chroma/` so re-uploading isn't required
  between server restarts — each upload gets its own collection keyed by a
  generated `doc_id`.
- CORS is wide open (`allow_origins=["*"]`) for local dev convenience; lock
  this down if you ever expose this beyond localhost.
