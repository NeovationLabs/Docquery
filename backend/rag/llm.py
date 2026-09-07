"""Loads the GGUF model ONCE and reuses it for every request.
This is the single biggest speed fix vs. reloading the model per-call.
"""
import json
import re
from llama_cpp import Llama

from .. import config

_llm = None


def get_llm() -> Llama:
    global _llm
    if _llm is None:
        print(f"[llm] Loading model from {config.MODEL_PATH} ...")
        _llm = Llama(
            model_path=config.MODEL_PATH,
            n_ctx=config.N_CTX,
            n_threads=config.N_THREADS,
            n_batch=config.N_BATCH,
            n_gpu_layers=config.N_GPU_LAYERS,
            verbose=False,
        )
        print("[llm] Model loaded and cached for reuse.")
    return _llm


def warmup():
    """Call once at server startup so the first real request isn't the slow one."""
    llm = get_llm()
    llm.create_chat_completion(
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=8,
    )
    print("[llm] Warmup complete.")


def answer_question(question: str, context_chunks) -> str:
    llm = get_llm()
    context = "\n\n---\n\n".join(context_chunks)
    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the "
        "provided document context. If the answer isn't in the context, say "
        "you don't know. Be concise."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=config.ANSWER_MAX_TOKENS,
        temperature=0.3,
    )
    return response["choices"][0]["message"]["content"].strip()


def generate_suggested_questions(sample_chunks, n: int = None) -> list:
    """Ask the LLM to propose questions a user could ask about this document."""
    n = n or config.NUM_SUGGESTED_QUESTIONS
    llm = get_llm()
    context = "\n\n---\n\n".join(sample_chunks)

    system_prompt = (
        "You generate short, specific questions a reader might ask about a document, "
        "based only on excerpts shown to you. Respond with ONLY a JSON array of strings, "
        "no preamble, no markdown fences."
    )
    user_prompt = (
        f"Document excerpts:\n{context}\n\n"
        f"Generate exactly {n} distinct, specific questions a curious reader could ask "
        f"about this document. Return them as a JSON array of strings."
    )

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=config.SUGGESTED_Q_MAX_TOKENS,
        temperature=0.6,
    )
    raw = response["choices"][0]["message"]["content"].strip()
    return _parse_question_list(raw, n)


def _parse_question_list(raw: str, n: int) -> list:
    # Strip markdown fences if the model added them anyway.
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(q).strip() for q in parsed if str(q).strip()][:n]
    except json.JSONDecodeError:
        pass
    # Fallback: split on newlines / numbered list markers
    lines = [re.sub(r"^[\d\.\)\-\*\s]+", "", l).strip(' "') for l in cleaned.splitlines()]
    return [l for l in lines if l][:n]
