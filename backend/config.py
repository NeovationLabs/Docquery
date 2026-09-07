import os
import multiprocessing

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "..", "storage")
CHROMA_DIR = os.path.join(STORAGE_DIR, "chroma")
UPLOAD_DIR = os.path.join(STORAGE_DIR, "uploads")

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---- LLM model ----
# Point this at your GGUF file. Update the path for your machine.
MODEL_PATH = os.environ.get(
    "LLM_MODEL_PATH",
    os.path.join(BASE_DIR, "..", "Meta-Llama-3-8B-Instruct-IQ3_M.gguf"),
)

# Physical cores tend to give better throughput than logical/hyperthreaded count.
# Override with env var if os.cpu_count() over-reports on your machine.
N_THREADS = int(os.environ.get("LLM_N_THREADS", max(1, multiprocessing.cpu_count() // 2)))
N_CTX = int(os.environ.get("LLM_N_CTX", 3072))
N_BATCH = int(os.environ.get("LLM_N_BATCH", 512))
# Set >0 if you have a CUDA/Metal-enabled build of llama-cpp-python.
# -1 offloads all layers to GPU, 0 = CPU only.
N_GPU_LAYERS = int(os.environ.get("LLM_N_GPU_LAYERS", 0))

# ---- Embedding model ----
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

# ---- Chunking ----
CHUNK_SIZE = 900          # characters per chunk
CHUNK_OVERLAP = 150       # characters of overlap between chunks

# ---- Retrieval ----
TOP_K = 4                 # chunks retrieved per question

# ---- Generation ----
ANSWER_MAX_TOKENS = 300
SUGGESTED_Q_MAX_TOKENS = 220
NUM_SUGGESTED_QUESTIONS = 5
