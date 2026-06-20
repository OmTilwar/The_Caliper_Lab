"""
Caliper Lab Pipeline Configuration
All tunable parameters in one place.
"""

# ─── Target Company ──────────────────────────────────────────────
TARGET_COMPANY_CIK = "0001045810"   # NVIDIA
TARGET_COMPANY_TICKER = "NVDA"
SEC_USER_AGENT = "CaliperLab research@caliperlab.com"

# ─── LLM Model ───────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"

# OpenRouter (used when OPENROUTER_API_KEY is set in .env)
OPENROUTER_MODEL = "openai/gpt-4o-mini"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ─── Chunking ────────────────────────────────────────────────────
CHUNK_SIZE_TOKENS = 2000            # Approximate tokens per sub-chunk
CHUNK_OVERLAP_TOKENS = 200          # Overlap between adjacent sub-chunks
MIN_CHUNK_SIZE_TOKENS = 300         # Discard chunks smaller than this

# ─── Generation ──────────────────────────────────────────────────
MIN_QA_PAIRS = 100                  # Minimum verified QA pairs to produce
QA_PER_CHUNK_MIN = 5                # Min questions per generation call
QA_PER_CHUNK_MAX = 8                # Max questions per generation call
GENERATION_TEMPERATURE = 0.3        # Higher = more creative questions

# ─── Verification ────────────────────────────────────────────────
VERIFICATION_BATCH_SIZE = 5         # QA pairs per verification call
VERIFICATION_TEMPERATURE = 0.0      # Deterministic verification

# ─── Deduplication ───────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.85         # Cosine similarity threshold for dedup

# ─── Rate Limiting ───────────────────────────────────────────────
GEMINI_RPM = 6                      # Requests per minute (conservative; limit ~10)
OPENROUTER_RPM = 18                 # OpenRouter allows 20 RPM
MAX_RETRIES = 3                     # Max retries on API errors
RETRY_BASE_DELAY = 10               # Base delay in seconds for exponential backoff

# ─── Paths ───────────────────────────────────────────────────────
DATA_DIR = "data"
DATASET_DIR = "dataset"
CHECKPOINT_DIR = "data/checkpoints"
