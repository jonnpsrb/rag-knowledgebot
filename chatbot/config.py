from common.config import get_required_env, get_optional_env
from common.prompt_loader import load_prompt_from_file

# Qdrant Configuration
QDRANT_URL = get_required_env("QDRANT_URL")
QDRANT_API_KEY = get_required_env("QDRANT_API_KEY")
QDRANT_COLLECTION = get_optional_env("QDRANT_COLLECTION", "knowledge_base")

# Embedding Model Configuration
EMBEDDING_MODEL = get_optional_env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Groq LLM Configuration
GROQ_API_KEY = get_required_env("GROQ_API_KEY")
LLM_MODEL_NAME = get_optional_env("LLM_MODEL_NAME", "openai/gpt-oss-120b")

# Retrieval Configuration
RETRIEVAL_TOP_K = int(get_optional_env("RETRIEVAL_TOP_K", "5"))
RETRIEVAL_SCORE_THRESHOLD = float(get_optional_env("RETRIEVAL_SCORE_THRESHOLD", "0.5"))

# Chatbot UI Configuration
BOT_NAME = get_optional_env("BOT_NAME", "Accurate Assistant")
BOT_GREETING_MESSAGE = get_optional_env("BOT_GREETING_MESSAGE", "Hai, ada yang bisa saya bantu?")
BOT_PROMPT = load_prompt_from_file()
