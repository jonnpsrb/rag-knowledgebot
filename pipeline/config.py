from common.config import get_required_env, get_optional_env

QDRANT_URL = get_required_env("QDRANT_URL")
QDRANT_API_KEY = get_required_env("QDRANT_API_KEY")
QDRANT_COLLECTION = get_optional_env("QDRANT_COLLECTION", "knowledge_base")

EMBEDDING_MODEL = get_optional_env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_SIZE = int(get_optional_env("VECTOR_SIZE", 384))
CHUNK_SIZE = int(get_optional_env("CHUNK_SIZE", 400))
CHUNK_OVERLAP = int(get_optional_env("CHUNK_OVERLAP", 100))
