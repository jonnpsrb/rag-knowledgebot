import qdrant_client
from qdrant_client.http import models
from .config import get_required_env, get_optional_env

class QdrantClient:
    """
    Centralized Qdrant client manager with common operations
    """
    
    def __init__(self):
        self.url = get_required_env("QDRANT_URL")
        self.api_key = get_required_env("QDRANT_API_KEY")
        self.client = qdrant_client.QdrantClient(
            url=self.url,
            api_key=self.api_key
        )
    
    def delete_collection(self, collection_name: str):
        """
        Delete a collection from Qdrant
        """
        try:
            self.client.delete_collection(collection_name=collection_name)
            print(f"Collection '{collection_name}' deleted successfully.")
            return True
        except Exception as e:
            print(f"Error deleting collection '{collection_name}': {e}")
            return False
    
    def create_collection(self, collection_name: str, vector_size: int = 384):
        """
        Create a new collection in Qdrant with necessary indexes
        """
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, 
                    distance=models.Distance.COSINE # de-facto choice for RAG systems
                ),
            )
            
            # Create indexes for filtering fields in nested metadata
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.sha256",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.type",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.url",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            print(f"Collection '{collection_name}' created successfully with indexes.")
            return True
        except Exception as e:
            print(f"Error creating collection '{collection_name}': {e}")
            return False
    
    def get_client(self):
        """
        Get the underlying Qdrant client for direct operations
        """
        return self.client
    
    def check_collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists
        """
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            return collection_name in collection_names
        except Exception as e:
            print(f"Error checking collection existence: {e}")
            return False
    
    def upsert_points(self, collection_name: str, points, wait: bool = True):
        """
        Upsert points to a collection
        """
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=wait,
            )
            return True
        except Exception as e:
            print(f"Error upserting points to '{collection_name}': {e}")
            return False
    
    def scroll_collection(self, collection_name: str, scroll_filter=None, limit: int = 10):
        """
        Scroll through collection points
        """
        try:
            response = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
            )
            return response
        except Exception as e:
            print(f"Error scrolling collection '{collection_name}': {e}")
            return None, None
    
    def delete_points(self, collection_name: str, points_selector):
        """
        Delete points from collection using selector
        """
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=points_selector,
            )
            return True
        except Exception as e:
            print(f"Error deleting points from '{collection_name}': {e}")
            return False
