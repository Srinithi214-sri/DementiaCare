import logging
from abc import ABC, abstractmethod
from typing import List
from config.settings import settings

class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        pass

class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic fallback for tests and failure scenarios."""
    def embed(self, text: str) -> List[float]:
        # Return a deterministic vector based on text length to allow similarity to function
        # Normally mock embeddings are just random or zero, but deterministic pseudo-random is better
        import hashlib
        h = hashlib.sha256(text.encode('utf-8')).digest()
        # Use first 32 bytes as floats normalized between -1 and 1
        vec = [(b / 127.5) - 1.0 for b in h]
        return vec

class LocalEmbeddingProvider(EmbeddingProvider):
    """Uses sentence-transformers locally."""
    def __init__(self):
        self.model = None
        
    def _load_model(self):
        if self.model is None:
            logging.info(f"Loading local embedding model: {settings.EMBEDDING_MODEL}")
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
                logging.info("Embedding model loaded successfully.")
            except Exception as e:
                logging.error(f"Failed to load sentence-transformers model: {e}")
                self.model = False # Mark as failed to avoid repeated loading attempts
                
    def embed(self, text: str) -> List[float]:
        if self.model is False:
            raise RuntimeError("Local embedding model failed to initialize.")
            
        self._load_model()
        
        if self.model is False:
            raise RuntimeError("Local embedding model failed to initialize.")
            
        try:
            # sentence-transformers encode returns a numpy array, we want a list of floats
            embeddings = self.model.encode([text])[0]
            return embeddings.tolist()
        except Exception as e:
            logging.error(f"Failed to generate embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")

# Singleton provider instance
_provider_instance = None

def get_embedding_provider() -> EmbeddingProvider:
    global _provider_instance
    if _provider_instance is None:
        provider_type = settings.EMBEDDING_PROVIDER.lower()
        if provider_type == "mock":
            _provider_instance = MockEmbeddingProvider()
        else:
            _provider_instance = LocalEmbeddingProvider()
    return _provider_instance
