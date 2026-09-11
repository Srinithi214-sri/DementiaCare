import logging
import math
from typing import List, Dict, Any
from config.db import db
from config.settings import settings
from agent.embeddings import get_embedding_provider
from services.memory_service import MemoryService

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot_product = sum(x * y for x, y in zip(v1, v2))
    norm_v1 = math.sqrt(sum(x * x for x in v1))
    norm_v2 = math.sqrt(sum(x * x for x in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

class SemanticMemoryService:
    @staticmethod
    def search_memories(
        patient_id: str,
        query_text: str,
        limit: int = None,
        min_score: float = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-K semantically relevant, approved, and active memories for a patient.
        Fallback to returning nothing if embedding fails or query is invalid (ContextBuilder will handle this).
        """
        if limit is None:
            limit = settings.SEMANTIC_TOP_K
        if min_score is None:
            min_score = settings.SEMANTIC_MIN_SCORE
            
        # Bound limits safely
        if limit <= 0:
            limit = 5
        elif limit > 20:
            limit = 20

        if not query_text or not query_text.strip():
            return []
            
        # Bound query size to prevent abuse or OOM
        MAX_QUERY_LEN = 2000
        safe_query = query_text.strip()[:MAX_QUERY_LEN]

        provider = get_embedding_provider()
        
        try:
            query_embedding = provider.embed(safe_query)
        except Exception as e:
            logging.error(f"Semantic search failed during embedding generation: {e}")
            return []

        # Enforce patient boundary explicitly at the MongoDB query level
        query = {
            "patient_id": patient_id,
            "approved": True,
            "active": True
        }
        
        try:
            collection = db.get_memories_collection()
            cursor = collection.find(query)
            
            scored_memories = []
            for doc in cursor:
                # If memory has no embedding, it can't be scored, skip it
                mem_embedding = doc.get("embedding")
                if not mem_embedding or not isinstance(mem_embedding, list):
                    continue
                    
                score = cosine_similarity(query_embedding, mem_embedding)
                if score >= min_score:
                    # Format memory safely
                    doc.pop("embedding", None) # Do not return embedding
                    formatted = MemoryService._format_memory(doc)
                    if formatted:
                        scored_memories.append((score, formatted))
                        
            # Sort by score descending
            scored_memories.sort(key=lambda x: x[0], reverse=True)
            
            # Extract top-K
            top_k_memories = [m for _, m in scored_memories[:limit]]
            logging.info(f"Semantic search for {patient_id} returned {len(top_k_memories)} memories.")
            return top_k_memories
            
        except Exception as e:
            logging.error(f"Error during semantic memory retrieval: {e}")
            return []
