import logging
from bson import ObjectId
from bson.errors import InvalidId
from config.db import db
from agent.embeddings import get_embedding_provider
from services.memory_embedding_service import normalize_memory_text

class MemoryIndexService:
    @staticmethod
    def index_memory(patient_id: str, memory_id: str):
        """
        Generates and stores an embedding for the given memory IF it is active and approved.
        """
        try:
            obj_id = ObjectId(memory_id)
        except InvalidId:
            logging.error(f"Invalid memory ID format during indexing: {memory_id}")
            return False
            
        try:
            collection = db.get_memories_collection()
            memory = collection.find_one({"_id": obj_id, "patient_id": patient_id})
            
            if not memory:
                logging.warning(f"Memory {memory_id} not found for indexing.")
                return False
                
            # CRITICAL: Do not index inactive or unapproved memories
            if not memory.get("active") or not memory.get("approved"):
                logging.info(f"Memory {memory_id} is inactive or unapproved. Removing embedding if present.")
                collection.update_one(
                    {"_id": obj_id},
                    {"$unset": {"embedding": ""}}
                )
                return True
                
            text_to_embed = normalize_memory_text(memory)
            if not text_to_embed:
                logging.info(f"Memory {memory_id} has no embeddable text.")
                return False
                
            provider = get_embedding_provider()
            try:
                embedding = provider.embed(text_to_embed)
            except Exception as e:
                logging.error(f"Failed to generate embedding for memory {memory_id}: {e}")
                return False # Safe failure, doesn't crash the memory update
                
            collection.update_one(
                {"_id": obj_id},
                {"$set": {"embedding": embedding}}
            )
            logging.info(f"Successfully indexed memory {memory_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error during memory indexing: {e}")
            return False

    @staticmethod
    def remove_memory_index(patient_id: str, memory_id: str):
        """
        Removes the embedding from a memory document.
        """
        try:
            obj_id = ObjectId(memory_id)
        except InvalidId:
            return False
            
        try:
            collection = db.get_memories_collection()
            collection.update_one(
                {"_id": obj_id, "patient_id": patient_id},
                {"$unset": {"embedding": ""}}
            )
            return True
        except Exception as e:
            logging.error(f"Error removing memory index: {e}")
            return False
