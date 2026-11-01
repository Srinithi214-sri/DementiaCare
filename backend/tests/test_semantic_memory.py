import pytest
from unittest.mock import patch, MagicMock
from services.semantic_memory_service import SemanticMemoryService
from services.memory_index_service import MemoryIndexService
from services.memory_embedding_service import normalize_memory_text
from bson import ObjectId

@pytest.fixture
def memory_doc():
    return {
        "_id": ObjectId(),
        "patient_id": "test_patient_1",
        "title": "Beach Trip",
        "content": "Went to the beach with family.",
        "category": "family",
        "date_reference": "Summer 2020",
        "approved": True,
        "active": True
    }

def test_normalize_memory_text(memory_doc):
    text = normalize_memory_text(memory_doc)
    assert "Title: Beach Trip" in text
    assert "Content: Went to the beach" in text
    assert "Category: family" in text
    assert "Date: Summer 2020" in text
    
def test_normalize_memory_text_strips_media():
    memory = {
        "title": "Secret",
        "content": "A secret memory",
        "image_url": "http://example.com/image.png",
        "caregiver_id": "cg_123"
    }
    text = normalize_memory_text(memory)
    assert "image_url" not in text
    assert "caregiver_id" not in text
    
def test_semantic_memory_isolation():
    # Setup mock collection
    mock_collection = MagicMock()
    # Memory 1 belongs to patient 1, Memory 2 belongs to patient 2
    mock_collection.find.return_value = [
        {"_id": ObjectId(), "patient_id": "test_patient_1", "title": "Mem 1", "content": "Text", "embedding": [0.1, 0.2]},
        {"_id": ObjectId(), "patient_id": "test_patient_2", "title": "Mem 2", "content": "Text", "embedding": [0.1, 0.2]}
    ]
    
    with patch("services.semantic_memory_service.db") as mock_db, \
         patch("services.semantic_memory_service.get_embedding_provider") as mock_provider:
        
        mock_db.get_memories_collection.return_value = mock_collection
        mock_provider_instance = MagicMock()
        mock_provider_instance.embed.return_value = [0.1, 0.2]
        mock_provider.return_value = mock_provider_instance
        
        # Act
        SemanticMemoryService.search_memories("test_patient_1", "Query")
        
        # Assert patient isolation explicitly requested at DB level
        mock_collection.find.assert_called_with({
            "patient_id": "test_patient_1",
            "approved": True,
            "active": True
        })

def test_semantic_memory_fallback_on_embedding_failure():
    with patch("services.semantic_memory_service.get_embedding_provider") as mock_provider:
        mock_provider_instance = MagicMock()
        mock_provider_instance.embed.side_effect = Exception("Model offline")
        mock_provider.return_value = mock_provider_instance
        
        results = SemanticMemoryService.search_memories("test_patient_1", "Query")
        
        # Should gracefully return empty list, allowing ContextBuilder to fallback
        assert results == []

def test_semantic_memory_respects_limits():
    mock_collection = MagicMock()
    # Generate 10 memories
    memories = [
        {"_id": ObjectId(), "patient_id": "test_patient_1", "title": f"Mem {i}", "content": "Text", "embedding": [0.1, 0.2]}
        for i in range(10)
    ]
    mock_collection.find.return_value = memories
    
    with patch("services.semantic_memory_service.db") as mock_db, \
         patch("services.semantic_memory_service.get_embedding_provider") as mock_provider:
        
        mock_db.get_memories_collection.return_value = mock_collection
        mock_provider_instance = MagicMock()
        mock_provider_instance.embed.return_value = [0.1, 0.2] # Perfect match
        mock_provider.return_value = mock_provider_instance
        
        # Act
        results = SemanticMemoryService.search_memories("test_patient_1", "Query", limit=3)
        
        # Assert
        assert len(results) == 3

def test_semantic_memory_rejects_empty_query():
    results = SemanticMemoryService.search_memories("test_patient_1", "   ")
    assert results == []

def test_memory_index_service_ignores_unapproved():
    mock_collection = MagicMock()
    # Unapproved memory
    memory = {"_id": ObjectId(), "patient_id": "test_patient_1", "title": "Bad Mem", "approved": False, "active": True}
    mock_collection.find_one.return_value = memory
    
    with patch("services.memory_index_service.db") as mock_db:
        mock_db.get_memories_collection.return_value = mock_collection
        
        # Act
        MemoryIndexService.index_memory("test_patient_1", str(memory["_id"]))
        
        # Assert it unset the embedding if present, instead of setting it
        mock_collection.update_one.assert_called_with(
            {"_id": memory["_id"]},
            {"$unset": {"embedding": ""}}
        )
