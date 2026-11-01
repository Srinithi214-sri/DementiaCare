# Phase 13: Semantic Memory & Vector Retrieval

This document outlines the architecture, constraints, and privacy guarantees for the Semantic Memory implementation in DementiaCare OS.

## Semantic Retrieval Architecture

The semantic retrieval layer enhances the Fusion Agent's ContextBuilder. When an event (e.g., speech or interaction) has textual content, the ContextBuilder extracts a query and attempts to perform a semantic search against the patient's approved memories.

### Local Embedding Model
We use a completely local embedding provider (`sentence-transformers` using `all-MiniLM-L6-v2`). This ensures that memory texts are never sent to external third-party APIs for embedding generation, preserving total patient privacy.

### Text Normalization
Memories are converted to text using a strict deterministic normalizer (`memory_embedding_service.py`) which ensures that:
- Caregiver IDs, JWTs, and internal IDs are stripped.
- Raw media (images/audio URLs) are completely excluded.
- Only the title, content, category, and date are embedded.

### Indexing and Storage
Vector embeddings are generated asynchronously when a memory is created or updated, and stored directly on the memory document in MongoDB.
**CRITICAL RULE:** Unapproved or inactive memories are immediately un-indexed and their embeddings are discarded.

### Retrieval and Patient Isolation
Semantic retrieval explicitly builds the boundary into the query *before* calculating similarity:
1. `SemanticMemoryService` fetches only memories matching `{"patient_id": patient_id, "approved": True, "active": True}`.
2. Cosine similarity is computed locally in-memory.
3. The Top-K memories scoring above a conservative threshold are returned.

This guarantees that:
- Cross-patient leaks are structurally impossible at the database retrieval level.
- Unapproved memories cannot slip into the context.

### Deterministic Fallback
If the embedding model fails, the query is invalid, or the database is unreachable, the system safely and deterministically falls back to the previous behavior: fetching the Top 20 most recent approved/active memories. This prevents a denial-of-service or failing open.

### Gemini LLM Guardrails
- Gemini is only provided the memories output by `ContextBuilder`.
- Gemini never queries MongoDB directly.
- The `FusionAgent` explicitly validates memory IDs hallucinated by Gemini in the resulting `FusionDecision`. If Gemini returns an action requiring a memory ID that was not provided in the bounded context, the action is safely stripped.

## Known Limitations and Future Migrations
- **In-Memory Similarity:** The current cosine similarity is computed in memory for deterministic behavior locally across all development environments. This is performant given the strict bounded nature of patient memories, but if scaling to thousands of memories per patient, migration to MongoDB Atlas Vector Search (using `$vectorSearch`) is straightforward given the embeddings are already stored.
- **Lazy Loading:** `sentence-transformers` is loaded lazily on the first request to minimize boot times.
